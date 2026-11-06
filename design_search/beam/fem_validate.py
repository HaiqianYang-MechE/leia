#!/usr/bin/env python3
"""FEM validation of beam search winner.

Runs ground-truth FEM characterization on the winner topology and computes
F-D curves for comparison with LEIA predictions.

Usage:
    python -m design_search.beam.fem_validate \
        --beam-results design_search/outputs/full_001/results.json \
        --output-dir design_search/outputs/full_001/fem_validation
"""

import argparse
import json
import os
import time

import numpy as np

from design_search.core.config import ServingConfig
from design_search.core.fem import FEMWorkerPool, SCORED_DIRECTIONS
from design_search.core.meshing import generate_plate_mesh, MeshingError
from leia.regime import MICROPLATE_LATTICE
from leia.stress import (
    tet4_dNdx,
    boundary_reaction_force,
    identify_right_boundary_nodes,
    precompute_boundary_geometry,
)


def compute_fem_fd_curves(fem_result, coords, topo, speeds):
    """Compute F-D curves from FEM displacement fields.

    For each direction, computes reaction forces at each step via
    leia.stress.boundary_reaction_force (boundary-tet-only assembly).
    """
    n_nodes = len(coords)
    right_ids, _ = identify_right_boundary_nodes(coords)
    cy = (coords[:, 1].min() + coords[:, 1].max()) / 2.0
    cz = (coords[:, 2].min() + coords[:, 2].max()) / 2.0

    dNdx_all, volumes_all = tet4_dNdx(coords, topo)
    bdy_geom = precompute_boundary_geometry(topo, dNdx_all, volumes_all, right_ids)

    mu = MICROPLATE_LATTICE.mu
    lmbda = MICROPLATE_LATTICE.lmbda

    direction_map = {
        "stretch": (0, [1, 0, 0, 0]),
        "twist": (1, [0, 1, 0, 0]),
        "shear_y": (2, [0, 0, 1, 0]),
        "shear_z": (3, [0, 0, 0, 1]),
    }

    fd_curves = {}
    for dir_label, dir_data in fem_result["directions"].items():
        dof_idx, action = direction_map[dir_label]
        outputs = dir_data["outputs"]  # (T+1, N, 3) — frame 0 is rest
        n_steps = len(outputs) - 1

        disp_list = [0.0]
        force_list = [0.0]

        cum_bc = np.zeros(4, dtype=np.float64)
        action_arr = np.array(action, dtype=np.float64)
        speeds_arr = np.array(speeds, dtype=np.float64)
        for t in range(1, n_steps + 1):
            cum_bc += action_arr * speeds_arr
            gen_force = boundary_reaction_force(
                outputs[t], coords, n_nodes, right_ids, cy, cz, bdy_geom,
                mu=mu, lmbda=lmbda,
            )
            disp_list.append(float(cum_bc[dof_idx]))
            force_list.append(float(gen_force[dof_idx]))

        fd_curves[dir_label] = {
            "disp": disp_list,
            "force": force_list,
        }

    return fd_curves


def run_fem_on_graph(graph, label, config, n_steps, pool=None):
    """Mesh a graph and run FEM characterization. Returns (fem_fd, fem_time, mesh).

    If pool is provided, uses the shared pool. Otherwise creates a temporary one.
    """
    from design_search.core.objectives import volume_fraction
    vf = volume_fraction(graph)
    print(f"\n--- FEM: {label} (vol_frac={vf:.6f}) ---")

    print(f"  Meshing...")
    try:
        mesh = generate_plate_mesh(graph, config)
    except MeshingError as e:
        print(f"  FEM SKIP: mesh failed ({e})")
        return {"status": "mesh_failed", "error": str(e)[:200]}, 0.0, None
    print(f"  {mesh.points.shape[0]} nodes, {mesh.cells.shape[0]} tets")

    print(f"  Running FEM ({n_steps} steps per direction)...")
    t0 = time.time()
    own_pool = pool is None
    if own_pool:
        pool = FEMWorkerPool(max_workers=1, mu=config.mu, lmbda=config.lmbda,
                             speeds=config.speeds)
    future = pool.submit(label, mesh.points, mesh.cells, n_steps=n_steps,
                         directions=SCORED_DIRECTIONS)   # the two loadings the score uses
    fem_result = future.result()
    if own_pool:
        pool.shutdown()
    fem_time = time.time() - t0

    print(f"  FEM completed in {fem_time:.1f}s ({fem_result['status']})")
    if fem_result["errors"]:
        print(f"  Errors: {fem_result['errors']}")

    print(f"  Computing F-D curves...")
    fem_fd = compute_fem_fd_curves(fem_result, mesh.points, mesh.cells, config.speeds)

    for dir_label, data in fem_fd.items():
        d, f = data["disp"], data["force"]
        print(f"    {dir_label}: disp=[{d[0]:.3f}..{d[-1]:.3f}], "
              f"force=[{f[0]:.3f}..{f[-1]:.3f}]")

    return fem_fd, fem_time, mesh


def best_candidate_at_iteration(all_candidates, iteration):
    """Return the highest-scoring candidate at or before this iteration, or
    None if the search never reached it (so a short run does not repeat its
    global best under labels of iterations that did not happen)."""
    if not all_candidates or iteration > max(c["iteration"] for c in all_candidates):
        return None
    eligible = [c for c in all_candidates if c["iteration"] <= iteration]
    if not eligible:
        return None
    return max(eligible, key=lambda c: c["score"])


ALL_ITERS = [-1] + list(range(10))  # iter -1 = raw seed; 0..9 = best-per-iter


def main():
    parser = argparse.ArgumentParser(
        description="FEM validation of the seed and the best candidate at every beam-search iteration")
    parser.add_argument("--beam-results", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--mesh-res", type=int, default=None,
                        help="Override ServingConfig.mesh_res (default: serving res=10)")
    args = parser.parse_args()

    from design_search.core.objectives import anisotropy_score

    with open(args.beam_results) as f:
        beam_data = json.load(f)

    all_cands = beam_data["all_candidates"]

    # Use the same scoring formula the search used (persisted in results.json).
    # Falls back to 'product' for legacy results that predate this field.
    bp = beam_data.get("params", {})
    score_formula = bp.get("score_formula", "product")
    # iter -1 = raw seed; 0..D-1 = best-so-far at each iteration of this run.
    iters = [-1] + list(range(int(bp.get("D", len(ALL_ITERS) - 1))))
    lambda_vf = bp.get("lambda_vf", 100.0)
    print(f"[score]      formula={score_formula}, lambda_vf={lambda_vf}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.beam_results), "fem_validation")
    os.makedirs(output_dir, exist_ok=True)

    config = ServingConfig()
    if getattr(args, "mesh_res", None) is not None:
        config.mesh_res = args.mesh_res
        print(f"[override] mesh_res = {config.mesh_res}")

    # Collect best candidate per iteration, deduplicating identical graphs
    snapshots = []
    for it in iters:
        cand = best_candidate_at_iteration(all_cands, it)
        if cand is not None:
            snapshots.append((it, cand))

    # Deduplicate: if consecutive iterations have the same score (same candidate),
    # only run FEM once and reuse the result
    unique_runs = {}  # score -> (it, cand) for FEM
    iter_to_score_key = {}  # it -> score (for result lookup)
    for it, cand in snapshots:
        score_key = round(cand["score"], 6)
        iter_to_score_key[it] = score_key
        if score_key not in unique_runs:
            unique_runs[score_key] = (it, cand)

    print(f"{len(snapshots)} iterations, {len(unique_runs)} unique candidates to FEM")

    # Mesh unique candidates — skip ones that fail to mesh
    print(f"Meshing {len(unique_runs)} unique candidates...")
    meshes = {}
    for score_key, (it, cand) in unique_runs.items():
        print(f"  iter {it} (score {cand['score']:.1f}): meshing...")
        try:
            meshes[score_key] = generate_plate_mesh(cand["graph"], config)
        except MeshingError as e:
            print(f"    SKIP iter {it}: mesh failed ({e})")
            continue
        print(f"    {meshes[score_key].points.shape[0]} nodes, "
              f"{meshes[score_key].cells.shape[0]} tets")

    # Run all FEM jobs in parallel (only on candidates that meshed)
    print(f"\nSubmitting {len(meshes)} FEM jobs in parallel...")
    t0_all = time.time()
    pool = FEMWorkerPool(max_workers=max(len(meshes), 1), mu=config.mu,
                         lmbda=config.lmbda, speeds=config.speeds)
    futures = {}
    for score_key, mesh in meshes.items():
        it, cand = unique_runs[score_key]
        future = pool.submit(f"iter{it}", mesh.points, mesh.cells, n_steps=args.n_steps,
                              directions=SCORED_DIRECTIONS)
        futures[score_key] = (future, cand, mesh)

    # Collect unique FEM results — tolerant to per-plate worker crashes
    fem_results_by_key = {}
    for score_key, (future, cand, mesh) in futures.items():
        it = unique_runs[score_key][0]
        print(f"  Waiting for iter {it} (score {score_key})...")
        try:
            fem_result = future.result()
            fem_time = fem_result["time_seconds"]
            print(f"    FEM completed in {fem_time:.1f}s ({fem_result['status']})")
            fem_fd = compute_fem_fd_curves(
                fem_result, mesh.points, mesh.cells, config.speeds)
            fem_score = anisotropy_score(
                cand["graph"], fem_fd,
                formula=score_formula, lambda_vf=lambda_vf,
            )
            fem_results_by_key[score_key] = {
                "fem_fd": fem_fd,
                "fem_score": fem_score,
                "fem_time_s": fem_time,
                "fem_status": fem_result["status"],
            }
        except Exception as e:
            print(f"    [WARN] FEM crashed for iter {it} (score {score_key}): "
                  f"{type(e).__name__}: {e}")
            fem_results_by_key[score_key] = {
                "fem_fd": {},
                "fem_score": 0.0,
                "fem_time_s": 0.0,
                "fem_status": "error",
            }

    pool.shutdown()
    total_fem_time = time.time() - t0_all
    print(f"All FEM jobs completed in {total_fem_time:.1f}s (wall-clock, parallel)")

    # Map back to all iterations
    fem_points = []
    all_fem_data = {}
    for it, cand in snapshots:
        score_key = iter_to_score_key[it]
        r = fem_results_by_key.get(score_key)
        if r is None:   # meshing failed for this snapshot: keep the row, mark it
            r = {"fem_score": 0.0, "fem_time_s": 0.0, "fem_fd": {},
                 "fem_status": "mesh_failed"}
        fem_points.append({
            "iteration": it,
            "surrogate_score": cand["score"],
            "fem_score": r["fem_score"],
            "fem_time_s": r["fem_time_s"],
            "vol_frac": cand["vol_frac"],
        })
        all_fem_data[f"iter{it}"] = {
            "fd_curves": r["fem_fd"],
            "fem_time_s": r["fem_time_s"],
            "fem_score": r["fem_score"],
            "fem_status": r.get("fem_status", "unknown"),
            "surrogate_score": cand["score"],
            "vol_frac": cand["vol_frac"],
        }

    # Print comparison table
    print(f"\n{'='*65}")
    print(f"{'Iteration':>10} {'Surrogate':>12} {'FEM':>12} {'FEM time':>10}")
    print(f"{'='*65}")
    for p in fem_points:
        it_label = "seed" if p["iteration"] == -1 else f"iter {p['iteration']}"
        print(f"{it_label:>10} {p['surrogate_score']:>12.2f} {p['fem_score']:>12.2f} "
              f"{p['fem_time_s']:>9.1f}s")
    print(f"{'='*65}")

    seed_fem = fem_points[0]["fem_score"]
    winner_fem = fem_points[-1]["fem_score"]
    seed_surr = fem_points[0]["surrogate_score"]
    winner_surr = fem_points[-1]["surrogate_score"]
    if seed_surr > 0:
        print(f"Surrogate improvement: {winner_surr/seed_surr:.1f}x")
    if seed_fem > 0:
        print(f"FEM improvement:       {winner_fem/seed_fem:.1f}x")
        print(f"FEM confirms winner is better: {winner_fem > seed_fem}")
    else:
        print(f"FEM improvement:       N/A (raw seed FEM failed)")

    # Save results — only count successful FEM runs (status=='ok') in the avg.
    # Using status (not time>0) so 'partial' runs don't pollute the average.
    ok_times = [
        d["fem_time_s"] for k, d in all_fem_data.items()
        if d.get("fem_status") == "ok" and d.get("fem_time_s", 0) > 0
    ]
    avg_fem_time = float(np.mean(ok_times)) if ok_times else 0.0
    n_candidates = len(all_cands)
    results = {
        "fem_time_per_plate_s": avg_fem_time,
        "n_ok_runs": len(ok_times),
        "n_steps": args.n_steps,
        "fem_points": fem_points,
        **all_fem_data,
        # Keys read by beam/figure.py: seed = iter -1 (raw), winner = last iter
        "seed": all_fem_data.get("iter-1", {}),
        "winner": all_fem_data.get(f"iter{iters[-1]}", {}),
    }

    results_path = os.path.join(output_dir, "fem_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    if avg_fem_time > 0:
        extrapolated_h = avg_fem_time * n_candidates / 3600
        print(f"Avg FEM time per plate: {avg_fem_time:.1f}s "
              f"(n={len(ok_times)} 'ok' runs) — "
              f"serial extrapolation across {n_candidates} candidates: "
              f"{extrapolated_h:.1f}h")
    else:
        print(f"Avg FEM time per plate: N/A (no 'ok' runs)")


if __name__ == "__main__":
    main()
