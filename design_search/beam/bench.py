#!/usr/bin/env python3
"""Per-candidate isolated timing benchmark for LEIA inference vs FEM solve.

Both sides characterize the same two loadings the anisotropy score uses
(stretch and shear_y): two 20-step LEIA rollouts against two 20-step FEM solves.

Stage 3 of the Figure 4 pipeline (figure4.sh). Run AFTER beam search and FEM
validation are complete so the GPU is idle and there is no contention.

Samples N_best (best-per-iteration, iter -1..D-1) + N_random candidates from the
beam search results, then for each candidate sequentially:

  1. Mesh (timed but excluded from comparison — both methods need a mesh)
  2. LEIA: encode + 2-direction × 20-step characterization, batch_size=1
  3. FEM:  solve_characterization called directly (no pool), single worker
`--fem-from-stage2` (opt-in; figure4.sh sets it only with
DESIGN_SEARCH_BENCH_REUSE_FEM=1) copies the validation stage's parallel-pool
times for the best-per-iteration plates instead of re-solving them; the output's
`fem_source` records it and panel (d) prints a note.

All timings are wall-clock with `torch.cuda.synchronize()` around the LEIA
section. Mesh time is recorded for transparency but not part of the per-
candidate cost claim. FEM failures are tolerated and recorded with status.

Usage:
    python -m design_search.beam.bench \\
        --beam-results design_search/outputs/<run>/results.json \\
        --output-dir   design_search/outputs/<run>/bench \\
        --n-random 20 --random-seed 42 --steps 20
"""

import argparse
import json
import os
import time
import traceback

import numpy as np
import torch

from design_search.core.config import ServingConfig
from design_search.core.engine import ModelEngine
from design_search.core.meshing import generate_plate_mesh, MeshingError
from design_search.core.fem import solve_characterization, SCORED_DIRECTIONS

from design_search.core.characterize import sync_characterize, DIRECTIONS_2
from design_search.core.objectives import volume_fraction
from design_search.beam.fem_validate import best_candidate_at_iteration


def build_benchmark_set(all_candidates, max_iter, n_random, random_seed):
    """Pick 11 best-per-iter (iter -1..max_iter) + n_random samples.

    Returns list of (label, source, candidate_dict). Skips overlaps so the
    same candidate is never measured twice.
    """
    seen_score_keys = set()
    bench = []

    # 1) Best-per-iter, iter -1 (raw seed) through max_iter (winner)
    for it in [-1] + list(range(max_iter + 1)):
        cand = best_candidate_at_iteration(all_candidates, it)
        if cand is None:
            continue
        score_key = round(cand["score"], 6)
        if score_key in seen_score_keys:
            continue
        label = "iter-1_seed" if it == -1 else f"iter{it}_best"
        bench.append((label, "best", cand))
        seen_score_keys.add(score_key)

    # 2) Random sample
    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(len(all_candidates))
    n_added = 0
    for i in indices:
        if n_added >= n_random:
            break
        cand = all_candidates[i]
        score_key = round(cand["score"], 6)
        if score_key in seen_score_keys:
            continue
        label = f"rand_idx{i}"
        bench.append((label, "random", cand))
        seen_score_keys.add(score_key)
        n_added += 1

    return bench


def time_leia_serial(engine, mesh, steps):
    """Time encode + 2-direction characterization with batch_size=1.

    Returns (elapsed_s, recon_error). Caller is responsible for cleanup.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    session = engine.create_session(mesh)
    fd = sync_characterize(engine, [session], DIRECTIONS_2,
                           steps_per_direction=steps, batch_size=1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    recon = float(engine.compute_recon_error(session))

    # Cleanup
    session.prevq = None
    if hasattr(session, "prev_state"):
        session.prev_state = None
    torch.cuda.empty_cache()

    return elapsed, recon, fd


def time_fem_serial(label, mesh, mu, lmbda, speeds, steps):
    """Run FEM serially via direct call to solve_characterization. Tolerant.

    Returns (elapsed_s, status, errors_dict).
    """
    task = {
        "session_id": label,
        "points": mesh.points,
        "cells": mesh.cells,
        "mu": float(mu),
        "lmbda": float(lmbda),
        "n_steps": int(steps),
        "speeds": tuple(speeds),
        "directions": SCORED_DIRECTIONS,   # same two loadings LEIA is timed on
    }
    try:
        result = solve_characterization(task)
        return (
            float(result.get("time_seconds", 0.0)),
            str(result.get("status", "unknown")),
            result.get("errors", {}),
        )
    except Exception as e:
        return (0.0, "error", {"exception": f"{type(e).__name__}: {e}"})


def run_bench(args):
    print(f"=== Per-candidate bench: n_random={args.n_random}, "
          f"random_seed={args.random_seed}, steps={args.steps} ===")
    print(f"Beam results: {args.beam_results}")
    print(f"Output:       {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.beam_results) as f:
        beam_data = json.load(f)

    all_candidates = beam_data["all_candidates"]
    params = beam_data.get("params", {})
    max_iter = int(params.get("D", 10)) - 1  # iters are 0..D-1

    bench_set = build_benchmark_set(
        all_candidates, max_iter=max_iter,
        n_random=args.n_random, random_seed=args.random_seed,
    )
    print(f"\nBenchmark set: {len(bench_set)} candidates "
          f"({sum(1 for _, src, _ in bench_set if src == 'best')} best-per-iter, "
          f"{sum(1 for _, src, _ in bench_set if src == 'random')} random)")

    # Optionally load stage-2 FEM results (skip in-process FEM solves).
    stage2_fem = None
    if args.fem_from_stage2:
        with open(args.fem_from_stage2) as f:
            stage2_fem = json.load(f)
        n_iters = sum(1 for k in stage2_fem if k.startswith("iter"))
        print(f"\n[fem-from-stage2] loaded {n_iters} per-iter FEM entries from "
              f"{args.fem_from_stage2}")
        print(f"                  in-process FEM solve will be SKIPPED for "
              f"plates with a matching iteration entry")

    config = ServingConfig()
    if getattr(args, "mesh_res", None) is not None:
        config.mesh_res = args.mesh_res
        print(f"[override] mesh_res = {config.mesh_res}")
    print(f"\n[checkpoints] tokenizer = {config.tokenizer_ckpt}")
    print(f"[checkpoints] dynamics  = {config.dynamics_ckpt}")
    print(f"[material]    mu = {config.mu}, lmbda = {config.lmbda}")
    print(f"[serving]     speeds = {config.speeds}\n")

    engine = ModelEngine(config)

    plates = []
    t_total = time.time()

    for i, (label, source, cand) in enumerate(bench_set):
        graph = cand["graph"]
        vf = volume_fraction(graph)
        print(f"\n--- [{i+1}/{len(bench_set)}] {label} "
              f"(source={source}, vf={vf:.4f}) ---")

        # Mesh (record time for transparency)
        try:
            t_mesh = time.time()
            mesh = generate_plate_mesh(graph, config)
            mesh_time_s = time.time() - t_mesh
            print(f"  mesh:  {mesh_time_s:.1f}s "
                  f"({mesh.points.shape[0]} nodes, {mesh.cells.shape[0]} tets)")
        except (MeshingError, Exception) as e:
            print(f"  [WARN] meshing failed: {type(e).__name__}: {e}")
            plates.append({
                "label": label, "from": source, "vol_frac": vf,
                "mesh_time_s": None, "leia_time_s": None,
                "fem_time_s": None, "fem_status": "skip-no-mesh",
                "recon_error": None,
            })
            continue

        # LEIA serial
        try:
            leia_time_s, recon, _ = time_leia_serial(engine, mesh, args.steps)
            print(f"  leia:  {leia_time_s:.2f}s (recon={recon:.6f})")
        except Exception as e:
            print(f"  [WARN] LEIA failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            leia_time_s = None
            recon = None

        # FEM: either copy from stage-2 parallel pool, or run serially in-process.
        # Stage-2 entries are the best-per-iteration candidates; a random plate at
        # the same iteration is a different mesh, so it is timed in-process.
        if stage2_fem is not None and source == "best":
            it = cand.get("iteration")
            stage2_entry = stage2_fem.get(f"iter{it}") if it is not None else None
            if stage2_entry is not None:
                fem_time_s = float(stage2_entry.get("fem_time_s", 0.0))
                fem_status = str(stage2_entry.get("fem_status", "unknown"))
                fem_source_tag = "parallel_pool"
                print(f"  fem:   {fem_time_s:.1f}s ({fem_status}) "
                      f"[from stage-2 iter{it}]")
            else:
                # No matching iter entry — fall back to in-process solve.
                fem_time_s, fem_status, fem_errors = time_fem_serial(
                    label, mesh, config.mu, config.lmbda, config.speeds, args.steps,
                )
                fem_source_tag = "serial_direct"
                print(f"  fem:   {fem_time_s:.1f}s ({fem_status}) "
                      f"[no stage-2 match for iter{it}; ran in-process]")
        else:
            fem_time_s, fem_status, fem_errors = time_fem_serial(
                label, mesh, config.mu, config.lmbda, config.speeds, args.steps,
            )
            fem_source_tag = "serial_direct"
            print(f"  fem:   {fem_time_s:.1f}s ({fem_status})")
            if fem_errors:
                print(f"         errors: {fem_errors}")

        plates.append({
            "label": label,
            "from": source,
            "iteration": cand.get("iteration"),
            "vol_frac": float(vf),
            "n_nodes": int(mesh.points.shape[0]),
            "n_tets": int(mesh.cells.shape[0]),
            "mesh_time_s": float(mesh_time_s),
            "leia_time_s": float(leia_time_s) if leia_time_s is not None else None,
            "fem_time_s": float(fem_time_s),
            "fem_status": fem_status,
            "fem_source": fem_source_tag,
            "recon_error": recon,
        })

        # Periodic checkpoint to disk in case of crash
        _save(args.output_dir, plates, args, partial=True)

    total_elapsed = time.time() - t_total
    print(f"\n=== Bench complete: {total_elapsed:.1f}s, {len(plates)} plates ===")

    _save(args.output_dir, plates, args, partial=False)


def _save(output_dir, plates, args, partial):
    leia_times = [p["leia_time_s"] for p in plates
                  if p.get("leia_time_s") is not None]
    # Include both 'ok' and 'partial' — partial runs measure real wall-clock
    # for a Newton solve that succeeded in some directions but diverged in
    # others. Only 'error' status (worker crash) is excluded from timing.
    fem_times = [p["fem_time_s"] for p in plates
                 if p.get("fem_status") in ("ok", "partial")
                 and p.get("fem_time_s", 0) > 0]
    mesh_times = [p["mesh_time_s"] for p in plates
                  if p.get("mesh_time_s") is not None]

    summary = {
        "leia_mean_s": float(np.mean(leia_times)) if leia_times else 0.0,
        "leia_std_s": float(np.std(leia_times)) if leia_times else 0.0,
        "leia_n": len(leia_times),
        "fem_mean_s": float(np.mean(fem_times)) if fem_times else 0.0,
        "fem_std_s": float(np.std(fem_times)) if fem_times else 0.0,
        "fem_n": len(fem_times),
        "mesh_mean_s": float(np.mean(mesh_times)) if mesh_times else 0.0,
        "mesh_std_s": float(np.std(mesh_times)) if mesh_times else 0.0,
        "mesh_n": len(mesh_times),
    }

    # Top-level fem_source tag: derive from per-plate tags. If all plates have the
    # same tag, use it; otherwise mark mixed.
    fem_sources = sorted({p["fem_source"] for p in plates if "fem_source" in p})
    if len(fem_sources) == 1:
        # Annotate parallel_pool with the n in stage-2 if known
        fem_source = fem_sources[0]
        if fem_source == "parallel_pool" and args.fem_from_stage2:
            try:
                with open(args.fem_from_stage2) as f:
                    s2 = json.load(f)
                n_iters = sum(1 for k in s2 if k.startswith("iter"))
                fem_source = f"parallel_pool_n={n_iters}"
            except Exception:
                pass
    elif len(fem_sources) > 1:
        fem_source = "mixed:" + ",".join(fem_sources)
    else:
        fem_source = "unknown"

    out = {
        "n_plates": len(plates),
        "config": {
            "steps": args.steps,
            "random_seed": args.random_seed,
            "n_random": args.n_random,
            "mesh_res": getattr(args, "mesh_res", None),
            "directions": [d[2] for d in DIRECTIONS_2],
        },
        "fem_source": fem_source,
        "plates": plates,
        "summary": summary,
        "partial": partial,
    }

    # Partial snapshots go to bench.partial.json; bench.json (figure4.sh's
    # stage-3 sentinel) appears only once the bench has completed.
    path = os.path.join(output_dir, "bench.partial.json" if partial else "bench.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, path)
    if not partial:
        partial_path = os.path.join(output_dir, "bench.partial.json")
        if os.path.exists(partial_path):
            os.remove(partial_path)
        print(f"\nSaved {path}")
        print(f"  LEIA: {summary['leia_mean_s']:.2f}±{summary['leia_std_s']:.2f}s "
              f"(n={summary['leia_n']})")
        print(f"  FEM:  {summary['fem_mean_s']:.1f}±{summary['fem_std_s']:.1f}s "
              f"(n={summary['fem_n']})")
        if summary["leia_mean_s"] > 0:
            print(f"  Ratio: {summary['fem_mean_s'] / summary['leia_mean_s']:.1f}x "
                  f"FEM/LEIA per candidate")


def main():
    parser = argparse.ArgumentParser(
        description="Per-candidate isolated timing for LEIA vs FEM")
    parser.add_argument("--beam-results", type=str, required=True,
                        help="Path to beam search results.json")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for bench.json")
    parser.add_argument("--n-random", type=int, default=20,
                        help="Number of random samples (in addition to best-per-iter)")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20,
                        help="Steps per characterization direction")
    parser.add_argument("--mesh-res", type=int, default=None,
                        help="Override ServingConfig.mesh_res (default: serving res=10)")
    parser.add_argument("--fem-from-stage2", type=str, default=None,
                        help="Opt-in shortcut (figure4.sh: DESIGN_SEARCH_BENCH_REUSE_FEM=1). "
                             "Path to fem_validation/fem_results.json. When set, "
                             "FEM solve is skipped and per-job times are copied "
                             "from stage 2's parallel pool by iteration match. "
                             "LEIA timing is still measured fresh.")
    args = parser.parse_args()

    run_bench(args)


if __name__ == "__main__":
    main()
