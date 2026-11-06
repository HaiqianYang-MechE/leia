#!/usr/bin/env python3
"""Stages 6-7 of the Figure 5 pipeline (figure5.sh): FEM ground truth for the landscape candidates.

For each candidate that already has a Stage-2 LEIA npz on disk, run FEM
stretch + shear ground truth on the same mesh and compute three
LEIA-vs-FEM comparison scalars per candidate:

  - disp_l2_err = ||u_LEIA - u_FEM|| / ||u_FEM||  (final step, per direction + mean)
  - vm_corr = Pearson corr(VM_LEIA, VM_FEM_per_node)  (per direction + mean)
  - total_work_err = |W_total_LEIA - W_total_FEM| / W_total_FEM
                     where W_total = W_stretch + W_shear

CPU-only: FEM workers (FEniCS/MUMPS) and metric computation. Training-safe.

Resumable: appends to fem_scores.jsonl, skipping IDs already present.
Submits FEM jobs in batches of `--workers` to keep result-queue memory bounded.

Usage:
    python -m design_search.landscape.fem \\
        --leia-dir design_search/outputs/<landscape_run> \\
        --workers 50 \\
        --n-steps 20

    # Pilot (only those candidates whose LEIA npz exists):
    python -m design_search.landscape.fem \\
        --leia-dir design_search/outputs/<landscape_run>_pilot \\
        --workers 8 \\
        --n-steps 20
"""

import argparse
import json
import os
import time

import numpy as np

from design_search.core.config import ServingConfig
from design_search.core.fem import FEMWorkerPool, SCORED_DIRECTIONS
from leia.regime import MICROPLATE_LATTICE

from leia.stress import (
    cauchy_from_P,
    neo_hookean_P,
    precompute_stress_geometry,
    stress_from_disps_nodal_cauchy,
    von_mises,
    tet4_dNdx,
    boundary_reaction_force,
    identify_right_boundary_nodes,
    precompute_boundary_geometry,
)

from design_search.core.objectives import compute_work


def compute_fem_fd_curves(fem_result, coords, topo, speeds):
    """Compute F-D curves from FEM displacement fields.

    For each direction, computes reaction forces at each step via
    leia.stress.boundary_reaction_force (boundary-tet-only assembly).
    Local copy of design_search/beam/fem_validate.py:compute_fem_fd_curves —
    kept here to avoid cross-pipeline imports between Figure-4 and Figure-5.
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
        outputs = dir_data["outputs"]
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


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.size != b.size or a.size < 2:
        return float("nan")
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def fem_per_node_vm_at_final(
    points: np.ndarray,
    cells: np.ndarray,
    u_final: np.ndarray,
    mu: float,
    lmbda: float,
) -> np.ndarray:
    """Compute per-node von Mises stress for FEM final-step displacement.

    Wraps `stress_from_disps_nodal_cauchy` with quality mask precomputation,
    matching the pipeline used for training-set stress GT.
    """
    dNdx, volumes, good = precompute_stress_geometry(points, cells)
    _sym6, vm_node = stress_from_disps_nodal_cauchy(
        cells, u_final, dNdx, volumes, good, mu=mu, lmbda=lmbda,
    )
    return vm_node


def already_processed_ids(jsonl_path: str) -> set[str]:
    if not os.path.exists(jsonl_path):
        return set()
    ids = set()
    with open(jsonl_path) as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "id" in obj:
                    ids.add(obj["id"])
            except json.JSONDecodeError:
                continue
    return ids


def load_leia_jsonl(path: str) -> dict:
    """Map id -> entry for entries with status=='ok'."""
    out = {}
    with open(path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("status") == "ok":
                out[obj["id"]] = obj
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leia-dir", type=str, default=None,
                        help="Output dir from `landscape.score --metric=score`; "
                             "if --run-root and --stage are set, defaults to "
                             "<root>/<stage>/.")
    parser.add_argument("--stage", type=str, default=None,
                        choices=["walk", "novelty", "heldout"],
                        help="Sub-stage to FEM-validate (when --run-root is set).")
    parser.add_argument("--workers", type=int, default=50,
                        help="FEM worker count (cap by RAM; ~10 GB/worker peak)")
    parser.add_argument("--n-steps", type=int, default=20,
                        help="FEM steps per direction. MUST match the LEIA "
                             "characterization --steps-per-direction in "
                             "`landscape.score` (default 20).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="How many FEM jobs to keep in flight at once "
                             "(bounds result-queue memory). Default: "
                             "max(100, --workers), so every worker is used.")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--mu", type=float, default=MICROPLATE_LATTICE.mu)
    parser.add_argument("--lmbda", type=float, default=MICROPLATE_LATTICE.lmbda)
    from design_search.landscape.layout import add_run_root_arg, resolve
    add_run_root_arg(parser)
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = max(100, args.workers)

    if args.leia_dir is None:
        if args.run_root is None or args.stage is None:
            parser.error("either --leia-dir or (--run-root and --stage) is required")
        args.leia_dir = resolve(args.run_root, "heldout_dir" if args.stage == "heldout" else args.stage)

    leia_jsonl = os.path.join(args.leia_dir, "leia_scores.jsonl")
    leia_outputs_dir = os.path.join(args.leia_dir, "leia_outputs")
    fem_outputs_dir = os.path.join(args.leia_dir, "fem_outputs")
    fem_jsonl = os.path.join(args.leia_dir, "fem_scores.jsonl")
    os.makedirs(fem_outputs_dir, exist_ok=True)

    # Load LEIA results
    leia_by_id = load_leia_jsonl(leia_jsonl)
    print(f"LEIA entries (status=ok): {len(leia_by_id)}")

    done = already_processed_ids(fem_jsonl)
    pending_ids = [cid for cid in leia_by_id if cid not in done]
    if args.max_candidates is not None:
        pending_ids = pending_ids[:args.max_candidates]
    print(f"FEM done: {len(done)}, pending: {len(pending_ids)}")

    if not pending_ids:
        print("Nothing to do.")
        return

    config = ServingConfig()
    speeds = config.speeds

    t_start = time.time()
    n_ok = 0
    n_fem_fail = 0
    n_total_work_fem = 0.0
    n_diverged = 0

    # Process in batches to bound result-queue memory
    with open(fem_jsonl, "a") as f_jsonl:
        for batch_start in range(0, len(pending_ids), args.batch_size):
            batch_ids = pending_ids[batch_start:batch_start + args.batch_size]
            print(f"\n=== Batch {batch_start // args.batch_size + 1} "
                  f"({len(batch_ids)} candidates) ===")

            # Spin up a fresh pool per batch (so worker memory is reclaimed)
            pool = FEMWorkerPool(
                max_workers=min(args.workers, len(batch_ids)),
                mu=args.mu, lmbda=args.lmbda, speeds=speeds,
            )

            # Submit
            submitted = []
            for cid in batch_ids:
                leia_npz_path = os.path.join(args.leia_dir,
                                             leia_by_id[cid]["leia_outputs_path"])
                try:
                    leia_npz = np.load(leia_npz_path)
                    points = leia_npz["mesh_points"].astype(np.float64)
                    cells = leia_npz["mesh_cells"].astype(np.int64)
                except Exception as e:
                    f_jsonl.write(json.dumps({
                        "id": cid, "fem_status": "leia_npz_missing",
                        "error": str(e)[:200],
                    }) + "\n")
                    f_jsonl.flush()
                    n_fem_fail += 1
                    continue

                future = pool.submit(cid, points, cells, n_steps=args.n_steps,
                                     directions=SCORED_DIRECTIONS)   # stretch + shear_y: all this stage reads
                submitted.append((cid, future, points, cells, leia_npz))

            print(f"  Submitted {len(submitted)} FEM jobs ({args.workers} workers)")

            # Collect (in submission order — straggler dominates batch time)
            for j, (cid, future, points, cells, leia_npz) in enumerate(submitted):
                try:
                    fem_result = future.result()
                except Exception as e:
                    f_jsonl.write(json.dumps({
                        "id": cid, "fem_status": "future_exception",
                        "error": str(e)[:200],
                    }) + "\n")
                    f_jsonl.flush()
                    n_fem_fail += 1
                    leia_npz.close()
                    continue

                # FEM may diverge on aggressive mutants. solve_characterization
                # returns status in {'ok','partial','failed'} and errors as
                # {direction: str}.
                fem_status = fem_result.get("status", "unknown")
                if fem_status != "ok":
                    f_jsonl.write(json.dumps({
                        "id": cid,
                        "fem_status": fem_status,
                        "fem_errors": fem_result.get("errors", {}),
                        "fem_time_s": float(fem_result.get("time_seconds", 0.0)),
                    }) + "\n")
                    f_jsonl.flush()
                    n_diverged += 1
                    leia_npz.close()
                    continue

                # ---- Compute F-D curves from FEM displacement fields ----
                try:
                    fem_fd = compute_fem_fd_curves(fem_result, points, cells, speeds)
                except Exception as e:
                    f_jsonl.write(json.dumps({
                        "id": cid, "fem_status": "fd_curve_failed",
                        "error": str(e)[:200],
                    }) + "\n")
                    f_jsonl.flush()
                    n_fem_fail += 1
                    leia_npz.close()
                    continue

                # ---- Final-step displacement + per-node VM, FEM side ----
                # The 'directions' dict maps label → {"outputs": (T+1, N, 3)}.
                # We want stretch + shear_y to match Stage 2.
                fem_disp_stretch = np.asarray(
                    fem_result["directions"]["stretch"]["outputs"][-1],
                    dtype=np.float32,
                )
                fem_disp_shear = np.asarray(
                    fem_result["directions"]["shear_y"]["outputs"][-1],
                    dtype=np.float32,
                )
                try:
                    fem_vm_stretch = fem_per_node_vm_at_final(
                        points, cells, fem_disp_stretch.astype(np.float64),
                        mu=args.mu, lmbda=args.lmbda,
                    ).astype(np.float32)
                    fem_vm_shear = fem_per_node_vm_at_final(
                        points, cells, fem_disp_shear.astype(np.float64),
                        mu=args.mu, lmbda=args.lmbda,
                    ).astype(np.float32)
                except Exception as e:
                    fem_vm_stretch = None
                    fem_vm_shear = None
                    print(f"    {cid}: VM compute failed: {e}")

                # ---- LEIA side (loaded from Stage 2 npz) ----
                leia_disp_stretch = leia_npz["disp_stretch"]
                leia_disp_shear = leia_npz["disp_shear"]
                leia_vm_stretch = leia_npz["vm_stretch"]
                leia_vm_shear = leia_npz["vm_shear"]

                # ---- Comparison metrics ----
                def _disp_l2(uA, uB):
                    den = float(np.linalg.norm(uB))
                    if den < 1e-12:
                        return float("nan")
                    return float(np.linalg.norm(uA - uB) / den)

                disp_l2_stretch = _disp_l2(leia_disp_stretch, fem_disp_stretch)
                disp_l2_shear = _disp_l2(leia_disp_shear, fem_disp_shear)
                disp_l2_mean = float(np.nanmean([disp_l2_stretch, disp_l2_shear]))

                if fem_vm_stretch is not None and fem_vm_shear is not None:
                    vm_corr_stretch = _safe_pearson(leia_vm_stretch, fem_vm_stretch)
                    vm_corr_shear = _safe_pearson(leia_vm_shear, fem_vm_shear)
                    vm_corr_mean = float(np.nanmean([vm_corr_stretch, vm_corr_shear]))
                else:
                    vm_corr_stretch = vm_corr_shear = vm_corr_mean = float("nan")

                # Total-work error
                fem_sw = compute_work(fem_fd["stretch"]["disp"],
                                      fem_fd["stretch"]["force"])
                fem_shw = compute_work(fem_fd["shear_y"]["disp"],
                                       fem_fd["shear_y"]["force"])
                fem_total = fem_sw + fem_shw
                leia_total = (float(leia_by_id[cid]["stretch_work"])
                              + float(leia_by_id[cid]["shear_work"]))
                if fem_total > 1e-12:
                    total_work_err = float(abs(leia_total - fem_total) / fem_total)
                else:
                    total_work_err = float("nan")

                # ---- Persist FEM fields ----
                npz_path = os.path.join(fem_outputs_dir, f"{cid}.npz")
                save_kwargs = {
                    "disp_stretch": fem_disp_stretch,
                    "disp_shear": fem_disp_shear,
                }
                if fem_vm_stretch is not None:
                    save_kwargs["vm_stretch"] = fem_vm_stretch
                    save_kwargs["vm_shear"] = fem_vm_shear
                np.savez_compressed(npz_path, **save_kwargs)
                leia_npz.close()

                entry = {
                    "id": cid,
                    "seed_plate": leia_by_id[cid].get("seed_plate"),
                    "depth": leia_by_id[cid].get("depth"),
                    "fem_status": "ok",
                    "fem_stretch_work": float(fem_sw),
                    "fem_shear_work": float(fem_shw),
                    "fem_total_work": float(fem_total),
                    "fem_fd_curves": fem_fd,
                    "fem_time_s": float(fem_result.get("time_seconds", 0.0)),
                    "disp_l2_err_stretch": disp_l2_stretch,
                    "disp_l2_err_shear": disp_l2_shear,
                    "disp_l2_err_mean": disp_l2_mean,
                    "vm_corr_stretch": vm_corr_stretch,
                    "vm_corr_shear": vm_corr_shear,
                    "vm_corr_mean": vm_corr_mean,
                    "total_work_err": total_work_err,
                    "leia_total_work": leia_total,
                    "fem_outputs_path": os.path.relpath(npz_path, args.leia_dir),
                }
                f_jsonl.write(json.dumps(entry) + "\n")
                f_jsonl.flush()
                n_ok += 1
                n_total_work_fem += fem_total

                if (j + 1) % 5 == 0 or j == len(submitted) - 1:
                    elapsed = time.time() - t_start
                    print(f"    [{batch_start + j + 1:04d}/{len(pending_ids)}] "
                          f"{cid}: disp_l2={disp_l2_mean:.3f}, "
                          f"vm_corr={vm_corr_mean:.3f}, "
                          f"work_err={total_work_err:.3f} | "
                          f"OK={n_ok}, fail={n_fem_fail}, diverged={n_diverged} | "
                          f"{(batch_start + j + 1) / max(elapsed, 1e-3):.2f} cand/s")

            pool.shutdown(wait=True)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed / 60:.1f} min. "
          f"OK={n_ok}, FEM_failed={n_fem_fail}, diverged={n_diverged}")
    print(f"  jsonl:  {fem_jsonl}")
    print(f"  npz:    {fem_outputs_dir}/")


if __name__ == "__main__":
    main()
