#!/usr/bin/env python3
"""Stage 4 of the Figure 5 pipeline (figure5.sh): novelty search.

55-seed novelty search with a shared global anchor list. Every training plate
seeds its own per-seed beam (B=2). Each iteration generates K mutations per
beam member; the global anchor pool — the union of the random-walk landscape's
projections,
training-plate projections, and every previously-evaluated mutant — is what
each candidate's novelty is computed against. Top-B by novelty per seed
becomes the next iteration's beam.

The shared anchor list is the collaboration mechanism: seeds in dense
neighborhoods get out-competed by mutations farther from their cluster, so
the cloud expands into white space rather than re-saturating the training
distribution.

Per-candidate work: mesh + encode + project (~2.5 s). NO characterization —
that happens in figure5.sh stage 5 (leia_novelty) on the saved candidates.

Usage:
    PYTHONHASHSEED=0 python -m design_search.landscape.novelty \\
        --output-dir design_search/outputs/<novelty_run> \\
        --pca-basis design_search/outputs/<landscape_run>/pca.json \\
        --B 2 --K 6 --D 8 --k-nn 15

    # Pilot:
    PYTHONHASHSEED=0 python -m design_search.landscape.novelty \\
        --output-dir design_search/outputs/<novelty_run>_pilot \\
        --pca-basis design_search/outputs/<landscape_run>/pca.json \\
        --B 2 --K 4 --D 2 --max-seeds 5
"""

import argparse
import sys
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np
import torch

from design_search.core.config import ServingConfig
from design_search.core.engine import ModelEngine
from design_search.core.meshing import generate_plate_mesh, MeshingError

from design_search.core.graph import sample_aggressive_random_mutation
from design_search.landscape.walk import (
    load_seed_graph_permissive,
    load_training_seed_ids,
)


def load_pca_basis(pca_path: str):
    """Read pca.json and return (mean, components, proj_v1, raw_dict)."""
    with open(pca_path) as f:
        d = json.load(f)
    mean = np.array(d["mean"], dtype=np.float64)              # (1024,)
    components = np.array(d["components"], dtype=np.float64)  # (3, 1024)
    proj_v1 = np.array(d["proj"], dtype=np.float64)           # (3080, 3)
    return mean, components, proj_v1, d


def project(latent: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Project a (1024,) latent to (3,) via (latent - mean) @ components.T."""
    return ((latent.astype(np.float64) - mean) @ components.T)


def evaluate_one(task: dict, engine_queue: Queue, config: ServingConfig,
                 mean: np.ndarray, components: np.ndarray) -> dict:
    """Mesh + encode + project a single candidate.

    Pulls a (gpu_idx, engine) pair from the queue, returns it after the call
    so other threads can reuse. Returns a dict with status='ok' (+ latent +
    proj), 'mesh_failed' (+ error), or 'encode_failed' (+ error).
    """
    gpu_idx, engine = engine_queue.get()
    try:
        torch.cuda.set_device(gpu_idx)

        try:
            mesh = generate_plate_mesh(task["graph"], config)
        except (MeshingError, Exception) as e:
            return {"task": task, "status": "mesh_failed",
                    "error": str(e)[:200]}

        session = None
        try:
            session = engine.create_session(mesh, rve_id=task["id"])
            latent = session.prevq.detach().mean(dim=0).cpu().numpy().astype(np.float32)
        except Exception as e:
            return {"task": task, "status": "encode_failed",
                    "error": str(e)[:200]}
        finally:
            if session is not None:
                session.prevq = None
                session.prev_state = None

        proj = project(latent, mean, components)
        return {
            "task": task, "status": "ok",
            "latent": latent, "proj": proj,
            "mesh_n_nodes": int(len(mesh.points)),
        }

    finally:
        torch.cuda.empty_cache()
        engine_queue.put((gpu_idx, engine))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output dir; if --run-root is set, "
                             "defaults to <run-root>/novelty.")
    parser.add_argument("--pca-basis", type=str, default=None,
                        help="Override PCA basis JSON; if --run-root is set, "
                             "defaults to <run-root>/pca.json.")
    parser.add_argument("--seeds-file", type=str, default=None,
                        help="JSON manifest with a 'plates' list. Default: "
                             "MICROPLATE_LATTICE.splits.training_plates.")
    parser.add_argument("--rve-bank", type=str, default="data/rve_bank")
    parser.add_argument("--B", type=int, default=2,
                        help="Beam width per seed.")
    parser.add_argument("--K", type=int, default=6,
                        help="Mutations per beam member per iter.")
    parser.add_argument("--D", type=int, default=8,
                        help="Iterations after the iter-0 seed.")
    parser.add_argument("--k-nn", type=int, default=15,
                        help="kNN for novelty (mean dist over k nearest).")
    parser.add_argument("--rng-base", type=int, default=100)
    parser.add_argument("--n-gpus", type=int, default=8)
    parser.add_argument("--max-seeds", type=int, default=None,
                        help="Cap on number of seeds (for pilot). None = all 55.")
    from design_search.landscape.layout import add_run_root_arg, resolve
    add_run_root_arg(parser)
    args = parser.parse_args()
    # Mutation seeds derive from hash() of a tuple that contains strings; Python
    # randomises string hashes per process unless PYTHONHASHSEED is set.
    hashseed = os.environ.get("PYTHONHASHSEED")
    if hashseed is None:
        sys.exit("novelty.py: set PYTHONHASHSEED (figure5.sh sets PYTHONHASHSEED=0); "
                 "without it the mutation seeds, and so the candidate set, change on every run.")
    if hashseed != "0":
        print(f"WARNING: PYTHONHASHSEED={hashseed}; reproducible, but the candidates differ "
              f"from a PYTHONHASHSEED=0 run.", flush=True)

    if args.output_dir is None:
        if args.run_root is None:
            parser.error("either --output-dir or --run-root is required")
        args.output_dir = resolve(args.run_root, "novelty")
    if args.pca_basis is None:
        if args.run_root is None:
            parser.error("either --pca-basis or --run-root is required")
        args.pca_basis = resolve(args.run_root, "pca")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- Load PCA basis + initial anchors ----------
    print(f"Loading PCA basis from {args.pca_basis}...")
    mean, components, proj_v1, _ = load_pca_basis(args.pca_basis)
    print(f"  components: {components.shape}, mean: {mean.shape}")
    print(f"  initial anchors from PCA basis: {proj_v1.shape}")

    anchors = list(proj_v1)            # list of (3,) np arrays
    initial_n_anchors = len(anchors)
    print(f"Initial anchors: {initial_n_anchors}")

    # ---------- Load seeds ----------
    print("\nLoading training-plate seeds...")
    seed_ids = load_training_seed_ids(args.seeds_file, args.max_seeds)
    seeds = []
    for sid in seed_ids:
        graph, errs = load_seed_graph_permissive(sid, rve_bank_dir=args.rve_bank)
        seeds.append({"id": sid, "graph": graph})
        if errs:
            print(f"  {sid} has {len(errs)} validation warnings (kept anyway)")
    print(f"Loaded {len(seeds)} seeds")

    # ---------- Spin up 8 ModelEngines, one per GPU ----------
    print(f"\nLoading {args.n_gpus} ModelEngines on cuda:0..{args.n_gpus - 1}...")
    config = ServingConfig()  # default device used only for reference; per-engine override below
    engines = []
    for i in range(args.n_gpus):
        cfg = ServingConfig()
        cfg.device = f"cuda:{i}"
        engines.append(ModelEngine(cfg))
    engine_queue = Queue()
    for i, e in enumerate(engines):
        engine_queue.put((i, e))
    print(f"All {args.n_gpus} engines ready.")

    # ---------- Helpers ----------
    def _id(seed_id: str, it: int, idx: int) -> str:
        return f"{seed_id}_iter{it:03d}_idx{idx:02d}"

    all_evaluated: list[dict] = []
    candidates_by_id: dict[str, dict] = {}

    stall_counts = {s["id"]: 0 for s in seeds}
    mesh_fail_counts = {s["id"]: 0 for s in seeds}
    encode_fail_counts = {s["id"]: 0 for s in seeds}

    # ---------- Iter 0: encode each seed's graph, build initial beams ----------
    print("\nIter 0: encoding 55 seed graphs to project them onto PCA...")
    iter0_tasks = [{
        "id": _id(s["id"], 0, 0), "seed_plate": s["id"],
        "iteration": 0, "parent_id": None, "graph": s["graph"],
    } for s in seeds]

    iter0_results = []
    with ThreadPoolExecutor(max_workers=args.n_gpus) as pool:
        futures = [pool.submit(evaluate_one, t, engine_queue, config, mean, components)
                   for t in iter0_tasks]
        for f in futures:
            iter0_results.append(f.result())

    n_seed_failures = sum(1 for r in iter0_results if r["status"] != "ok")
    if n_seed_failures > 0:
        print(f"  WARNING: {n_seed_failures}/{len(seeds)} seeds failed; "
              "their lineages are skipped.")

    beams: dict[str, list[dict]] = {}
    beam_history: dict[str, list[list[str]]] = {}

    for r in iter0_results:
        if r["status"] != "ok":
            sid = r["task"]["seed_plate"]
            if r["status"] == "mesh_failed":
                mesh_fail_counts[sid] += 1
            else:
                encode_fail_counts[sid] += 1
            continue
        t = r["task"]
        sid = t["seed_plate"]
        cid = t["id"]
        cand = {
            "id": cid, "seed_plate": sid,
            "depth": 0, "iteration": 0,
            "parent_id": None, "in_beam_at_iter": 0,
            "novelty": None,
            "proj": r["proj"].tolist(),
            "graph": t["graph"],
        }
        candidates_by_id[cid] = cand
        all_evaluated.append(cand)
        beams[sid] = [cand]
        beam_history[sid] = [[cid]]
        anchors.append(r["proj"])

    print(f"Iter 0: {len(beams)} active seed beams; anchors now {len(anchors)}")

    # ---------- Search loop ----------
    anchor_count_per_iter = [len(anchors)]
    mean_novelty_per_iter: list[float | None] = [None]
    iter_elapsed_s = [0.0]

    t_start = time.time()
    for it in range(1, args.D + 1):
        t_iter = time.time()
        print(f"\nIter {it}/{args.D}:")

        # 1. Generate K mutations per beam member, per seed
        queue: list[dict] = []
        per_seed_idx = {sid: 0 for sid in beams}
        for sid, beam in beams.items():
            for parent in beam:
                for k_idx in range(args.K):
                    # Per-(seed, iter, parent_id, k_idx) RNG; deterministic given PYTHONHASHSEED (checked in main)
                    seed_int = abs(hash(
                        (args.rng_base, sid, it, parent["id"], k_idx)
                    )) % (2**31)
                    rng = np.random.default_rng(seed_int)
                    mutant = sample_aggressive_random_mutation(parent["graph"], rng)
                    if mutant is None:
                        stall_counts[sid] += 1
                        continue
                    idx = per_seed_idx[sid]
                    per_seed_idx[sid] += 1
                    queue.append({
                        "task": {
                            "id": _id(sid, it, idx),
                            "seed_plate": sid,
                            "iteration": it,
                            "parent_id": parent["id"],
                            "graph": mutant,
                        }
                    })
        n_proposed = len(queue)
        print(f"  Generated {n_proposed} mutations "
              f"(stalls so far: {sum(stall_counts.values())})")

        # 2. Mesh + encode + project across 8 GPUs
        results = []
        with ThreadPoolExecutor(max_workers=args.n_gpus) as pool:
            futures = [pool.submit(evaluate_one, q["task"], engine_queue,
                                   config, mean, components) for q in queue]
            for f in futures:
                results.append(f.result())

        n_ok = sum(1 for r in results if r["status"] == "ok")
        n_mesh = sum(1 for r in results if r["status"] == "mesh_failed")
        n_enc = sum(1 for r in results if r["status"] == "encode_failed")
        for r in results:
            if r["status"] == "mesh_failed":
                mesh_fail_counts[r["task"]["seed_plate"]] += 1
            elif r["status"] == "encode_failed":
                encode_fail_counts[r["task"]["seed_plate"]] += 1
        print(f"  ok={n_ok}, mesh_fail={n_mesh}, encode_fail={n_enc}")

        # 3. Compute novelty against the GLOBAL anchor set
        ok_results = [r for r in results if r["status"] == "ok"]
        if ok_results:
            anchor_arr = np.array(anchors)
            for r in ok_results:
                d = np.linalg.norm(anchor_arr - r["proj"][None, :], axis=1)
                k = min(args.k_nn, len(d))
                r["novelty"] = float(np.partition(d, k - 1)[:k].mean())
            mean_novelty = float(np.mean([r["novelty"] for r in ok_results]))
        else:
            mean_novelty = float("nan")

        # 4. Per-seed top-B selection — preserves seed identity in beams
        next_beams: dict[str, list[dict]] = {}
        selected_ids = set()
        for sid in beams:
            seed_results = [r for r in ok_results if r["task"]["seed_plate"] == sid]
            seed_results.sort(key=lambda r: -r["novelty"])
            picked = seed_results[:args.B]
            next_beams[sid] = []
            beam_history[sid].append([])
            for r in picked:
                t = r["task"]
                cand = {
                    "id": t["id"], "seed_plate": sid,
                    "depth": it, "iteration": it,
                    "parent_id": t["parent_id"],
                    "in_beam_at_iter": it,
                    "novelty": r["novelty"],
                    "proj": r["proj"].tolist(),
                    "graph": t["graph"],
                }
                candidates_by_id[t["id"]] = cand
                next_beams[sid].append(cand)
                beam_history[sid][-1].append(t["id"])
                selected_ids.add(t["id"])
                all_evaluated.append(cand)

        # 5. Add unselected ok candidates (and update global anchor list with ALL)
        for r in ok_results:
            if r["task"]["id"] in selected_ids:
                continue
            t = r["task"]
            cand = {
                "id": t["id"], "seed_plate": t["seed_plate"],
                "depth": it, "iteration": it,
                "parent_id": t["parent_id"],
                "in_beam_at_iter": None,
                "novelty": r["novelty"],
                "proj": r["proj"].tolist(),
                "graph": t["graph"],
            }
            candidates_by_id[t["id"]] = cand
            all_evaluated.append(cand)

        for r in ok_results:
            anchors.append(r["proj"])

        beams = next_beams
        anchor_count_per_iter.append(len(anchors))
        mean_novelty_per_iter.append(mean_novelty)
        iter_elapsed_s.append(time.time() - t_iter)
        print(f"  mean novelty: {mean_novelty:.4f}, anchors now {len(anchors)}, "
              f"iter took {iter_elapsed_s[-1]:.1f}s")

    # ---------- Save outputs ----------
    elapsed = time.time() - t_start
    print(f"\nSearch complete: {elapsed / 60:.1f} min")

    candidates_path = os.path.join(args.output_dir, "candidates.json")
    with open(candidates_path + ".tmp", "w") as f:   # atomic: figure5.sh gates on this file
        json.dump({"candidates": all_evaluated}, f)
    os.replace(candidates_path + ".tmp", candidates_path)
    print(f"Saved {candidates_path}: {len(all_evaluated)} candidates")

    meta = {
        "B": args.B, "K": args.K, "D": args.D, "k_nn": args.k_nn,
        "rng_base": args.rng_base,
        "n_seeds_loaded": len(seeds),
        "n_seeds_active": len(beams),
        "n_initial_anchors": initial_n_anchors,
        "anchor_count_per_iter": anchor_count_per_iter,
        "mean_novelty_per_iter": mean_novelty_per_iter,
        "iter_elapsed_s": iter_elapsed_s,
        "beam_history": beam_history,
        "stall_counts": stall_counts,
        "mesh_fail_counts": mesh_fail_counts,
        "encode_fail_counts": encode_fail_counts,
        "pca_basis_path": args.pca_basis,
        "elapsed_s": elapsed,
        "n_evaluated_total": len(all_evaluated),
    }
    meta_path = os.path.join(args.output_dir, "novelty_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
