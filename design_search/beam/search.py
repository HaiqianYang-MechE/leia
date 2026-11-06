#!/usr/bin/env python3
"""Surrogate-guided beam search over lattice topology mutations.

Uses the production LEIA model as a fast evaluation oracle to search for
lattice microstructures optimized for mechanical anisotropy: hard to stretch,
easy to bend, at low volume fraction.

Usage:
    python -m design_search.beam.search \
        --seed mg_046 --B 5 --K 8 --D 10 \
        --output-dir design_search/outputs/beam_001
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from design_search.core.config import ServingConfig
from design_search.core.engine import ModelEngine
from design_search.core.graph import validate_graph
from design_search.core.meshing import generate_plate_mesh, MeshingError

from design_search.core.graph import sample_random_mutation
from design_search.core.objectives import volume_fraction, compute_work, anisotropy_score
from design_search.core.characterize import (
    sync_characterize,
    mesh_candidates_parallel,
    load_seed_graph,
    save_checkpoint,
    load_checkpoint,
    DIRECTIONS_2,
    DIRECTIONS_4,
)


def make_candidate_entry(
    graph: dict,
    char_data: dict,
    recon_error: float,
    iteration: int,
    parent_idx: int,
    ood_threshold: float,
    score_formula: str = "product",
    lambda_vf: float = 100.0,
) -> dict:
    """Create a serializable candidate entry for logging."""
    vf = volume_fraction(graph)
    sw = compute_work(char_data["stretch"]["disp"], char_data["stretch"]["force"])
    shw = compute_work(char_data["shear_y"]["disp"], char_data["shear_y"]["force"])
    score = anisotropy_score(graph, char_data,
                             formula=score_formula, lambda_vf=lambda_vf)

    return {
        "graph": graph,
        "score": score,
        "stretch_work": sw,
        "shear_work": shw,
        "vol_frac": vf,
        "recon_error": recon_error,
        "ood": recon_error > ood_threshold,
        "iteration": iteration,
        "parent_idx": parent_idx,
        "fd_curves": char_data,
        "n_nodes": len(graph["nodes"]),
        "n_beams": len(graph["beams"]),
    }


def run_beam_search(args):
    """Main beam search loop."""
    print(f"=== Beam Search: seed={args.seed}, B={args.B}, K={args.K}, D={args.D} ===")
    print(f"Output: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load model ---
    config = ServingConfig()
    if getattr(args, "mesh_res", None) is not None:
        config.mesh_res = args.mesh_res
        print(f"[override] mesh_res = {config.mesh_res}")
    engine = ModelEngine(config)

    # --- Check for checkpoint to resume ---
    checkpoint = load_checkpoint(args.output_dir)
    if checkpoint is not None:
        start_iter, beam, all_candidates = checkpoint
        start_iter += 1  # resume from next iteration
        print(f"Resuming from iteration {start_iter} ({len(all_candidates)} candidates so far)")
    else:
        start_iter = 0
        all_candidates = []
        beam = []

    # --- Initialize seed (uses base random_seed for seed-phase RNG) ---
    if not beam:
        print(f"\nLoading seed: {args.seed}")
        seed_graph = load_seed_graph(args.seed)
        print(f"  {len(seed_graph['nodes'])} nodes, {len(seed_graph['beams'])} beams, "
              f"vol_frac={volume_fraction(seed_graph):.6f}")

        print("Meshing seed...")
        try:
            seed_mesh = generate_plate_mesh(seed_graph, config)
        except MeshingError as e:
            print(f"  FATAL: seed mesh failed ({e}) — seed must be valid")
            raise
        print(f"  {seed_mesh.points.shape[0]} nodes, {seed_mesh.cells.shape[0]} tets")

        seed_session = engine.create_session(seed_mesh, rve_id=args.seed)
        seed_session.graph_data = seed_graph

        print("Characterizing seed (stretch + shear_y)...")
        fd = sync_characterize(engine, [seed_session], DIRECTIONS_2, args.steps,
                               batch_size=args.batch_size)
        seed_char = fd[seed_session.session_id]
        seed_recon = engine.compute_recon_error(seed_session)

        seed_entry = make_candidate_entry(
            seed_graph, seed_char, seed_recon, -1, -1, config.ood_threshold,
            score_formula=args.score_formula, lambda_vf=args.lambda_vf,
        )
        all_candidates.append(seed_entry)
        beam = [seed_entry]

        print(f"  score={seed_entry['score']:.2f}, stretch_work={seed_entry['stretch_work']:.4f}, "
              f"shear_work={seed_entry['shear_work']:.4f}, recon={seed_recon:.8f}")

        # Cleanup seed session
        seed_session.prevq = None
        torch.cuda.empty_cache()

    # --- Main search loop ---
    t_search_start = time.time()

    for iteration in range(start_iter, args.D):
        t_iter = time.time()
        # Per-iteration RNG: deterministic and resumable
        rng = np.random.default_rng(args.random_seed + iteration + 1)
        print(f"\n--- Iteration {iteration}/{args.D} ---")

        # (a) Generate mutations from each beam member
        raw_candidates = []
        # Over-generate to compensate for mesh failures. Every mutant that
        # meshes is evaluated (no trim back to K), so K is a nominal count:
        # up to 1.5*K children per parent reach the beam selection.
        n_per_parent = int(args.K * 1.5)

        for parent_idx, parent in enumerate(beam):
            parent_graph = parent["graph"]
            count = 0
            for _ in range(n_per_parent * 3):  # extra attempts
                if count >= n_per_parent:
                    break
                mutant = sample_random_mutation(parent_graph, rng)
                if mutant is not None:
                    raw_candidates.append((parent_idx, mutant))
                    count += 1

        print(f"  Generated {len(raw_candidates)} raw candidates "
              f"({len(beam)} parents × ~{n_per_parent} each)")

        # (b) Pre-filter: validate + volume fraction bounds + per-beam radius floor
        filtered = []
        for parent_idx, g in raw_candidates:
            valid, _ = validate_graph(g)
            if not valid:
                continue
            vf = volume_fraction(g)
            if vf < 0.001 or vf > 0.5:
                continue
            # Per-beam minimum radius: rejects "compliant hinge" designs whose
            # FEM forces hit the noise floor at typical mesh resolutions.
            if args.min_radius > 0:
                min_r = min(
                    (b.get("r_start", 0.05) + b.get("r_end", 0.05)) / 2
                    for b in g["beams"]
                )
                if min_r < args.min_radius:
                    continue
            filtered.append((parent_idx, g))

        print(f"  After pre-filter: {len(filtered)} candidates")

        if not filtered:
            print("  WARNING: No valid candidates this iteration, keeping beam unchanged")
            save_checkpoint(args.output_dir, iteration,
                            beam, all_candidates)
            continue

        # (c) Mesh all candidates in parallel
        print(f"  Meshing {len(filtered)} candidates...")
        t_mesh = time.time()
        graphs_only = [g for _, g in filtered]
        mesh_results = mesh_candidates_parallel(graphs_only, config, max_workers=7)
        mesh_time = time.time() - t_mesh
        print(f"  Meshed {len(mesh_results)}/{len(filtered)} successfully ({mesh_time:.1f}s)")

        if not mesh_results:
            print("  WARNING: All meshes failed, keeping beam unchanged")
            save_checkpoint(args.output_dir, iteration,
                            beam, all_candidates)
            continue

        # (d) Create sessions + encode
        sessions = []
        session_meta = []  # (parent_idx, graph) per session
        for mesh_idx, mesh in mesh_results:
            parent_idx, graph = filtered[mesh_idx]
            session = engine.create_session(mesh)
            session.graph_data = graph
            sessions.append(session)
            session_meta.append((parent_idx, graph))

        # (e) Characterize (stretch + shear_y)
        print(f"  Characterizing {len(sessions)} candidates...")
        t_char = time.time()
        fd_all = sync_characterize(engine, sessions, DIRECTIONS_2, args.steps,
                                   batch_size=args.batch_size)
        char_time = time.time() - t_char
        print(f"  Characterization: {char_time:.1f}s")

        # (f) Score + OOD detection
        iter_entries = []
        for s, (parent_idx, graph) in zip(sessions, session_meta):
            sid = s.session_id
            recon = engine.compute_recon_error(s)
            entry = make_candidate_entry(
                graph, fd_all[sid], recon, iteration, parent_idx,
                config.ood_threshold,
                score_formula=args.score_formula, lambda_vf=args.lambda_vf,
            )
            iter_entries.append(entry)
            all_candidates.append(entry)

        # (g) Sort by score, keep top-B
        iter_entries.sort(key=lambda x: x["score"], reverse=True)
        beam = iter_entries[:args.B]

        # (h) Cleanup sessions
        for s in sessions:
            s.prevq = None
        del sessions
        torch.cuda.empty_cache()

        # (i) Log
        iter_time = time.time() - t_iter
        n_ood = sum(1 for e in iter_entries if e["ood"])
        beam_scores = ", ".join(f"{e['score']:.1f}" for e in beam)
        best = beam[0]
        print(f"  Beam scores: [{beam_scores}]")
        print(f"  Best: score={best['score']:.2f}, vol_frac={best['vol_frac']:.6f}, "
              f"nodes={best['n_nodes']}, beams={best['n_beams']}")
        print(f"  OOD: {n_ood}/{len(iter_entries)}, iter time: {iter_time:.1f}s")

        # (j) Checkpoint
        save_checkpoint(args.output_dir, iteration, beam, all_candidates)

    # --- Final: full 4-direction characterization on winner ---
    total_time = time.time() - t_search_start
    print(f"\n=== Search complete: {total_time:.1f}s total, "
          f"{len(all_candidates)} candidates evaluated ===")

    # The beam is non-elitist (each iteration keeps the top-B children only),
    # so the final beam's best is not necessarily the run's best. Report the
    # best candidate of the whole run — the design the figure stars and the
    # FEM validation checks.
    winner = max(all_candidates, key=lambda c: c["score"])
    print(f"\nWinner: score={winner['score']:.2f}, vol_frac={winner['vol_frac']:.6f}, "
          f"nodes={winner['n_nodes']}, beams={winner['n_beams']}")

    print("Running full 4-direction characterization on winner...")
    winner_full_char = None
    winner_session = None
    try:
        winner_mesh = generate_plate_mesh(winner["graph"], config)
        winner_session = engine.create_session(winner_mesh)
        winner_fd = sync_characterize(
            engine, [winner_session], DIRECTIONS_4, args.steps,
            batch_size=args.batch_size,
        )
        winner_full_char = winner_fd[winner_session.session_id]
    except MeshingError as e:
        print(f"  WARNING: winner mesh failed ({e}); saving partial results without full_char")

    # Save final results
    results = {
        "seed": args.seed,
        "params": {
            "B": args.B, "K": args.K, "D": args.D, "steps": args.steps,
            "mesh_res": getattr(args, "mesh_res", None),
            "batch_size": args.batch_size,
            "score_formula": args.score_formula,
            "lambda_vf": args.lambda_vf,
            "min_radius": args.min_radius,
        },
        "total_time_s": total_time,
        "n_candidates": len(all_candidates),
        "winner": {
            **winner,
            "full_char": winner_full_char,
        },
        "beam": beam,
        "all_candidates": all_candidates,
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path + ".tmp", "w") as f:   # atomic: figure4.sh gates on this file
        json.dump(results, f, indent=2)
    os.replace(results_path + ".tmp", results_path)
    print(f"Results saved to {results_path}")

    # Cleanup
    if winner_session is not None:
        winner_session.prevq = None
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Beam search over lattice topologies")
    parser.add_argument("--seed", type=str, default="mg_046",
                        help="RVE ID for seed topology")
    parser.add_argument("--B", type=int, default=5,
                        help="Beam width (top-B kept per iteration)")
    parser.add_argument("--K", type=int, default=8,
                        help="Nominal expansions per beam member per iteration. The "
                             "search generates 1.5*K mutants per parent to absorb mesh "
                             "failures and evaluates every one that meshes (up to 12 for "
                             "K=8; the paper's runs did the same).")
    parser.add_argument("--D", type=int, default=10,
                        help="Number of iterations")
    parser.add_argument("--steps", type=int, default=20,
                        help="Steps per characterization direction")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--mesh-res", type=int, default=None,
                        help="Override ServingConfig.mesh_res (default: serving res=10)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Sub-batch size for sync_characterize (decode is "
                             "memory-bound — reduce to 2 at mesh_res=30)")
    parser.add_argument("--score-formula", type=str, default="product",
                        choices=["product", "additive"],
                        help="Anisotropy score formula. 'product' (default): "
                             "stretch/(shear*vol_frac) — original, has 1/vf "
                             "singularity. 'additive': stretch/shear - "
                             "lambda_vf*vol_frac — bounded.")
    parser.add_argument("--lambda-vf", type=float, default=100.0,
                        help="Volume-fraction penalty for the 'additive' "
                             "formula (ignored for 'product')")
    parser.add_argument("--min-radius", type=float, default=0.0,
                        help="Per-beam minimum radius (cell-length units). "
                             "Candidates with any beam radius below this are "
                             "rejected during pre-filter. Default 0 = no floor.")
    parser.add_argument("--output-dir", type=str,
                        default="design_search/outputs/beam_001")
    args = parser.parse_args()

    run_beam_search(args)


if __name__ == "__main__":
    main()
