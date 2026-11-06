#!/usr/bin/env python3
"""Per-candidate LEIA scoring and dynamics-consistency for the landscape pipeline.

Two metrics, sharing engine load + candidate iteration + JSONL I/O:

  --metric=score
      Mesh + encode + 2-direction characterization (stretch + shear).
      Outputs `leia_scores.jsonl` + per-candidate `leia_outputs/<id>.npz`
      (displacement_final, von_mises_final, mesh_points, mesh_cells,
      latent_meanpool). Scalars: recon_error, stretch_work, shear_work,
      vol_frac, anisotropy_score.

  --metric=consistency
      Mesh + create_session + AR rollout (uniaxial stretch, --n-steps frames).
      At each step decode + re-encode (3-frame window) and record
      ||z_t - z_t_recovered||/||z_t||. Outputs `consistency.jsonl` with
      per-frame `c_per_frame` and `z_norms`.

Resumable in score mode (skips IDs already in the JSONL); consistency mode
overwrites. Both modes support `--shard-id i --num-shards N` for parallel
multi-GPU runs; consistency additionally supports an in-process
`--n-gpus N` ThreadPoolExecutor mode (no sharding).

Usage:
  python -m design_search.landscape.score --metric=score \\
      --candidates design_search/outputs/run/candidates.json \\
      --output-dir design_search/outputs/run

  python -m design_search.landscape.score --metric=consistency \\
      --output-dir design_search/outputs/run/consistency \\
      --landscape-dirs design_search/outputs/run/walk design_search/outputs/run/novelty
"""

import argparse
import json
import os
import time
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from design_search.core.config import ServingConfig
from design_search.core.engine import ModelEngine
from design_search.core.meshing import generate_plate_mesh, MeshingError
from design_search.core.objectives import (
    anisotropy_score,
    compute_work,
    volume_fraction,
)
from design_search.core.characterize import DIRECTIONS_2


# Standard fixed action used by --metric=consistency: uniaxial stretch.
# After 10 steps at speed=0.15, the cumulative BC reaches 1.5 — moderate
# deformation, well inside training distribution.
STD_ACTION = [1, 0, 0, 0]


# ---------------------------------------------------------------------------
# --metric=score: per-candidate 2-direction characterization
# ---------------------------------------------------------------------------

def _characterize_with_fields(
    engine: ModelEngine,
    session,
    directions: list[tuple[list[int], int, str]],
    steps_per_direction: int = 20,
) -> dict:
    """Step `session` through each direction, recording F-D plus final-step
    displacement (N, 3) and von Mises (N,) fields per direction."""
    out = {}
    for action, dof_idx, label in directions:
        disps = [0.0]
        forces = [0.0]
        last_disp = None
        last_vm = None
        for _ in range(steps_per_direction):
            results = engine.step_batch([session], action)
            r = results[0]
            disps.append(float(r.gen_disp[dof_idx]))
            forces.append(float(r.gen_force[dof_idx]))
            last_disp = r.displacement
            last_vm = r.von_mises
        out[label] = {
            "disp": disps,
            "force": forces,
            "displacement_final": last_disp.astype(np.float32),
            "von_mises_final": last_vm.astype(np.float32),
        }
        engine.reset_session(session)
    return out


def _evaluate_score(cand: dict, engine: ModelEngine, config: ServingConfig,
                    leia_outputs_dir: str, output_dir: str,
                    steps_per_direction: int) -> dict:
    """Run --metric=score on a single candidate. Returns the JSONL entry dict."""
    cid = cand["id"]
    graph = cand["graph"]
    t0 = time.time()

    try:
        mesh = generate_plate_mesh(graph, config)
    except (MeshingError, Exception) as e:
        return {
            "id": cid,
            "seed_plate": cand.get("seed_plate"),
            "depth": cand.get("depth"),
            "status": "mesh_failed",
            "error": str(e)[:200],
        }

    session = None
    try:
        session = engine.create_session(mesh, rve_id=cid)
        t_encode = time.time() - t0

        latent_meanpool = (
            session.prevq.detach().mean(dim=0).cpu().numpy().astype(np.float32)
        )
        recon_error = engine.compute_recon_error(session)

        t_char_start = time.time()
        char = _characterize_with_fields(
            engine, session, DIRECTIONS_2,
            steps_per_direction=steps_per_direction,
        )
        t_char = time.time() - t_char_start

        sw = compute_work(char["stretch"]["disp"], char["stretch"]["force"])
        shw = compute_work(char["shear_y"]["disp"], char["shear_y"]["force"])
        fd_curves = {
            "stretch": {"disp": char["stretch"]["disp"],
                        "force": char["stretch"]["force"]},
            "shear_y": {"disp": char["shear_y"]["disp"],
                        "force": char["shear_y"]["force"]},
        }
        score = anisotropy_score(graph, fd_curves)
        vf = volume_fraction(graph)

        npz_path = os.path.join(leia_outputs_dir, f"{cid}.npz")
        np.savez_compressed(
            npz_path,
            disp_stretch=char["stretch"]["displacement_final"],
            vm_stretch=char["stretch"]["von_mises_final"],
            disp_shear=char["shear_y"]["displacement_final"],
            vm_shear=char["shear_y"]["von_mises_final"],
            mesh_points=mesh.points.astype(np.float32),
            mesh_cells=mesh.cells.astype(np.int32),
            latent_meanpool=latent_meanpool,
        )

        return {
            "id": cid,
            "seed_plate": cand.get("seed_plate"),
            "depth": cand.get("depth"),
            "status": "ok",
            "recon_error": float(recon_error),
            "ood": bool(recon_error > config.ood_threshold),
            "stretch_work": float(sw),
            "shear_work": float(shw),
            "vol_frac": float(vf),
            "score": float(score),
            "fd_curves": fd_curves,
            "n_nodes": len(graph["nodes"]),
            "n_beams": len(graph["beams"]),
            "mesh_n_nodes": int(len(mesh.points)),
            "mesh_n_cells": int(len(mesh.cells)),
            "encode_time_s": float(t_encode),
            "char_time_s": float(t_char),
            "leia_outputs_path": os.path.relpath(npz_path, output_dir),
        }
    except Exception as e:
        return {
            "id": cid,
            "seed_plate": cand.get("seed_plate"),
            "depth": cand.get("depth"),
            "status": "score_failed",
            "error": str(e)[:200],
        }
    finally:
        if session is not None:
            session.prevq = None
            session.prev_state = None
            del session
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# --metric=consistency: AR-rollout encoder-cycle consistency
# ---------------------------------------------------------------------------

def _encode_3frame(engine: ModelEngine, ref_norm_sub: np.ndarray,
                   u_prev2: np.ndarray | None,
                   u_prev1: np.ndarray | None,
                   u_curr: np.ndarray) -> torch.Tensor:
    """Re-encode a 3-frame window as the v2 tokenizer expects."""
    n_input_frames = int(engine.tok_cfg.get('n_input_frames', 1))
    assert n_input_frames == 3, f"Expected n_input_frames=3, got {n_input_frames}"
    zero = np.zeros_like(u_curr)
    f0 = u_prev2 if u_prev2 is not None else zero
    f1 = u_prev1 if u_prev1 is not None else zero
    return engine.engine.encode_history(np.stack([f0, f1, u_curr]), ref_norm_sub)


def _evaluate_consistency(task: dict, engine_queue: Queue,
                          config: ServingConfig, n_steps: int) -> dict:
    """Run --metric=consistency on a single candidate."""
    gpu_idx, engine = engine_queue.get()
    try:
        torch.cuda.set_device(gpu_idx)

        try:
            mesh = generate_plate_mesh(task["graph"], config)
        except (MeshingError, Exception) as e:
            return {"id": task["id"], "status": "mesh_failed",
                    "error": str(e)[:200]}

        session = None
        try:
            session = engine.create_session(mesh, rve_id=task["id"])

            N = len(mesh.points)
            subsample_n = min(10240, N)
            rng = np.random.default_rng(42)
            sub_idx = rng.permutation(N)[:subsample_n]

            c_per_frame = []
            z_norms = []
            u_prev2 = None
            u_prev1 = None

            for _ in range(1, n_steps + 1):
                results = engine.step_batch([session], STD_ACTION)
                z_t = session.prevq.detach().clone()
                disp_full = results[0].displacement
                u_curr_sub = disp_full[sub_idx].astype(np.float32)

                z_t_rec = _encode_3frame(
                    engine, session.ref_norm_sub,
                    u_prev2, u_prev1, u_curr_sub,
                )

                num = float(torch.norm(z_t - z_t_rec, p="fro").item())
                den = float(torch.norm(z_t, p="fro").item())
                c_per_frame.append(num / den if den > 0 else float("nan"))
                z_norms.append(den)

                u_prev2 = u_prev1
                u_prev1 = u_curr_sub

            return {
                "id": task["id"], "status": "ok",
                "c_per_frame": c_per_frame,
                "z_norms": z_norms,
                "mesh_n_nodes": int(N),
            }

        except Exception as e:
            return {"id": task["id"], "status": "encode_failed",
                    "error": str(e)[:200]}
        finally:
            if session is not None:
                session.prevq = None
                session.prev_state = None

    finally:
        torch.cuda.empty_cache()
        engine_queue.put((gpu_idx, engine))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _already_processed_ids(jsonl_path: str) -> set[str]:
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


def _load_candidates_combined(landscape_dirs: list[str]) -> list[dict]:
    """Concatenate candidates.json across multiple landscape sub-stage dirs."""
    out = []
    for ldir in landscape_dirs:
        path = os.path.join(ldir, "candidates.json")
        with open(path) as f:
            cd = json.load(f)
        for c in cd["candidates"]:
            out.append({"id": c["id"], "graph": c["graph"], "source_dir": ldir})
        print(f"  Loaded {len(cd['candidates'])} from {path}")
    return out


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def _run_score(args):
    """--metric=score: read --candidates, write leia_scores.jsonl + npz/."""
    os.makedirs(args.output_dir, exist_ok=True)
    leia_outputs_dir = os.path.join(args.output_dir, "leia_outputs")
    os.makedirs(leia_outputs_dir, exist_ok=True)
    if args.num_shards == 1:
        leia_jsonl = os.path.join(args.output_dir, "leia_scores.jsonl")
    else:
        leia_jsonl = os.path.join(
            args.output_dir, f"leia_scores_shard{args.shard_id}.jsonl")

    config = ServingConfig()
    config.device = args.device
    print(f"Loading production v2 ModelEngine on {config.device}...")
    print(f"  tokenizer: {config.tokenizer_ckpt}")
    print(f"  dynamics:  {config.dynamics_ckpt}")
    engine = ModelEngine(config)

    with open(args.candidates) as f:
        cand_data = json.load(f)
    candidates = cand_data["candidates"]
    if args.max_candidates is not None:
        candidates = candidates[:args.max_candidates]

    if args.num_shards > 1:
        candidates = [c for i, c in enumerate(candidates)
                      if i % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id}/{args.num_shards}: "
              f"{len(candidates)} candidates")

    done = _already_processed_ids(leia_jsonl)
    pending = [c for c in candidates if c["id"] not in done]
    print(f"\nCandidates: total={len(candidates)}, done={len(done)}, "
          f"pending={len(pending)}\n")

    n_ok = 0
    n_mesh_fail = 0
    n_score_fail = 0
    t_start = time.time()

    with open(leia_jsonl, "a") as f_jsonl:
        for i, cand in enumerate(pending):
            entry = _evaluate_score(
                cand, engine, config,
                leia_outputs_dir, args.output_dir,
                args.steps_per_direction,
            )
            f_jsonl.write(json.dumps(entry) + "\n")
            f_jsonl.flush()
            status = entry["status"]
            if status == "ok":
                n_ok += 1
            elif status == "mesh_failed":
                n_mesh_fail += 1
            else:
                n_score_fail += 1
            if (i + 1) % 5 == 0 or i == len(pending) - 1:
                elapsed = time.time() - t_start
                rate = (i + 1) / max(elapsed, 1e-3)
                if status == "ok":
                    print(f"  [{i + 1:04d}/{len(pending)}] {entry['id']}: "
                          f"recon={entry['recon_error']:.2e}, "
                          f"score={entry['score']:.1f}, "
                          f"vf={entry['vol_frac']:.3f} | OK={n_ok}, "
                          f"mesh_fail={n_mesh_fail}, score_fail={n_score_fail} "
                          f"| {rate:.2f} cand/s")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed / 60:.1f} min. "
          f"OK={n_ok}, mesh_failed={n_mesh_fail}, score_failed={n_score_fail}")
    print(f"  jsonl:  {leia_jsonl}")
    print(f"  npz:    {leia_outputs_dir}/")


def _run_consistency(args):
    """--metric=consistency: read --landscape-dirs, write consistency.jsonl."""
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading candidates from:")
    candidates = _load_candidates_combined(args.landscape_dirs)
    print(f"Total candidates: {len(candidates)}")

    if args.shard_id is not None:
        assert args.num_shards is not None
        candidates = [c for i, c in enumerate(candidates)
                      if i % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id}/{args.num_shards}: "
              f"{len(candidates)} candidates")

    if args.shard_id is not None:
        devices = [args.device]
    else:
        devices = [f"cuda:{i}" for i in range(args.n_gpus)]

    print(f"Loading {len(devices)} ModelEngine(s) on {devices}...")
    engines = []
    for dev in devices:
        cfg = ServingConfig()
        cfg.device = dev
        engines.append(ModelEngine(cfg))
    engine_queue = Queue()
    for i, e in enumerate(engines):
        gpu_idx = int(devices[i].split(":")[1]) if ":" in devices[i] else 0
        engine_queue.put((gpu_idx, e))
    print(f"All {len(engines)} engines ready.")

    # Mirrors _run_score: a single shard writes the final file directly, so
    # figure5.sh's merge step (which only runs for OOD_N_GPUS > 1) is not needed.
    if args.num_shards and args.num_shards > 1:
        out_file = os.path.join(
            args.output_dir, f"consistency_shard{args.shard_id}.jsonl")
    else:
        out_file = os.path.join(args.output_dir, "consistency.jsonl")

    f_out = open(out_file, "w")
    n_ok = 0
    n_fail = 0
    t_start = time.time()

    if len(engines) == 1:
        for i, task in enumerate(candidates):
            r = _evaluate_consistency(task, engine_queue, ServingConfig(),
                                      args.n_steps)
            f_out.write(json.dumps(r) + "\n")
            f_out.flush()
            if r["status"] == "ok":
                n_ok += 1
            else:
                n_fail += 1
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (len(candidates) - (i + 1)) / max(rate, 1e-6)
                print(f"  [shard {args.shard_id}] {i + 1}/{len(candidates)}  "
                      f"ok={n_ok}  fail={n_fail}  "
                      f"({rate:.2f}/s, ETA {eta / 60:.1f} min)")
    else:
        with ThreadPoolExecutor(max_workers=len(engines)) as pool:
            futures = [pool.submit(_evaluate_consistency, t, engine_queue,
                                   ServingConfig(), args.n_steps)
                       for t in candidates]
            for i, fut in enumerate(futures):
                r = fut.result()
                f_out.write(json.dumps(r) + "\n")
                f_out.flush()
                if r["status"] == "ok":
                    n_ok += 1
                else:
                    n_fail += 1
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed
                    eta = (len(candidates) - (i + 1)) / max(rate, 1e-6)
                    print(f"  {i + 1}/{len(candidates)}  ok={n_ok}  "
                          f"fail={n_fail}  ({rate:.2f}/s, "
                          f"ETA {eta / 60:.1f} min)")

    f_out.close()
    elapsed = time.time() - t_start
    print(f"\nDone: {n_ok} ok, {n_fail} failed in {elapsed / 60:.1f} min")
    print(f"Saved {out_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, required=True,
                        choices=["score", "consistency"],
                        help="What to compute per candidate.")

    # Shared
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--stage", type=str, default=None,
                        choices=["walk", "novelty", "heldout"],
                        help="--metric=score: which sub-stage to score "
                             "(when --run-root is set).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--shard-id", type=int, default=None,
                        help="Shard index in [0, num-shards).")
    parser.add_argument("--num-shards", type=int, default=None)

    # --metric=score
    parser.add_argument("--candidates", type=str, default=None,
                        help="(score) Path to candidates.json.")
    parser.add_argument("--steps-per-direction", type=int, default=20,
                        help="(score) Steps per direction.")
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="(score) Cap for pilot runs.")

    # --metric=consistency
    parser.add_argument("--landscape-dirs", type=str, nargs="+", default=None,
                        help="(consistency) List of landscape sub-stage dirs "
                             "to combine.")
    parser.add_argument("--n-steps", type=int, default=10,
                        help="(consistency) AR rollout horizon.")
    parser.add_argument("--n-gpus", type=int, default=8,
                        help="(consistency) ModelEngine instances spawned in-process. "
                             "Set to 1 for single-GPU sharded mode.")

    from design_search.landscape.layout import add_run_root_arg, resolve
    add_run_root_arg(parser)
    args = parser.parse_args()

    if args.metric == "score":
        # Default num_shards to 1 for score (it ran without --num-shards before).
        if args.num_shards is None:
            args.num_shards = 1
            args.shard_id = 0
        else:
            assert args.shard_id is not None and 0 <= args.shard_id < args.num_shards, \
                f"--shard-id must be in [0, {args.num_shards})"

        if args.output_dir is None:
            if args.run_root is None or args.stage is None:
                parser.error("either --output-dir or (--run-root and --stage) is required")
            args.output_dir = resolve(args.run_root, "heldout_dir" if args.stage == "heldout" else args.stage)
        if args.candidates is None:
            if args.run_root is None or args.stage is None:
                parser.error("either --candidates or (--run-root and --stage) is required")
            args.candidates = os.path.join(
                resolve(args.run_root, "heldout_dir" if args.stage == "heldout" else args.stage), "candidates.json")

        _run_score(args)

    else:  # consistency
        if args.output_dir is None:
            if args.run_root is None:
                parser.error("either --output-dir or --run-root is required")
            args.output_dir = resolve(args.run_root, "consistency_dir")
        if args.landscape_dirs is None:
            if args.run_root is None:
                parser.error("either --landscape-dirs or --run-root is required")
            args.landscape_dirs = [resolve(args.run_root, "walk"),
                                   resolve(args.run_root, "novelty")]

        _run_consistency(args)


if __name__ == "__main__":
    main()
