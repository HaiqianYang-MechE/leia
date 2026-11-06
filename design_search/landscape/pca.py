#!/usr/bin/env python3
"""Mean-pool latent encoding for the landscape pipeline.

Two subcommands:

  fit-pca
      Fit a 3-component PCA on the 55 training-plate mean-pooled latents.
      Used as the *bootstrap* PCA for `landscape.novelty` (anchor distances).
      The figure-rendering PCA in `landscape.figure` re-fits on the
      training-seed mask inside the candidate set; the two will agree closely
      because the depth-0 entries of the candidate set are the 55 training
      plates encoded by this script.

      Output schema (matches what landscape.novelty expects):
          {
            "method": "pca",
            "fit_on": "training",
            "n_components": 3,
            "n_train": 55,
            "mean": [...],                       # (1024,)
            "components": [[...], [...], [...]], # (3, 1024)
            "explained_variance_ratio": [r0, r1, r2],
            "plate_ids": ["mg_046", ...],
            "proj": [[...], ...]                 # (55, 3)
          }

  encode-heldout
      Encode the 8 held-out plates with the production tokenizer and save
      their mean-pooled 1024-d latents to a JSON consumed by
      `landscape.figure` for held-out marker overlays.

Usage:
  python -m design_search.landscape.pca fit-pca --run-root <root>
  python -m design_search.landscape.pca encode-heldout --run-root <root>
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.decomposition import PCA

from design_search.core.config import ServingConfig
from design_search.core.engine import ModelEngine
from design_search.core.meshing import generate_plate_mesh, MeshingError
from design_search.landscape.layout import add_run_root_arg, resolve
from leia.regime import MICROPLATE_LATTICE


def encode_plate_meanpool(engine, cfg, pid: str,
                          rve_bank_dir: str = "data/rve_bank",
                          load_graph_fn=None):
    """Mesh + encode plate ``pid`` and return (mean-pooled 1024-d latent, n_nodes).

    Returns ``None`` on mesh or encode failure (caller skips). Frees the GPU
    session after extracting the latent. ``load_graph_fn`` overrides the
    default `landscape.walk.load_seed_graph_permissive` (used by tests).
    """
    if load_graph_fn is None:
        from design_search.landscape.walk import load_seed_graph_permissive
        load_graph_fn = load_seed_graph_permissive
    graph, _ = load_graph_fn(pid, rve_bank_dir=rve_bank_dir)
    try:
        mesh = generate_plate_mesh(graph, cfg)
    except (MeshingError, Exception):
        return None
    session = None
    try:
        session = engine.create_session(mesh, rve_id=pid)
        latent = session.prevq.detach().mean(dim=0).cpu().numpy().astype(np.float32)
        return latent, int(len(mesh.points))
    except Exception:
        return None
    finally:
        if session is not None:
            try:
                session.prevq = None
                session.prev_state = None
            except Exception:
                pass
        torch.cuda.empty_cache()


def _run_fit_pca(args):
    from design_search.landscape.walk import load_training_seed_ids
    plate_ids = load_training_seed_ids(args.seeds_file, args.max_seeds)
    print(f"Fitting PCA on {len(plate_ids)} training plates...")

    print(f"Loading ModelEngine on {args.device}...")
    cfg = ServingConfig()
    cfg.device = args.device
    engine = ModelEngine(cfg)

    t0 = time.time()
    encoded_ids = []
    latents = []
    for i, pid in enumerate(plate_ids):
        result = encode_plate_meanpool(engine, cfg, pid, rve_bank_dir=args.rve_bank)
        if result is None:
            print(f"  [{i + 1:02d}/{len(plate_ids)}] {pid}: SKIP (mesh/encode failed)")
            continue
        latent, n_nodes = result
        encoded_ids.append(pid)
        latents.append(latent)
        print(f"  [{i + 1:02d}/{len(plate_ids)}] {pid}: ok  "
              f"({n_nodes} nodes, |z|={np.linalg.norm(latent):.3f})")

    elapsed = time.time() - t0
    print(f"\nEncoded {len(encoded_ids)}/{len(plate_ids)} plates in {elapsed:.1f}s.")

    if len(encoded_ids) < 4:
        raise RuntimeError(
            f"Need at least 4 latents to fit a 3-component PCA, got {len(encoded_ids)}.")

    X = np.stack(latents).astype(np.float64)
    n_components = min(3, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    print("Explained variance: " + ", ".join(
        f"PC{i + 1}={v:.2%}" for i, v in enumerate(var)))

    dump = {
        "method": "pca",
        "fit_on": "training",
        "n_components": int(n_components),
        "n_train": int(X.shape[0]),
        "mean": pca.mean_.tolist(),
        "components": pca.components_.tolist(),
        "explained_variance_ratio": var.tolist(),
        "plate_ids": encoded_ids,
        "proj": proj.tolist(),
        "seeds_file": args.seeds_file,
        "rve_bank": args.rve_bank,
        "elapsed_s": elapsed,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dump, f)
    print(f"Saved {args.output}")

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _run_encode_heldout(args):
    if args.heldout_file is not None:
        with open(args.heldout_file) as f:
            plate_ids = json.load(f)["plates"]
    else:
        plate_ids = list(MICROPLATE_LATTICE.splits.heldout_plates)
    print(f"Held-out plates: {plate_ids}")

    print(f"Loading ModelEngine on {args.device}...")
    cfg = ServingConfig()
    cfg.device = args.device
    engine = ModelEngine(cfg)

    records = []
    for pid in plate_ids:
        result = encode_plate_meanpool(engine, cfg, pid, rve_bank_dir=args.rve_bank)
        if result is None:
            print(f"  {pid}: SKIP (mesh or encode failed)")
            continue
        latent, n_nodes = result
        records.append({
            "id": pid,
            "latent": latent.tolist(),
            "mesh_n_nodes": n_nodes,
        })
        print(f"  {pid}: ok  ({n_nodes} nodes, latent norm={np.linalg.norm(latent):.3f})")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"plates": records, "n": len(records)}, f)
    print(f"\nSaved {args.output}  ({len(records)}/{len(plate_ids)} encoded)")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit-pca",
                           help="Fit 3-component PCA on training-plate latents.")
    p_fit.add_argument("--seeds-file", type=str, default=None,
                       help="JSON manifest with a 'plates' list. Default: "
                            "MICROPLATE_LATTICE.splits.training_plates.")
    p_fit.add_argument("--rve-bank", type=str, default="data/rve_bank")
    p_fit.add_argument("--max-seeds", type=int, default=None,
                       help="Cap on number of seeds for pilots.")
    p_fit.add_argument("--device", type=str, default="cuda:0")
    p_fit.add_argument("--output", type=str, default=None,
                       help="Override output path; if --run-root is set, "
                            "defaults to <run-root>/pca.json.")
    add_run_root_arg(p_fit)

    p_he = sub.add_parser("encode-heldout",
                          help="Encode the 8 held-out plates' mean-pooled latents.")
    p_he.add_argument("--heldout-file", type=str, default=None,
                      help="JSON manifest with a 'plates' list. Default: "
                           "MICROPLATE_LATTICE.splits.heldout_plates.")
    p_he.add_argument("--rve-bank", type=str, default="data/rve_bank")
    p_he.add_argument("--output", type=str, default=None,
                      help="Override output path; if --run-root is set, "
                           "defaults to <run-root>/heldout_latents.json.")
    p_he.add_argument("--device", type=str, default="cuda:0")
    add_run_root_arg(p_he)

    args = parser.parse_args()

    if args.cmd == "fit-pca":
        if args.output is None:
            if args.run_root is None:
                parser.error("either --output or --run-root is required")
            args.output = resolve(args.run_root, "pca")
        _run_fit_pca(args)
    elif args.cmd == "encode-heldout":
        if args.output is None:
            if args.run_root is None:
                parser.error("either --output or --run-root is required")
            args.output = resolve(args.run_root, "heldout_latents")
        _run_encode_heldout(args)


if __name__ == "__main__":
    main()
