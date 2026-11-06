#!/usr/bin/env python3
"""Stages 9 and 14 of the Figure 5 pipeline (figure5.sh): held-out plates.

For each of the 8 held-out plates (mg_086, mg_310, mg_323, mg_340, mg_342,
mg_356, mg_388, mg_431), runs:

  1. Build a tiny candidates.json (8 entries).
  2. LEIA characterization (`landscape.score --metric=score`) — disp + VM
     + mesh + recon_error + latent_meanpool, saved to leia_outputs/.
  3. FEM ground truth (`landscape.fem`) — disp + VM under DIRECTIONS_2.
     Computes vm_corr_mean per plate.
  4. Dynamics consistency (`landscape.score --metric=consistency`) —
     c_per_frame for the 10-frame uniaxial-stretch rollout.
  5. Apply the trained OOD head (`full` config) on the 8 plates.
  6. Save heldout_predictions.jsonl + heldout_eval.md.

Reuses existing infrastructure; this script just orchestrates and applies
the trained head.

Usage:
  python -m design_search.landscape.heldout \\
      --output-dir design_search/outputs/landscape_heldout \\
      --ood-ckpt   design_search/outputs/landscape_ood/ckpts/full/best.pt
"""

import argparse
import json
import os
import subprocess

import numpy as np
import torch

from design_search.landscape.ood.head import OODConfidenceHead
from design_search.landscape.walk import load_seed_graph_permissive
from leia.regime import MICROPLATE_LATTICE


def build_candidates_json(heldout_ids, rve_bank, out_path):
    """Construct a tiny candidates.json with the 8 held-out plates."""
    cands = []
    for pid in heldout_ids:
        graph, errs = load_seed_graph_permissive(pid, rve_bank_dir=rve_bank)
        if errs:
            print(f"  {pid}: {len(errs)} validation warnings (kept)")
        cands.append({
            "id": pid,
            "seed_plate": pid,
            "depth": 0,
            "parent_id": None,
            "graph": graph,
        })
    with open(out_path, "w") as f:
        json.dump({"candidates": cands}, f)
    print(f"Saved {out_path} ({len(cands)} plates)")
    return out_path


def run_subprocess(cmd, stage_name):
    print(f"\n=== {stage_name} ===")
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout[-2000:]}")
        print(f"STDERR:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"{stage_name} failed (rc={result.returncode})")
    print(result.stdout[-500:])


def compute_graph_stats(heldout_ids, rve_bank):
    from leia.microgen import compute_features
    out = {}
    for pid in heldout_ids:
        graph, _ = load_seed_graph_permissive(pid, rve_bank_dir=rve_bank)
        out[pid] = compute_features(graph).astype(np.float32)
    return out


def load_jsonl(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output dir; if --run-root is set, "
                             "defaults to <root>/heldout/.")
    parser.add_argument("--rve-bank", type=str, default="data/rve_bank")
    parser.add_argument("--heldout-file", type=str, default=None,
                        help="JSON manifest with a 'plates' list. Default: "
                             "MICROPLATE_LATTICE.splits.heldout_plates.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="GPU for LEIA pass + OOD head inference.")
    parser.add_argument("--fem-workers", type=int, default=8,
                        help="FEM workers (one job per held-out plate; both "
                             "directions are solved inside the job).")
    parser.add_argument("--ood-ckpt", type=str, default=None,
                        help="Override OOD checkpoint; if --run-root is set, "
                             "defaults to <root>/ood/ckpts/full/best.pt.")
    parser.add_argument(
        "--consistency-file-out", type=str, default=None,
        help="Override the consistency.jsonl path; defaults to "
             "<output-dir>/consistency.jsonl.")
    from design_search.landscape.layout import add_run_root_arg, resolve
    add_run_root_arg(parser)
    args = parser.parse_args()

    if args.output_dir is None:
        if args.run_root is None:
            parser.error("either --output-dir or --run-root is required")
        args.output_dir = resolve(args.run_root, "heldout_dir")
    if args.ood_ckpt is None:
        if args.run_root is None:
            parser.error("either --ood-ckpt or --run-root is required")
        args.ood_ckpt = resolve(args.run_root, "ood_ckpt_full")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.heldout_file is not None:
        with open(args.heldout_file) as f:
            heldout_ids = json.load(f)["plates"]
    else:
        heldout_ids = list(MICROPLATE_LATTICE.splits.heldout_plates)
    print(f"Held-out plates: {heldout_ids}")

    # 1. Build candidates.json
    cand_path = os.path.join(args.output_dir, "candidates.json")
    build_candidates_json(heldout_ids, args.rve_bank, cand_path)

    # 2. LEIA characterization
    run_subprocess([
        "python", "-m", "design_search.landscape.score",
        "--metric=score",
        "--candidates", cand_path,
        "--output-dir", args.output_dir,
        "--device", args.device,
        "--shard-id", "0",
        "--num-shards", "1",
    ], "Stage 5.2: LEIA characterization")

    # Merge shard JSONL → leia_scores.jsonl (single-shard run, just rename)
    shard_path = os.path.join(args.output_dir, "leia_scores_shard0.jsonl")
    leia_path = os.path.join(args.output_dir, "leia_scores.jsonl")
    if os.path.exists(shard_path) and not os.path.exists(leia_path):
        os.rename(shard_path, leia_path)

    # 3. FEM ground truth
    run_subprocess([
        "env", "OMPI_MCA_btl=self,tcp",
        "python", "-m", "design_search.landscape.fem",
        "--leia-dir", args.output_dir,
        "--workers", str(args.fem_workers),
        "--batch-size", "16",
    ], "Stage 5.3: FEM ground truth")

    # 4. Dynamics consistency on the 8 plates
    consistency_dir = args.output_dir
    consistency_out = (args.consistency_file_out
                       or os.path.join(consistency_dir, "consistency.jsonl"))
    # The consistency module shards; we want a single shard of all 8.
    run_subprocess([
        "python", "-m", "design_search.landscape.score",
        "--metric=consistency",
        "--output-dir", args.output_dir,
        "--landscape-dirs", args.output_dir,
        "--n-steps", "10",
        "--shard-id", "0",
        "--num-shards", "1",
        "--device", args.device,
    ], "Stage 5.4: dynamics consistency")
    # Move consistency_shard0.jsonl → consistency.jsonl
    shard0 = os.path.join(args.output_dir, "consistency_shard0.jsonl")
    if os.path.exists(shard0) and not os.path.exists(consistency_out):
        os.rename(shard0, consistency_out)

    # 5. Compute graph stats + apply trained OOD head
    print("\n=== Stage 5.5: applying trained OOD head ===")
    g_stats = compute_graph_stats(heldout_ids, args.rve_bank)
    leia_scores = load_jsonl(leia_path)
    fem_scores = load_jsonl(os.path.join(args.output_dir, "fem_scores.jsonl"))
    cons = load_jsonl(consistency_out)

    # Load latents from leia_outputs/<id>.npz
    latents = {}
    for pid in heldout_ids:
        npz_path = os.path.join(args.output_dir, "leia_outputs", f"{pid}.npz")
        with np.load(npz_path) as z:
            latents[pid] = z["latent_meanpool"].astype(np.float32)

    # Load OOD head
    ckpt = torch.load(args.ood_ckpt, weights_only=False, map_location=args.device)
    model = OODConfidenceHead(
        d_in=ckpt["d_in"], d_hidden=ckpt["d_hidden"],
        n_bins=ckpt["n_bins"], dropout=ckpt["dropout"],
        bin_centers=ckpt["bin_centers"],
    ).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    feat_idx = ckpt["feature_idx"]
    feat_mean = ckpt["feat_mean_slice"]
    feat_std = ckpt["feat_std_slice"]
    print(f"OOD head: {ckpt['config']} (d_in={ckpt['d_in']}); val_xent {ckpt['val_xent']:.4f}")

    # Build feature vectors in the same 1050-d order as the training dataset.
    rows = []
    for pid in heldout_ids:
        z = latents[pid]                                                  # (1024,)
        recon = float(leia_scores[pid]["recon_error"])
        c_arr = np.array(cons[pid]["c_per_frame"], dtype=np.float32)      # (10,)
        gs = g_stats[pid]                                                 # (15,)
        feat_full = np.concatenate([z, [recon], c_arr, gs]).astype(np.float32)
        feat_slice = (feat_full[feat_idx] - feat_mean) / feat_std
        with torch.no_grad():
            x = torch.from_numpy(feat_slice).unsqueeze(0).to(args.device)
            scalar, entropy, probs = model.predict(x)
        true_vm = (float(fem_scores[pid]["vm_corr_mean"])
                   if fem_scores.get(pid, {}).get("fem_status") == "ok" else None)
        rows.append({
            "id": pid,
            "predicted_vm_corr": float(scalar.item()),
            "predicted_entropy": float(entropy.item()),
            "true_vm_corr": true_vm,
            "true_vm_corr_stretch":
                fem_scores.get(pid, {}).get("vm_corr_stretch"),
            "true_vm_corr_shear":
                fem_scores.get(pid, {}).get("vm_corr_shear"),
            "fem_status": fem_scores.get(pid, {}).get("fem_status"),
            "recon_error": recon,
            "consistency_mean": float(c_arr.mean()),
            "consistency_t3": float(c_arr[2]),
        })

    out_jsonl = os.path.join(args.output_dir, "heldout_predictions.jsonl")
    with open(out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {out_jsonl}")

    # 6. Aggregate metrics + report
    valid = [r for r in rows if r["true_vm_corr"] is not None
             and np.isfinite(r["true_vm_corr"])]
    md_path = os.path.join(args.output_dir, "heldout_eval.md")
    with open(md_path, "w") as f:
        f.write("# Held-out (8 plates) — OOD head validation\n\n")
        f.write("| plate | true vm_corr | predicted | entropy | "
                "recon_err | c_mean | c_t3 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            tv = "—" if r["true_vm_corr"] is None else f"{r['true_vm_corr']:.4f}"
            f.write(f"| {r['id']} | {tv} | {r['predicted_vm_corr']:.4f} | "
                    f"{r['predicted_entropy']:.3f} | {r['recon_error']:.2e} | "
                    f"{r['consistency_mean']:.4f} | {r['consistency_t3']:.4f} |\n")
        if len(valid) >= 3:
            from scipy.stats import spearmanr, pearsonr
            tv = np.array([r["true_vm_corr"] for r in valid])
            pv = np.array([r["predicted_vm_corr"] for r in valid])
            sp = spearmanr(tv, pv)[0]
            pe = pearsonr(tv, pv)[0]
            mse = float(np.mean((tv - pv) ** 2))
            f.write(f"\n**Aggregate** (n={len(valid)}):  "
                    f"Spearman={sp:+.4f}, Pearson={pe:+.4f}, RMSE={np.sqrt(mse):.4f}\n")
        else:
            f.write(f"\nOnly {len(valid)} plates with valid FEM truth — "
                    "aggregate metrics omitted.\n")
    print(f"Saved {md_path}")
    # Print to stdout for the user
    with open(md_path) as f:
        print("\n" + f.read())


if __name__ == "__main__":
    main()
