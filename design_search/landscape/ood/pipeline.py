#!/usr/bin/env python3
"""OOD-confidence head pipeline — build, train, eval, predict.

Four subcommands sharing feature-loaders + dataset I/O:

  build
      Assemble the (features, target) dataset for the 6,919 FEM-validated
      candidates. Writes <ood_dir>/dataset.npz with seed-stratified 80/20
      split (44 train seeds / 11 val seeds) and per-feature normalization
      stats. Features per candidate (1050-d total):
        - z_0_meanpool   1024-d  (latent_meanpool from leia_outputs/<id>.npz)
        - recon_error      1-d   (leia_scores.jsonl)
        - c_per_frame     10-d   (consistency.jsonl)
        - graph_stats     15-d   (leia.microgen.compute_features)

  train
      Trains 3 ablation configs ("free", "latent", "full") with Adam,
      cross-entropy on bin_centers, early stop on val xent (patience=30).
      Writes <ood_dir>/ckpts/<config>/{best,final}.pt + train_<config>.log
      + train_<config>_history.json.

  eval
      Per-config val Spearman/Pearson/RMSE, baseline comparison
      (recon_error / consistency_mean / consistency_t3), calibration plot,
      stripe diagnosis, and full-set inference using the best config.
      Writes <ood_dir>/eval_summary.md + calibration_<cfg>.png +
      predictions.jsonl.

  predict
      predict — run the trained "full" config on ALL candidates (no FEM
      filter). Writes <ood_dir>/predictions_full.jsonl.

Usage:
  python -m design_search.landscape.ood.pipeline build   --run-root <root>
  python -m design_search.landscape.ood.pipeline train   --run-root <root>
  python -m design_search.landscape.ood.pipeline eval    --run-root <root>
  python -m design_search.landscape.ood.pipeline predict --run-root <root>
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from design_search.landscape.layout import add_run_root_arg, resolve
from design_search.landscape.ood.head import OODConfidenceHead


CONFIGS = ["free", "latent", "full"]


# ---------------------------------------------------------------------------
# Shared feature loaders
# ---------------------------------------------------------------------------

def load_leia_meta(landscape_dirs):
    """Returns dict id -> {seed_plate, depth, recon_error, source_dir}."""
    out = {}
    for d in landscape_dirs:
        path = os.path.join(d, "leia_scores.jsonl")
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("status") != "ok":
                    continue
                out[r["id"]] = {
                    "seed_plate": r["seed_plate"],
                    "depth": r["depth"],
                    "recon_error": float(r["recon_error"]),
                    "source_dir": d,
                }
    return out


def load_fem_targets(landscape_dirs):
    """Returns dict id -> vm_corr_mean (only fem_status=='ok')."""
    out = {}
    for d in landscape_dirs:
        path = os.path.join(d, "fem_scores.jsonl")
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("fem_status") != "ok":
                    continue
                v = r.get("vm_corr_mean")
                if v is None or not np.isfinite(v):
                    continue
                out[r["id"]] = float(v)
    return out


def load_consistency(jsonl_path):
    """Returns dict id -> 10-d c_per_frame array."""
    out = {}
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            cs = r.get("c_per_frame")
            if cs is None or len(cs) != 10:
                continue
            arr = np.array(cs, dtype=np.float32)
            if not np.isfinite(arr).all():
                continue
            out[r["id"]] = arr
    return out


def load_latent_meanpools(ids, leia_meta):
    """Returns dict id -> (1024,) mean-pooled latent."""
    out = {}
    for cid in ids:
        npz_path = os.path.join(leia_meta[cid]["source_dir"],
                                "leia_outputs", f"{cid}.npz")
        try:
            with np.load(npz_path) as z:
                lat = z["latent_meanpool"].astype(np.float32)
            if lat.shape != (1024,):
                continue
            out[cid] = lat
        except (OSError, KeyError):
            continue
    return out


def load_graphs(landscape_dirs):
    """Returns dict id -> graph dict from candidates.json across landscape_dirs."""
    out = {}
    for d in landscape_dirs:
        path = os.path.join(d, "candidates.json")
        with open(path) as f:
            cd = json.load(f)
        for c in cd["candidates"]:
            out[c["id"]] = c["graph"]
    return out


def compute_graph_stats(graphs_by_id, ids):
    """Returns (N, 15) graph-stats matrix in same order as `ids`."""
    from leia.microgen import compute_features

    out = np.zeros((len(ids), 15), dtype=np.float32)
    for i, cid in enumerate(ids):
        g = graphs_by_id.get(cid)
        if g is None:
            continue
        try:
            out[i] = compute_features(g)
        except Exception as e:
            print(f"  graph_stats failed for {cid}: {e}")
    return out


def seed_stratified_split(seeds_per_id: dict, val_frac: float, rng_seed: int):
    """Pick val_frac of unique seeds as val. Returns (train_ids, val_ids, val_seeds)."""
    unique_seeds = sorted(set(seeds_per_id.values()))
    n_val = max(1, int(round(val_frac * len(unique_seeds))))
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(len(unique_seeds))
    val_seeds = set(unique_seeds[i] for i in perm[:n_val])
    train_ids = [i for i, s in seeds_per_id.items() if s not in val_seeds]
    val_ids = [i for i, s in seeds_per_id.items() if s in val_seeds]
    return train_ids, val_ids, val_seeds


def load_dataset(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model = OODConfidenceHead(
        d_in=ckpt["d_in"],
        d_hidden=ckpt["d_hidden"],
        n_bins=ckpt["n_bins"],
        dropout=ckpt["dropout"],
        bin_centers=ckpt["bin_centers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_all(model, X_normalized, device, batch_size=512):
    """Returns (scalars, entropies, probs) for all rows of X."""
    n = X_normalized.shape[0]
    scalars = np.zeros(n, dtype=np.float32)
    entropies = np.zeros(n, dtype=np.float32)
    probs = np.zeros((n, model.n_bins), dtype=np.float32)
    for s in range(0, n, batch_size):
        x = torch.from_numpy(X_normalized[s:s + batch_size].astype(np.float32)).to(device)
        sc, ent, pr = model.predict(x)
        scalars[s:s + batch_size] = sc.cpu().numpy()
        entropies[s:s + batch_size] = ent.cpu().numpy()
        probs[s:s + batch_size] = pr.cpu().numpy()
    return scalars, entropies, probs


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _run_build(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading leia metadata...")
    leia_meta = load_leia_meta(args.landscape_dirs)
    print(f"  {len(leia_meta)} candidates with status==ok")

    print("Loading FEM targets...")
    targets = load_fem_targets(args.landscape_dirs)
    print(f"  {len(targets)} candidates with fem_status==ok and finite vm_corr_mean")

    print("Loading c_per_frame...")
    c_by_id = load_consistency(args.consistency_file)
    print(f"  {len(c_by_id)} candidates with valid c_per_frame")

    ids_complete = sorted(set(leia_meta) & set(targets) & set(c_by_id))
    print(f"  intersection (leia ∩ fem ∩ consistency) = {len(ids_complete)}")

    print("Loading latent_meanpool from npz files...")
    latents = load_latent_meanpools(ids_complete, leia_meta)
    print(f"  {len(latents)} candidates with valid latent_meanpool")

    ids = sorted(set(latents) & set(ids_complete))
    print(f"  final usable ids: {len(ids)}")

    print("Loading graphs and computing graph_stats (15-d)...")
    graphs = load_graphs(args.landscape_dirs)
    g_stats = compute_graph_stats(graphs, ids)
    print(f"  graph_stats matrix: {g_stats.shape}")

    n = len(ids)
    z = np.stack([latents[i] for i in ids], axis=0).astype(np.float32)
    recon = np.array([leia_meta[i]["recon_error"] for i in ids],
                     dtype=np.float32)[:, None]
    c_mat = np.stack([c_by_id[i] for i in ids], axis=0).astype(np.float32)
    g_mat = g_stats.astype(np.float32)
    X = np.concatenate([z, recon, c_mat, g_mat], axis=1)
    print(f"X shape: {X.shape}")

    y = np.array([targets[i] for i in ids], dtype=np.float32)
    seeds = [leia_meta[i]["seed_plate"] for i in ids]
    seeds_per_id = {i: s for i, s in zip(ids, seeds)}

    bin_edges = np.linspace(args.bin_min, args.bin_max,
                            args.n_bins + 1).astype(np.float32)
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).astype(np.float32)
    y_clipped = np.clip(y, args.bin_min + 1e-6, args.bin_max - 1e-6)
    y_bin = np.digitize(y_clipped, bin_edges) - 1
    y_bin = np.clip(y_bin, 0, args.n_bins - 1).astype(np.int64)

    train_ids, val_ids, val_seeds = seed_stratified_split(
        seeds_per_id, args.val_frac, args.rng_seed)
    train_idx = np.array([ids.index(i) for i in train_ids], dtype=np.int64)
    val_idx = np.array([ids.index(i) for i in val_ids], dtype=np.int64)
    print(f"Split: {len(train_idx)} train ({len(set(seeds[k] for k in train_idx))} seeds), "
          f"{len(val_idx)} val ({len(val_seeds)} seeds)")
    print(f"  val seeds: {sorted(val_seeds)}")

    feat_mean = X[train_idx].mean(axis=0).astype(np.float32)
    feat_std = X[train_idx].std(axis=0).astype(np.float32)
    feat_std[feat_std < 1e-6] = 1.0

    idx_latent = np.arange(0, 1024, dtype=np.int64)
    idx_recon = np.array([1024], dtype=np.int64)
    idx_c = np.arange(1025, 1035, dtype=np.int64)
    idx_graph = np.arange(1035, 1050, dtype=np.int64)
    idx_free = np.concatenate([idx_recon, idx_c])
    idx_full = np.arange(0, 1050, dtype=np.int64)

    out_path = os.path.join(args.output_dir, "dataset.npz")
    np.savez(
        out_path,
        ids=np.array(ids, dtype=object),
        seeds=np.array(seeds, dtype=object),
        X_train=X[train_idx], y_train=y[train_idx], y_bin_train=y_bin[train_idx],
        ids_train=np.array([ids[k] for k in train_idx], dtype=object),
        seeds_train=np.array([seeds[k] for k in train_idx], dtype=object),
        X_val=X[val_idx], y_val=y[val_idx], y_bin_val=y_bin[val_idx],
        ids_val=np.array([ids[k] for k in val_idx], dtype=object),
        seeds_val=np.array([seeds[k] for k in val_idx], dtype=object),
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        feat_mean=feat_mean,
        feat_std=feat_std,
        idx_latent=idx_latent,
        idx_recon=idx_recon,
        idx_c=idx_c,
        idx_graph=idx_graph,
        idx_free=idx_free,
        idx_full=idx_full,
    )
    print(f"\nSaved {out_path}")
    print(f"  X_train: {X[train_idx].shape}, y_train: {y[train_idx].shape}")
    print(f"  X_val:   {X[val_idx].shape}, y_val:   {y[val_idx].shape}")
    print(f"  bin counts on train: "
          f"{np.bincount(y_bin[train_idx], minlength=args.n_bins).tolist()}")
    print(f"  bin counts on val:   "
          f"{np.bincount(y_bin[val_idx], minlength=args.n_bins).tolist()}")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def _build_loaders(d, idx, feat_mean, feat_std, batch_size):
    mean_s = feat_mean[idx]
    std_s = feat_std[idx]

    Xtr = (d["X_train"][:, idx] - mean_s) / std_s
    Xva = (d["X_val"][:, idx] - mean_s) / std_s
    train_ds = TensorDataset(
        torch.from_numpy(Xtr.astype(np.float32)),
        torch.from_numpy(d["y_bin_train"].astype(np.int64)),
        torch.from_numpy(d["y_train"].astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(Xva.astype(np.float32)),
        torch.from_numpy(d["y_bin_val"].astype(np.int64)),
        torch.from_numpy(d["y_val"].astype(np.float32)),
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False),
    )


@torch.no_grad()
def _eval_pass(model, loader, device):
    from scipy.stats import spearmanr, pearsonr
    model.eval()
    total_xent = 0.0
    total_n = 0
    pred_scalars = []
    true_scalars = []
    for X, y_bin, y in loader:
        X = X.to(device)
        y_bin = y_bin.to(device)
        logits = model(X)
        xent = F.cross_entropy(logits, y_bin, reduction="sum")
        total_xent += xent.item()
        total_n += X.shape[0]
        scalar, _, _ = model.predict(X)
        pred_scalars.append(scalar.cpu().numpy())
        true_scalars.append(y.numpy())
    pred = np.concatenate(pred_scalars)
    true = np.concatenate(true_scalars)
    return total_xent / total_n, float(spearmanr(pred, true)[0]), float(pearsonr(pred, true)[0])


def _train_one(config_name, dataset, output_dir, device, args):
    print(f"\n{'='*60}")
    print(f"Training config: {config_name}")
    print('='*60)

    idx = dataset[f"idx_{config_name}"]
    d_in = len(idx)
    n_bins = len(dataset["bin_centers"])
    bin_centers = torch.from_numpy(dataset["bin_centers"]).float()

    train_loader, val_loader = _build_loaders(
        dataset, idx, dataset["feat_mean"], dataset["feat_std"], args.batch_size)
    print(f"  d_in={d_in}, n_bins={n_bins}, "
          f"|train|={len(train_loader.dataset)}, |val|={len(val_loader.dataset)}")

    model = OODConfidenceHead(
        d_in=d_in, d_hidden=args.d_hidden, n_bins=n_bins,
        dropout=args.dropout, bin_centers=bin_centers,
    ).to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters())}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    ckpt_dir = os.path.join(output_dir, "ckpts", config_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_file = open(os.path.join(output_dir, f"train_{config_name}.log"), "w")

    best_val_xent = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    history = []

    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_xent_sum = 0.0
        train_n = 0
        for X, y_bin, _ in train_loader:
            X = X.to(device)
            y_bin = y_bin.to(device)
            logits = model(X)
            loss = F.cross_entropy(logits, y_bin)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_xent_sum += loss.item() * X.shape[0]
            train_n += X.shape[0]
        train_xent = train_xent_sum / train_n

        val_xent, val_spearman, val_pearson = _eval_pass(model, val_loader, device)
        elapsed = time.time() - t0

        improved = val_xent < best_val_xent - 1e-5
        ckpt_payload = {
            "config": config_name,
            "epoch": epoch,
            "val_xent": val_xent,
            "val_spearman": val_spearman,
            "model_state": model.state_dict(),
            "d_in": d_in,
            "n_bins": n_bins,
            "d_hidden": args.d_hidden,
            "dropout": args.dropout,
            "bin_centers": bin_centers,
            "feat_mean_slice": dataset["feat_mean"][idx],
            "feat_std_slice": dataset["feat_std"][idx],
            "feature_idx": idx,
        }
        if improved:
            best_val_xent = val_xent
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save(ckpt_payload, os.path.join(ckpt_dir, "best.pt"))
        else:
            epochs_since_improve += 1

        log_line = (f"epoch {epoch:3d}  train_xent={train_xent:.4f}  "
                    f"val_xent={val_xent:.4f}  val_spearman={val_spearman:+.4f}  "
                    f"val_pearson={val_pearson:+.4f}  "
                    f"{'*' if improved else ' '}  ({elapsed:.2f}s)")
        print(log_line)
        log_file.write(log_line + "\n")
        log_file.flush()
        history.append({
            "epoch": epoch, "train_xent": train_xent,
            "val_xent": val_xent, "val_spearman": val_spearman,
            "val_pearson": val_pearson,
        })

        if epochs_since_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch}.")
            break

    total_elapsed = time.time() - t_start
    print(f"\n  Best val cross-entropy: {best_val_xent:.4f} at epoch {best_epoch}")
    print(f"  Total training time: {total_elapsed:.1f}s")

    # final.pt mirrors the last-epoch payload
    ckpt_payload["epoch"] = epoch
    ckpt_payload["val_xent"] = val_xent
    torch.save(ckpt_payload, os.path.join(ckpt_dir, "final.pt"))

    def _to_py(o):
        if isinstance(o, dict):
            return {k: _to_py(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_to_py(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    with open(os.path.join(output_dir, f"train_{config_name}_history.json"), "w") as f:
        json.dump(_to_py({
            "config": config_name,
            "best_epoch": best_epoch,
            "best_val_xent": best_val_xent,
            "history": history,
            "total_elapsed_s": total_elapsed,
        }), f, indent=2)
    log_file.close()
    return best_val_xent, best_epoch


def _run_train(args):
    seed = int(getattr(args, "rng_seed", 20260428))
    import random as _random
    _random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = load_dataset(args.dataset)
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Configs to train: {args.configs}")

    summary = {}
    for cfg in args.configs:
        best_xent, best_epoch = _train_one(cfg, dataset, args.output_dir, device, args)
        summary[cfg] = {"best_val_xent": float(best_xent),
                        "best_epoch": int(best_epoch)}

    print("\n" + "="*60)
    print("Training summary:")
    for cfg, s in summary.items():
        print(f"  {cfg:8s}: best_val_xent={s['best_val_xent']:.4f}  "
              f"(epoch {s['best_epoch']})")
    print("="*60)


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def _calibration_plot(probs, y_bin, n_bins, out_path):
    pred_bin = probs.argmax(axis=1)
    pred_conf = probs.max(axis=1)
    n_buckets = 10
    bucket_edges = np.linspace(0.0, 1.0, n_buckets + 1)
    bucket_centers = (bucket_edges[:-1] + bucket_edges[1:]) / 2
    fractions = np.zeros(n_buckets)
    counts = np.zeros(n_buckets, dtype=np.int64)
    for b in range(n_buckets):
        mask = (pred_conf >= bucket_edges[b]) & (pred_conf < bucket_edges[b + 1])
        if b == n_buckets - 1:
            mask |= pred_conf == bucket_edges[-1]
        if mask.sum() == 0:
            continue
        correct = (pred_bin[mask] == y_bin[mask]).mean()
        fractions[b] = correct
        counts[b] = mask.sum()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(bucket_centers, fractions, "o-", markersize=8, label="Observed accuracy")
    for c, f, n in zip(bucket_centers, fractions, counts):
        if n > 0:
            ax.annotate(f"n={n}", xy=(c, f), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel("Predicted confidence (max bin prob)")
    ax.set_ylabel("Observed accuracy (frac in argmax bin)")
    ax.set_title("Reliability diagram (val)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fractions, counts


def _stripe_diagnosis(scalars_full, ydata_full, pca_proj, stripe_box):
    pc1, pc2 = pca_proj[:, 0], pca_proj[:, 1]
    in_stripe = ((pc1 >= stripe_box[0]) & (pc1 <= stripe_box[1])
                 & (pc2 >= stripe_box[2]) & (pc2 <= stripe_box[3]))
    out_stripe = ~in_stripe
    return {
        "n_in_stripe": int(in_stripe.sum()),
        "n_out_stripe": int(out_stripe.sum()),
        "true_vm_in_stripe":  float(np.mean(ydata_full[in_stripe])) if in_stripe.any() else None,
        "true_vm_out_stripe": float(np.mean(ydata_full[out_stripe])) if out_stripe.any() else None,
        "pred_vm_in_stripe":  float(np.mean(scalars_full[in_stripe])) if in_stripe.any() else None,
        "pred_vm_out_stripe": float(np.mean(scalars_full[out_stripe])) if out_stripe.any() else None,
    }


def _run_eval(args):
    from scipy.stats import pearsonr, spearmanr

    dataset = load_dataset(args.dataset)
    device = torch.device(args.device)

    rows = []
    config_results = {}
    best_config = None
    best_spearman = -np.inf

    for cfg in CONFIGS:
        ckpt_path = os.path.join(args.output_dir, "ckpts", cfg, "best.pt")
        model, ckpt = load_ckpt(ckpt_path, device)

        idx = ckpt["feature_idx"]
        mean_s = ckpt["feat_mean_slice"]
        std_s = ckpt["feat_std_slice"]
        Xva = (dataset["X_val"][:, idx] - mean_s) / std_s
        yva = dataset["y_val"]
        yva_bin = dataset["y_bin_val"]

        scalars, entropies, probs = predict_all(model, Xva, device)
        sp = float(spearmanr(scalars, yva)[0])
        pe = float(pearsonr(scalars, yva)[0])
        rmse = float(np.sqrt(np.mean((scalars - yva) ** 2)))
        log_probs = np.log(np.clip(probs, 1e-12, 1.0))
        xent = float(-np.mean(log_probs[np.arange(len(yva_bin)), yva_bin]))

        cal_path = os.path.join(args.output_dir, f"calibration_{cfg}.png")
        _calibration_plot(probs, yva_bin, model.n_bins, cal_path)

        rows.append({
            "config": cfg,
            "d_in": int(ckpt["d_in"]),
            "epoch": int(ckpt["epoch"]),
            "val_xent": xent,
            "val_spearman": sp,
            "val_pearson": pe,
            "val_rmse": rmse,
        })
        config_results[cfg] = {"scalars_val": scalars, "probs_val": probs,
                               "entropies_val": entropies}
        if sp > best_spearman:
            best_spearman = sp
            best_config = cfg

    yva = dataset["y_val"]
    recon_val = dataset["X_val"][:, 1024]
    cmean_val = dataset["X_val"][:, 1025:1035].mean(axis=1)
    ct3_val = dataset["X_val"][:, 1027]
    baseline_table = {
        "recon_error": float(spearmanr(recon_val, yva)[0]),
        "consistency_mean": float(spearmanr(cmean_val, yva)[0]),
        "consistency_t3": float(spearmanr(ct3_val, yva)[0]),
    }

    # Stripe diagnosis on the FULL set with PCA projections
    stripe_summary = {"requested": False}
    pca_proj_for_full = None
    if args.pca_file and os.path.exists(args.pca_file):
        with open(args.pca_file) as f:
            pca_data = json.load(f)
        pca_ids = list(pca_data["ids"])
        pca_proj = np.array(pca_data["proj"])
        pca_idx_by_id = {i: k for k, i in enumerate(pca_ids)}
        all_ids = list(dataset["ids"])
        keep = [i for i, cid in enumerate(all_ids) if cid in pca_idx_by_id]
        if keep:
            pca_proj_for_full = np.array([pca_proj[pca_idx_by_id[all_ids[k]]]
                                          for k in keep])
            stripe_summary["requested"] = True

    if args.predict_best_on_full:
        ckpt_path = os.path.join(args.output_dir, "ckpts", best_config, "best.pt")
        model, ckpt = load_ckpt(ckpt_path, device)
        idx = ckpt["feature_idx"]
        mean_s = ckpt["feat_mean_slice"]
        std_s = ckpt["feat_std_slice"]

        X_full = np.concatenate([dataset["X_train"], dataset["X_val"]], axis=0)
        y_full = np.concatenate([dataset["y_train"], dataset["y_val"]], axis=0)
        ids_full = np.concatenate([dataset["ids_train"], dataset["ids_val"]])
        seeds_full = np.concatenate([dataset["seeds_train"], dataset["seeds_val"]])
        from_train = np.concatenate([
            np.ones(len(dataset["X_train"]), dtype=bool),
            np.zeros(len(dataset["X_val"]), dtype=bool),
        ])

        Xn = (X_full[:, idx] - mean_s) / std_s
        scalars, entropies, probs = predict_all(model, Xn, device)

        out_path = os.path.join(args.output_dir, "predictions.jsonl")
        with open(out_path, "w") as f_out:
            for k in range(len(ids_full)):
                rec = {
                    "id": str(ids_full[k]),
                    "seed_plate": str(seeds_full[k]),
                    "in_train_split": bool(from_train[k]),
                    "predicted_vm_corr": float(scalars[k]),
                    "predicted_entropy": float(entropies[k]),
                    "predicted_probs": probs[k].tolist(),
                    "true_vm_corr": float(y_full[k]),
                }
                f_out.write(json.dumps(rec) + "\n")
        print(f"Saved full-set predictions to {out_path}")

        if pca_proj_for_full is not None and len(pca_proj_for_full) == len(ids_full):
            stripe_summary.update(
                _stripe_diagnosis(scalars, y_full, pca_proj_for_full,
                                  args.stripe_box))
            stripe_summary["box"] = list(args.stripe_box)

    md_path = os.path.join(args.output_dir, "eval_summary.md")
    with open(md_path, "w") as f:
        f.write("# OOD Confidence Head — Evaluation Summary\n\n")
        f.write(f"## Val-set metrics (seed-stratified val: "
                f"{len(dataset['y_val'])} candidates)\n\n")
        f.write("| Config | d_in | best_epoch | val_xent | val_spearman | val_pearson | val_rmse |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| `{r['config']}` | {r['d_in']} | {r['epoch']} | "
                    f"{r['val_xent']:.4f} | {r['val_spearman']:+.4f} | "
                    f"{r['val_pearson']:+.4f} | {r['val_rmse']:.4f} |\n")
        f.write("\n## Baseline Spearman against vm_corr_mean (val split)\n\n")
        f.write("| Signal | Spearman |\n|---|---:|\n")
        for k, v in baseline_table.items():
            f.write(f"| {k} | {v:+.4f} |\n")
        f.write(f"\nBest learned config: **`{best_config}`** with val Spearman "
                f"**{best_spearman:+.4f}**.\n")
        f.write("\n## Stripe diagnosis\n\n")
        if stripe_summary.get("requested"):
            box = stripe_summary["box"]
            f.write(f"Stripe box (PC1 ∈ [{box[0]}, {box[1]}], "
                    f"PC2 ∈ [{box[2]}, {box[3]}]):\n\n")
            f.write(f"- in stripe: n={stripe_summary['n_in_stripe']}, "
                    f"true_vm={stripe_summary['true_vm_in_stripe']:.4f}, "
                    f"pred_vm={stripe_summary['pred_vm_in_stripe']:.4f}\n")
            f.write(f"- out stripe: n={stripe_summary['n_out_stripe']}, "
                    f"true_vm={stripe_summary['true_vm_out_stripe']:.4f}, "
                    f"pred_vm={stripe_summary['pred_vm_out_stripe']:.4f}\n")
            true_gap = (stripe_summary['true_vm_out_stripe']
                        - stripe_summary['true_vm_in_stripe'])
            pred_gap = (stripe_summary['pred_vm_out_stripe']
                        - stripe_summary['pred_vm_in_stripe'])
            f.write(f"\nTrue vm gap (out − in): **{true_gap:+.4f}**;  "
                    f"predicted gap: **{pred_gap:+.4f}**.\n")
        else:
            f.write("No PCA file supplied; stripe diagnosis skipped.\n")

    print(f"Saved {md_path}")
    print("\n=== Final eval (val) ===")
    for r in rows:
        print(f"  {r['config']:8s}  d_in={r['d_in']:>4d}  "
              f"xent={r['val_xent']:.4f}  spearman={r['val_spearman']:+.4f}  "
              f"pearson={r['val_pearson']:+.4f}  rmse={r['val_rmse']:.4f}")
    print("\nBaselines:")
    for k, v in baseline_table.items():
        print(f"  {k:20s}  spearman={v:+.4f}")


# ---------------------------------------------------------------------------
# predict (full set, no FEM filter)
# ---------------------------------------------------------------------------

def _run_predict(args):
    print("Loading leia metadata...")
    leia_meta = load_leia_meta(args.landscape_dirs)
    print(f"  {len(leia_meta)} candidates with status==ok")

    print("Loading consistency...")
    c_by_id = load_consistency(args.consistency_file)
    print(f"  {len(c_by_id)} candidates with valid c_per_frame")

    ids_complete = sorted(set(leia_meta) & set(c_by_id))
    print(f"  intersection (leia ∩ consistency) = {len(ids_complete)}")

    print("Loading latent_meanpool...")
    latents = load_latent_meanpools(ids_complete, leia_meta)
    print(f"  {len(latents)} candidates with valid latent_meanpool")

    ids = sorted(set(latents) & set(ids_complete))
    print(f"  final usable: {len(ids)}")

    print("Loading graphs and computing graph_stats...")
    graphs = load_graphs(args.landscape_dirs)
    g_stats = compute_graph_stats(graphs, ids)
    print(f"  graph_stats: {g_stats.shape}")

    n = len(ids)
    z = np.stack([latents[i] for i in ids], axis=0).astype(np.float32)
    recon = np.array([leia_meta[i]["recon_error"] for i in ids],
                     dtype=np.float32)[:, None]
    c_mat = np.stack([c_by_id[i] for i in ids], axis=0).astype(np.float32)
    g_mat = g_stats.astype(np.float32)
    X_full = np.concatenate([z, recon, c_mat, g_mat], axis=1)
    print(f"X_full: {X_full.shape}")

    print(f"Loading {args.ckpt}...")
    ckpt = torch.load(args.ckpt, weights_only=False, map_location=args.device)
    feat_idx = ckpt["feature_idx"]
    feat_mean = ckpt["feat_mean_slice"]
    feat_std = ckpt["feat_std_slice"]
    assert len(feat_idx) == ckpt["d_in"]
    print(f"  config: {ckpt['config']}, d_in: {ckpt['d_in']}, "
          f"val_xent: {ckpt['val_xent']:.4f}")

    X_sliced = (X_full[:, feat_idx] - feat_mean) / feat_std
    print(f"X_sliced: {X_sliced.shape}")

    model = OODConfidenceHead(
        d_in=ckpt["d_in"], d_hidden=ckpt["d_hidden"],
        n_bins=ckpt["n_bins"], dropout=ckpt["dropout"],
        bin_centers=ckpt["bin_centers"],
    ).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Running inference on {n} candidates (batch={args.batch_size})...")
    scalars = np.zeros(n, dtype=np.float32)
    entropies = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, args.batch_size):
            x = torch.from_numpy(
                X_sliced[s:s + args.batch_size].astype(np.float32)
            ).to(args.device)
            sc, ent, _ = model.predict(x)
            scalars[s:s + args.batch_size] = sc.cpu().numpy()
            entropies[s:s + args.batch_size] = ent.cpu().numpy()

    assert np.isfinite(scalars).all(), "predicted scalars contain NaN/inf"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for k in range(n):
            rec = {
                "id": ids[k],
                "predicted_vm_corr": float(scalars[k]),
                "predicted_entropy": float(entropies[k]),
            }
            f.write(json.dumps(rec) + "\n")
    print(f"\nSaved {args.output}  ({n} records)")
    print(f"Predicted vm_corr range: [{scalars.min():.4f}, {scalars.max():.4f}], "
          f"mean={scalars.mean():.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # build
    pb = sub.add_parser("build", help="Assemble dataset.npz from per-stage outputs.")
    pb.add_argument("--landscape-dirs", nargs="+", default=None)
    pb.add_argument("--consistency-file", type=str, default=None)
    pb.add_argument("--output-dir", type=str, default=None)
    pb.add_argument("--val-frac", type=float, default=0.20)
    pb.add_argument("--rng-seed", type=int, default=20260428)
    pb.add_argument("--n-bins", type=int, default=20)
    pb.add_argument("--bin-min", type=float, default=-0.05)
    pb.add_argument("--bin-max", type=float, default=1.0)
    add_run_root_arg(pb)

    # train
    pt = sub.add_parser("train", help="Train 3 ablation configs.")
    pt.add_argument("--rng-seed", type=int, default=20260428,
                    help="Seeds torch/numpy/random (weight init, dropout, shuffling).")
    pt.add_argument("--dataset", type=str, default=None)
    pt.add_argument("--output-dir", type=str, default=None)
    pt.add_argument("--device", type=str, default="cuda:0")
    pt.add_argument("--configs", nargs="+", default=CONFIGS, choices=CONFIGS)
    pt.add_argument("--d-hidden", type=int, default=256)
    pt.add_argument("--dropout", type=float, default=0.2)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--weight-decay", type=float, default=1e-4)
    pt.add_argument("--batch-size", type=int, default=256)
    pt.add_argument("--epochs", type=int, default=200)
    pt.add_argument("--patience", type=int, default=30)
    add_run_root_arg(pt)

    # eval
    pe = sub.add_parser("eval", help="Per-config val metrics + figures + full-set inference.")
    pe.add_argument("--dataset", type=str, default=None)
    pe.add_argument("--output-dir", type=str, default=None)
    pe.add_argument("--device", type=str, default="cuda:0")
    pe.add_argument("--pca-file", type=str, default=None)
    pe.add_argument("--stripe-box", type=float, nargs=4, default=[0.0, 4.0, 0.5, 1.5])
    pe.add_argument("--predict-best-on-full", action=argparse.BooleanOptionalAction,
                    default=True)
    add_run_root_arg(pe)

    # predict (full set, no FEM filter)
    pp = sub.add_parser("predict", help="Run trained head on ALL candidates.")
    pp.add_argument("--landscape-dirs", nargs="+", default=None)
    pp.add_argument("--consistency-file", type=str, default=None)
    pp.add_argument("--ckpt", type=str, default=None)
    pp.add_argument("--output", type=str, default=None)
    pp.add_argument("--device", type=str, default="cuda:0")
    pp.add_argument("--batch-size", type=int, default=512)
    add_run_root_arg(pp)

    args = parser.parse_args()

    # Resolve --run-root defaults per subcommand.
    if args.cmd == "build":
        if args.landscape_dirs is None:
            if args.run_root is None:
                parser.error("either --landscape-dirs or --run-root is required")
            args.landscape_dirs = [resolve(args.run_root, "walk"),
                                   resolve(args.run_root, "novelty")]
        if args.consistency_file is None:
            if args.run_root is None:
                parser.error("either --consistency-file or --run-root is required")
            args.consistency_file = resolve(args.run_root, "consistency_jsonl")
        if args.output_dir is None:
            if args.run_root is None:
                parser.error("either --output-dir or --run-root is required")
            args.output_dir = resolve(args.run_root, "ood_dir")
        _run_build(args)

    elif args.cmd == "train":
        if args.dataset is None:
            if args.run_root is None:
                parser.error("either --dataset or --run-root is required")
            args.dataset = resolve(args.run_root, "ood_dataset")
        if args.output_dir is None:
            if args.run_root is None:
                parser.error("either --output-dir or --run-root is required")
            args.output_dir = resolve(args.run_root, "ood_dir")
        _run_train(args)

    elif args.cmd == "eval":
        if args.dataset is None:
            if args.run_root is None:
                parser.error("either --dataset or --run-root is required")
            args.dataset = resolve(args.run_root, "ood_dataset")
        if args.output_dir is None:
            if args.run_root is None:
                parser.error("either --output-dir or --run-root is required")
            args.output_dir = resolve(args.run_root, "ood_dir")
        if args.pca_file is None and args.run_root is not None:
            args.pca_file = os.path.join(
                resolve(args.run_root, "figures_dir"), "pca_combined.json")
        if args.pca_file is None:
            args.pca_file = ""
        _run_eval(args)

    elif args.cmd == "predict":
        if args.landscape_dirs is None:
            if args.run_root is None:
                parser.error("either --landscape-dirs or --run-root is required")
            args.landscape_dirs = [resolve(args.run_root, "walk"),
                                   resolve(args.run_root, "novelty")]
        if args.consistency_file is None:
            if args.run_root is None:
                parser.error("either --consistency-file or --run-root is required")
            args.consistency_file = resolve(args.run_root, "consistency_jsonl")
        if args.ckpt is None:
            if args.run_root is None:
                parser.error("either --ckpt or --run-root is required")
            args.ckpt = resolve(args.run_root, "ood_ckpt_full")
        if args.output is None:
            if args.run_root is None:
                parser.error("either --output or --run-root is required")
            args.output = resolve(args.run_root, "ood_predictions_full")
        _run_predict(args)


if __name__ == "__main__":
    main()
