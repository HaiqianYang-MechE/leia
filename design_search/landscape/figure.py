#!/usr/bin/env python3
"""Stage 15 of the Figure 5 pipeline (figure5.sh): the 4-panel OOD-landscape figure.

One subcommand:

  pca — fit PCA on the training-plate latents (mean-pooled prevq), project
        all candidates, save two scatter panels:
          Panel A: PC1×PC2, colored by tokenizer recon_error (log).
          Panel B: PC1×PC2, colored by LEIA-vs-FEM model error (log).

Usage:
    python -m design_search.landscape.figure pca \\
        --landscape-dir design_search/outputs/<landscape_run> \\
        --metric-b total_work_err
"""

import argparse
import json
import os

import numpy as np


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


METRIC_B_LABELS = {
    "total_work_err": "|W_LEIA − W_FEM| / W_FEM",
    "disp_l2_err_mean": "‖u_LEIA − u_FEM‖ / ‖u_FEM‖",
    "vm_corr_mean": r"$\rho$",
}


def _load_data_and_metrics(landscape: str, metric_b_field: str):
    """Load LEIA latents + recon error + chosen LEIA-vs-FEM error metric.

    Returns
    -------
    leia : list[dict]                  # entries with status='ok' and latent
    latents : ndarray (N, 1024)
    seeds_mask : ndarray (N,) bool     # depth == 0 (training plates)
    metric_a : ndarray (N,)            # recon_error
    metric_b : ndarray (N,)            # chosen LEIA-vs-FEM error (nan if no FEM)
    fem_paintable : ndarray (N,) bool  # finite metric_b
    """
    leia_jsonl = os.path.join(landscape, "leia_scores.jsonl")
    fem_jsonl = os.path.join(landscape, "fem_scores.jsonl")

    leia = [e for e in _read_jsonl(leia_jsonl) if e.get("status") == "ok"]
    print(f"LEIA ok entries: {len(leia)}")

    latents = np.zeros((len(leia), 1024), dtype=np.float32)
    valid_mask = np.zeros(len(leia), dtype=bool)
    for i, e in enumerate(leia):
        npz_path = os.path.join(landscape, e["leia_outputs_path"])
        try:
            with np.load(npz_path) as z:
                latents[i] = z["latent_meanpool"]
            valid_mask[i] = True
        except Exception as ex:
            print(f"  Skip {e['id']}: {ex}")
    leia = [e for e, v in zip(leia, valid_mask) if v]
    latents = latents[valid_mask]
    print(f"Loaded latents: {latents.shape}")

    depths = np.array([e["depth"] for e in leia])
    seeds_mask = depths == 0
    print(f"Training-plate seeds (depth=0): {seeds_mask.sum()}")

    fem_by_id = {
        e["id"]: e for e in _read_jsonl(fem_jsonl)
        if e.get("fem_status") == "ok"
    }

    metric_a = np.array([e["recon_error"] for e in leia], dtype=np.float64)
    metric_b = np.full(len(leia), np.nan, dtype=np.float64)
    fem_paintable = np.zeros(len(leia), dtype=bool)
    for i, e in enumerate(leia):
        f = fem_by_id.get(e["id"])
        if f is None:
            continue
        v = f.get(metric_b_field)
        if v is None or not np.isfinite(v):
            continue
        metric_b[i] = float(v)
        fem_paintable[i] = True
    print(f"Panel B paintable ({metric_b_field}): "
          f"{fem_paintable.sum()}/{len(leia)}")

    return leia, latents, seeds_mask, metric_a, metric_b, fem_paintable


def _load_data_and_metrics_multi(landscape_dirs: list[str], metric_b_field: str):
    """Load + concatenate data across multiple landscape dirs.

    Only the first dir's depth=0 entries are marked as `seeds_mask` (training
    plates). All other entries (including any depth=0 from extra dirs) are
    treated as mutated/exploratory candidates. Each `leia` entry gets an
    extra `landscape_dir_idx` + `landscape_dir` field for downstream routing.
    """
    leia_all: list[dict] = []
    latents_all = []
    seeds_mask_parts = []
    metric_a_parts = []
    metric_b_parts = []
    fem_paintable_parts = []
    for di, ldir in enumerate(landscape_dirs):
        print(f"\n--- Loading landscape dir [{di}]: {ldir} ---")
        leia, latents, sm, ma, mb, fp = _load_data_and_metrics(
            ldir, metric_b_field)
        if di == 0:
            seeds_mask_parts.append(sm)
        else:
            seeds_mask_parts.append(np.zeros_like(sm))
        for e in leia:
            e["landscape_dir_idx"] = di
            e["landscape_dir"] = ldir
        leia_all.extend(leia)
        latents_all.append(latents)
        metric_a_parts.append(ma)
        metric_b_parts.append(mb)
        fem_paintable_parts.append(fp)

    latents = np.concatenate(latents_all, axis=0)
    seeds_mask = np.concatenate(seeds_mask_parts)
    metric_a = np.concatenate(metric_a_parts)
    metric_b = np.concatenate(metric_b_parts)
    fem_paintable = np.concatenate(fem_paintable_parts)
    print(f"\nCombined: {len(leia_all)} candidates, "
          f"{seeds_mask.sum()} training-plate seeds (from first dir only)")
    return leia_all, latents, seeds_mask, metric_a, metric_b, fem_paintable


def _vlims(values, lo=2, hi=98):
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 1e-6, 1.0
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def _draw_recon_panel(ax, x_arr, y_arr, seeds_mask, metric_a,
                       xlabel, ylabel, panel_label,
                       heldout_xy=None, spearman=None,
                       a_label: str = "Tokenizer reconstruction error",
                       show_legend: bool = True,
                       cloud_scale: float = 1.0,
                       prominent_scale: float = 1.0,
                       short_title: bool = False):
    """Draw Panel A (reconstruction error landscape) on a given axis."""
    from matplotlib.colors import LogNorm
    import matplotlib.pyplot as plt

    s_cloud = 12 * cloud_scale
    s_seed = 55 * prominent_scale
    s_held = 85 * prominent_scale
    a_clipped = np.clip(metric_a, 1e-9, None)
    a_lo, a_hi = _vlims(a_clipped)
    sc = ax.scatter(
        x_arr[~seeds_mask], y_arr[~seeds_mask],
        c=a_clipped[~seeds_mask], cmap="viridis",
        norm=LogNorm(vmin=max(a_lo, 1e-9), vmax=a_hi),
        s=s_cloud, alpha=0.55, edgecolors="none", zorder=2,
    )
    ax.scatter(
        x_arr[seeds_mask], y_arr[seeds_mask],
        c="black", s=s_seed, marker="D",
        edgecolors="white", linewidths=0.8, zorder=4,
        label=f"Training plates ({int(seeds_mask.sum())})",
    )
    if heldout_xy is not None and len(heldout_xy) > 0:
        ax.scatter(
            heldout_xy[:, 0], heldout_xy[:, 1],
            c="red", s=s_held, marker="^",
            edgecolors="black", linewidths=0.8, zorder=5,
            label=f"Held-out plates ({len(heldout_xy)})",
        )
    cb = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(a_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title_text = ("Recon error" if short_title
                  else "Reconstruction error landscape")
    ax.set_title(f"({panel_label}) {title_text}")
    if spearman is not None:
        ax.text(
            0.97, 0.03, f"corr vs FEM = {spearman:+.3f}",
            transform=ax.transAxes, va="bottom", ha="right",
            fontsize=12, fontweight="bold",
        )
    if show_legend:
        ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_pred_panel(ax, x_arr, y_arr, seeds_mask, metric_d,
                     d_paintable, xlabel, ylabel, panel_label,
                     d_label, heldout_xy=None, heldout_metric=None,
                     spearman=None, show_legend: bool = True,
                     cloud_scale: float = 1.0,
                     prominent_scale: float = 1.0,
                     short_title: bool = False):
    """Draw Panel D (predicted vm_corr from learned OOD head).

    Uses RdYlGn so colors match Panel B's semantics: green = high predicted
    vm_corr = healthy, red = low predicted = warning. Robust quantile vlims.
    """
    import matplotlib.pyplot as plt

    s_cloud = 12 * cloud_scale
    s_seed = 55 * prominent_scale
    s_held_color = 140 * prominent_scale
    s_held_red = 85 * prominent_scale
    s_grey = 10 * cloud_scale
    paint = d_paintable & ~seeds_mask
    no_d_nontrain = ~d_paintable & ~seeds_mask
    if no_d_nontrain.any():
        ax.scatter(
            x_arr[no_d_nontrain], y_arr[no_d_nontrain],
            c="lightgrey", s=s_grey, alpha=0.4, edgecolors="none",
            zorder=1,
        )
    # Anchor predicted-vm-corr colorbar to [0, 1] to match Panel A's scale.
    scatter_vmin = 0.0
    scatter_vmax = 1.0
    sc = ax.scatter(
        x_arr[paint], y_arr[paint],
        c=metric_d[paint], cmap="RdBu",
        vmin=scatter_vmin, vmax=scatter_vmax,
        s=s_cloud, alpha=0.55, edgecolors="none", zorder=2,
    )
    ax.scatter(
        x_arr[seeds_mask], y_arr[seeds_mask],
        c="black", s=s_seed, marker="D",
        edgecolors="white", linewidths=0.8, zorder=4,
        label=f"Training plates ({int(seeds_mask.sum())})",
    )
    if heldout_xy is not None and len(heldout_xy) > 0:
        if heldout_metric is not None and len(heldout_metric) == len(heldout_xy):
            # Color held-out triangles by their predicted vm_corr — share
            # the cloud's vmin/vmax so identical predictions look identical.
            ax.scatter(
                heldout_xy[:, 0], heldout_xy[:, 1],
                c=heldout_metric, cmap="RdBu",
                vmin=scatter_vmin, vmax=scatter_vmax,
                s=s_held_color, marker="^",
                edgecolors="black", linewidths=1.2, zorder=5,
                label=f"Held-out plates ({len(heldout_xy)})",
            )
        else:
            ax.scatter(
                heldout_xy[:, 0], heldout_xy[:, 1],
                c="red", s=s_held_red, marker="^",
                edgecolors="black", linewidths=0.8, zorder=5,
                label=f"Held-out plates ({len(heldout_xy)})",
            )
    cb = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(d_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title_text = ("Predicted" if short_title
                  else "Predicted model error landscape")
    ax.set_title(f"({panel_label}) {title_text}")
    if spearman is not None:
        ax.text(
            0.97, 0.03, f"corr vs FEM = {spearman:+.3f}",
            transform=ax.transAxes, va="bottom", ha="right",
            fontsize=12, fontweight="bold",
        )
    if show_legend:
        ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_consistency_panel(ax, x_arr, y_arr, seeds_mask, metric_c,
                             c_paintable, xlabel, ylabel, panel_label,
                             c_label, heldout_xy=None, spearman=None,
                             show_legend: bool = True,
                             cloud_scale: float = 1.0,
                             prominent_scale: float = 1.0,
                             short_title: bool = False):
    """Draw Panel C (dynamics encoder-cycle consistency)."""
    import matplotlib.pyplot as plt

    s_cloud = 12 * cloud_scale
    s_seed = 55 * prominent_scale
    s_held = 85 * prominent_scale
    s_grey = 10 * cloud_scale
    paint = c_paintable & ~seeds_mask
    no_c_nontrain = ~c_paintable & ~seeds_mask
    if no_c_nontrain.any():
        ax.scatter(
            x_arr[no_c_nontrain], y_arr[no_c_nontrain],
            c="lightgrey", s=s_grey, alpha=0.4, edgecolors="none",
            zorder=1, label=f"No consistency ({int(no_c_nontrain.sum())})",
        )

    c_lo, c_hi = _vlims(metric_c[c_paintable])
    sc = ax.scatter(
        x_arr[paint], y_arr[paint],
        c=metric_c[paint], cmap="magma",
        vmin=c_lo, vmax=c_hi,
        s=s_cloud, alpha=0.55, edgecolors="none", zorder=2,
    )
    ax.scatter(
        x_arr[seeds_mask], y_arr[seeds_mask],
        c="black", s=s_seed, marker="D",
        edgecolors="white", linewidths=0.8, zorder=4,
        label=f"Training plates ({int(seeds_mask.sum())})",
    )
    if heldout_xy is not None and len(heldout_xy) > 0:
        ax.scatter(
            heldout_xy[:, 0], heldout_xy[:, 1],
            c="red", s=s_held, marker="^",
            edgecolors="black", linewidths=0.8, zorder=5,
            label=f"Held-out plates ({len(heldout_xy)})",
        )
    cb = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(c_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title_text = ("Consistency" if short_title
                  else "Dynamics consistency landscape")
    ax.set_title(f"({panel_label}) {title_text}")
    if spearman is not None:
        ax.text(
            0.97, 0.03, f"corr vs FEM = {spearman:+.3f}",
            transform=ax.transAxes, va="bottom", ha="right",
            fontsize=12, fontweight="bold",
        )
    if show_legend:
        ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_model_panel(ax, x_arr, y_arr, seeds_mask, metric_b, fem_paintable,
                       metric_b_field, xlabel, ylabel, panel_label,
                       clean: bool = False, heldout_xy=None,
                       show_legend: bool = True,
                       cloud_scale: float = 1.0,
                       prominent_scale: float = 1.0,
                       short_title: bool = False):
    """Draw Panel B (LEIA-vs-FEM model error) on a given axis.

    If `clean=True`, strip the "No FEM" grey dots and "Training plate, FEM
    failed" red X markers — only colored paintable scatter + training-plate
    diamonds remain. For paper-figure cleanliness.
    """
    from matplotlib.colors import LogNorm
    import matplotlib.pyplot as plt

    s_cloud = 12 * cloud_scale
    s_seed = 55 * prominent_scale
    s_held = 85 * prominent_scale
    s_grey = 10 * cloud_scale
    paint = fem_paintable & ~seeds_mask
    no_fem_nontrain = ~fem_paintable & ~seeds_mask
    if not clean and no_fem_nontrain.any():
        ax.scatter(
            x_arr[no_fem_nontrain], y_arr[no_fem_nontrain],
            c="lightgrey", s=s_grey, alpha=0.4, edgecolors="none",
            zorder=1,
            label=f"FEM failed ({int(no_fem_nontrain.sum())})",
        )

    use_log = metric_b_field != "vm_corr_mean"
    if use_log:
        b_clipped = np.clip(metric_b, 1e-6, None)
        b_lo, b_hi = _vlims(b_clipped[fem_paintable])
        norm = LogNorm(vmin=max(b_lo, 1e-6), vmax=b_hi)
        scatter_vmin, scatter_vmax = None, None
    else:
        # vm_corr_mean: anchor the colorbar to the full [0, 1] range so Panel A
        # and Panel D (predicted) share an identical color scale. A few
        # candidates may have vm_corr < 0; those clip to red.
        b_clipped = metric_b
        norm = None
        scatter_vmin = 0.0
        scatter_vmax = 1.0

    sc = ax.scatter(
        x_arr[paint], y_arr[paint],
        c=b_clipped[paint], cmap="magma" if use_log else "RdBu",
        norm=norm, s=s_cloud, alpha=0.55, edgecolors="none", zorder=2,
        vmin=scatter_vmin, vmax=scatter_vmax,
    )
    ax.scatter(
        x_arr[seeds_mask], y_arr[seeds_mask],
        c="black", s=s_seed, marker="D",
        edgecolors="white", linewidths=0.8, zorder=4,
        label=f"Training plates ({int(seeds_mask.sum())})",
    )
    if heldout_xy is not None and len(heldout_xy) > 0:
        ax.scatter(
            heldout_xy[:, 0], heldout_xy[:, 1],
            c="red", s=s_held, marker="^",
            edgecolors="black", linewidths=0.8, zorder=5,
            label=f"Held-out plates ({len(heldout_xy)})",
        )
    # (Training plates whose FEM failed are still drawn as black diamonds via
    # the seeds_mask scatter above — no separate red-X annotation.)
    cb = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(METRIC_B_LABELS[metric_b_field])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title_text = ("Model error" if short_title
                  else "Model error landscape")
    ax.set_title(f"({panel_label}) {title_text}")
    if show_legend:
        ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_trajectory(ax, x_idx, y_idx, traj_dir, leia_all, proj,
                     subset_seed_ids=None):
    """Overlay per-seed beam-to-beam arrows from a novelty-search dir.

    For each seed `s`, draws arrows from each `in_beam_at_iter == n - 1`
    candidate to its child `in_beam_at_iter == n` candidate, colored by
    iteration (viridis). Subset seeds get full alpha; others alpha=0.25.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    cands_path = os.path.join(traj_dir, "candidates.json")
    meta_path = os.path.join(traj_dir, "novelty_meta.json")
    if not (os.path.exists(cands_path) and os.path.exists(meta_path)):
        print(f"  No trajectory data in {traj_dir} (candidates.json / "
              f"novelty_meta.json missing); skipping overlay.")
        return

    with open(cands_path) as f:
        cdata = json.load(f)
    with open(meta_path) as f:
        meta = json.load(f)
    D = int(meta.get("D", 8))

    # Map id -> projection (from the *full* landscape proj, since the novelty
    # candidates' latents may have been transformed into a different basis than
    # the file's stored proj). Also map id -> candidate dict.
    id_to_idx = {e["id"]: i for i, e in enumerate(leia_all)}
    cand_by_id = {c["id"]: c for c in cdata["candidates"]}

    cmap = cm.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=max(D, 1))

    # Group candidates by seed
    by_seed: dict[str, list[dict]] = {}
    for c in cdata["candidates"]:
        by_seed.setdefault(c["seed_plate"], []).append(c)

    n_arrows = 0
    for seed_id, seed_cands in by_seed.items():
        is_subset = (subset_seed_ids is None
                     or seed_id in subset_seed_ids)
        alpha = 0.85 if is_subset else 0.18
        lw = 1.4 if is_subset else 0.7

        # Walk all in_beam_at_iter > 0 candidates and draw to parent
        for c in seed_cands:
            iab = c.get("in_beam_at_iter")
            pid = c.get("parent_id")
            if iab is None or iab == 0 or pid is None:
                continue
            if c["id"] not in id_to_idx or pid not in id_to_idx:
                continue
            child_proj = proj[id_to_idx[c["id"]]]
            parent_proj = proj[id_to_idx[pid]]
            color = cmap(norm(iab))
            ax.annotate(
                "",
                xy=(child_proj[x_idx], child_proj[y_idx]),
                xytext=(parent_proj[x_idx], parent_proj[y_idx]),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, alpha=alpha,
                                shrinkA=2, shrinkB=2),
                zorder=3,
            )
            n_arrows += 1

    print(f"  Trajectory overlay: drew {n_arrows} arrows "
          f"(subset highlighted: {len(subset_seed_ids) if subset_seed_ids else 0})")

    # Mark final-iter beam members with stars
    final_beam = [c for c in cdata["candidates"]
                  if c.get("in_beam_at_iter") == D]
    if final_beam:
        fb_xs, fb_ys = [], []
        for c in final_beam:
            if c["id"] in id_to_idx:
                p = proj[id_to_idx[c["id"]]]
                fb_xs.append(p[x_idx])
                fb_ys.append(p[y_idx])
        if fb_xs:
            ax.scatter(fb_xs, fb_ys, marker="*", s=80,
                       c="orange", edgecolors="black", linewidths=0.6,
                       zorder=6,
                       label=f"Final iter-{D} beam ({len(fb_xs)})")


def _plot_panels_grid(
    proj: np.ndarray,
    seeds_mask: np.ndarray,
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    fem_paintable: np.ndarray,
    metric_b_field: str,
    pc_pairs: list,            # list of (x_idx, y_idx, x_label, y_label)
    suptitle: str,
    out_pdf: str,
    out_png: str,
    trajectory_dir: str | None = None,
    trajectory_subset: list[str] | None = None,
    leia_all: list | None = None,
    clean_panel_b: bool = False,
    heldout_proj: np.ndarray | None = None,
    show_suptitle: bool = True,
    metric_c: np.ndarray | None = None,
    c_paintable: np.ndarray | None = None,
    metric_c_label: str = "Dynamics consistency",
    metric_d: np.ndarray | None = None,
    d_paintable: np.ndarray | None = None,
    metric_d_label: str = r"$\hat{\rho}$",
    heldout_metric_d: np.ndarray | None = None,
    spearman_a: float | None = None,
    spearman_c: float | None = None,
    spearman_d: float | None = None,
    grid_2x2: bool = False,
    panel_order: str = "truth-first",
    minimal: bool = False,
    metric_a_label: str = "Tokenizer reconstruction error",
):
    """Render an N-row × {2, 3, or 4}-column grid.

    Columns are (recon, model[, dynamics-consistency]). The 3rd column appears
    only when `metric_c` is provided (e.g. encoder-cycle consistency from the
    AR rollout).

    If `clean_panel_b=True`, the model-error column is stripped of "No FEM",
    "FEM failed", "Final iter-D beam" markers and trajectory arrows — only
    the colored paintable scatter and training-plate diamonds remain.

    `heldout_proj` is an optional (M, n_components) array of held-out plate
    projections — if provided, drawn as red triangles on every panel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if minimal:
        # Same overall figure size as the non-minimal default (markers and
        # canvas proportions match the reference figure). Fonts bumped a bit
        # so they read better when LaTeX shrinks the figure to \textwidth.
        plt.rcParams.update({
            "font.size": 17,
            "axes.labelsize": 18,
            "axes.titlesize": 19,
            "legend.fontsize": 14,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
        })
    else:
        plt.rcParams.update({
            "font.size": 14,
            "axes.labelsize": 15,
            "axes.titlesize": 16,
            "legend.fontsize": 12,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        })

    n_rows = len(pc_pairs)
    n_panels_per_row = (
        2 + (1 if metric_c is not None else 0) + (1 if metric_d is not None else 0)
    )
    # 2×2 path: only meaningful when there's a single PC pair AND four panels.
    use_2x2 = bool(grid_2x2) and n_rows == 1 and n_panels_per_row == 4
    if use_2x2:
        fig_size = (13, 11)
        fig, ax_grid = plt.subplots(2, 2, figsize=fig_size)
        # ax_grid[0,0]=A (recon), [0,1]=B (true vm), [1,0]=C (consistency), [1,1]=D (pred)
        slot_for = {"A": ax_grid[0, 0], "B": ax_grid[0, 1],
                    "C": ax_grid[1, 0], "D": ax_grid[1, 1]}
        labels_seq = ["A", "B", "C", "D"]
    else:
        fig_size = (7 * n_panels_per_row, 5.6 * n_rows)
        # Share y-axis across panels in minimal 1×N: all panels use the same
        # PCA basis so the ylabel + y-ticks only need to appear on column 0.
        fig, axes = plt.subplots(n_rows, n_panels_per_row, figsize=fig_size,
                                 sharey=bool(minimal),
                                 sharex=bool(minimal))
        if n_rows == 1:
            axes = axes.reshape(1, n_panels_per_row)
        # Panel labels: row-major across columns.
        _LABEL_BANK = "ABCDEFGHIJKL"
        label_pairs = []
        for r in range(n_rows):
            label_pairs.append(tuple(_LABEL_BANK[r * n_panels_per_row + c]
                                     for c in range(n_panels_per_row)))

    for row_i, (x_i, y_i, xl, yl) in enumerate(pc_pairs):
        x_arr = proj[:, x_i]
        y_arr = proj[:, y_i]
        ho_xy = heldout_proj[:, [x_i, y_i]] if heldout_proj is not None else None
        if use_2x2:
            ax_a = slot_for["A"]; ax_b = slot_for["B"]
            ax_c = slot_for["C"]; ax_d = slot_for["D"]
            la, lb, lc, ld = labels_seq
        else:
            labels = label_pairs[row_i]
            ax_a = axes[row_i, 0]; ax_b = axes[row_i, 1]
            la, lb = labels[0], labels[1]
            ax_c = axes[row_i, 2] if metric_c is not None else None
            lc = labels[2] if metric_c is not None else None
            d_col = 2 + (1 if metric_c is not None else 0)
            ax_d = (axes[row_i, d_col] if metric_d is not None else None)
            ld = labels[d_col] if metric_d is not None else None

        # Panel order:
        #   "recon-first" (legacy):  A=recon, B=model error, C=consistency, D=predicted
        #   "truth-first" (default): A=model error, B=recon, C=consistency, D=predicted
        # 2×2 path always groups (predictors top / truth+pred bottom) — see use_2x2 setup.
        show_legend = not minimal
        # In minimal mode, suppress the corr-vs-FEM insets too.
        sa = None if minimal else spearman_a
        sc_ = None if minimal else spearman_c
        sd = None if minimal else spearman_d
        # Markers stay at their default sizes (matching the reference figure).
        # The minimal flag only affects fonts, legends, insets, and labels.
        cloud_scale = 1.0
        prominent_scale = 1.0
        short_title = bool(minimal)
        kw_extra = dict(cloud_scale=cloud_scale,
                        prominent_scale=prominent_scale,
                        short_title=short_title)
        if panel_order == "truth-first" and not use_2x2:
            _draw_model_panel(ax_a, x_arr, y_arr, seeds_mask, metric_b,
                              fem_paintable, metric_b_field, xl, yl, panel_label=la,
                              clean=clean_panel_b, heldout_xy=ho_xy,
                              show_legend=show_legend, **kw_extra)
            _draw_recon_panel(ax_b, x_arr, y_arr, seeds_mask, metric_a,
                              xl, yl, panel_label=lb, heldout_xy=ho_xy,
                              spearman=sa, a_label=metric_a_label,
                              show_legend=show_legend, **kw_extra)
        else:
            _draw_recon_panel(ax_a, x_arr, y_arr, seeds_mask, metric_a,
                              xl, yl, panel_label=la, heldout_xy=ho_xy,
                              spearman=sa, a_label=metric_a_label,
                              show_legend=show_legend, **kw_extra)
            _draw_model_panel(ax_b, x_arr, y_arr, seeds_mask, metric_b,
                              fem_paintable, metric_b_field, xl, yl, panel_label=lb,
                              clean=clean_panel_b, heldout_xy=ho_xy,
                              show_legend=show_legend, **kw_extra)
        if metric_c is not None and ax_c is not None:
            _draw_consistency_panel(ax_c, x_arr, y_arr, seeds_mask,
                                    metric_c, c_paintable, xl, yl,
                                    panel_label=lc,
                                    c_label=metric_c_label,
                                    heldout_xy=ho_xy,
                                    spearman=sc_,
                                    show_legend=show_legend, **kw_extra)
        if metric_d is not None and ax_d is not None:
            _draw_pred_panel(ax_d, x_arr, y_arr, seeds_mask,
                             metric_d, d_paintable, xl, yl,
                             panel_label=ld,
                             d_label=metric_d_label,
                             heldout_xy=ho_xy,
                             heldout_metric=heldout_metric_d,
                             spearman=sd,
                             show_legend=show_legend, **kw_extra)
        # Reduce x/y tick density for paper-print-size figures.
        if minimal:
            from matplotlib.ticker import MaxNLocator
            for ax_ in [ax_a, ax_b, ax_c, ax_d]:
                if ax_ is not None:
                    ax_.xaxis.set_major_locator(MaxNLocator(nbins=4))
                    ax_.yaxis.set_major_locator(MaxNLocator(nbins=4))
            # 1×N: drop redundant ylabels on non-leftmost panels (sharey=True
            # already aligns them; the labels just waste horizontal space).
            if not use_2x2:
                for ax_ in [ax_b, ax_c, ax_d]:
                    if ax_ is not None:
                        ax_.set_ylabel("")
        # Trajectory overlay always on the *model-error* panel, regardless of
        # which slot it's in (depends on panel_order).
        model_ax = (ax_a if (panel_order == "truth-first" and not use_2x2)
                    else ax_b)
        if (not clean_panel_b
                and trajectory_dir is not None and leia_all is not None):
            _draw_trajectory(model_ax, x_i, y_i, trajectory_dir,
                             leia_all, proj,
                             subset_seed_ids=trajectory_subset)
            # Re-render legend so the "Final iter-D beam" entry appears
            model_ax.legend(loc="best", framealpha=0.9, fontsize=11)

    if show_suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.0)
    fig.tight_layout()

    fig.savefig(out_pdf, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")



def cmd_pca(args):
    """Fit PCA on training-plate latents, project all candidates, save panels.

    Features = tokenizer mean-pooled prevq (1024-d). PCA fit on the 55
    training-plate seeds (depth=0 from --landscape-dir).

    --extra-landscape-dirs (multi-value): merge candidates from additional
        landscape dirs into the same figure. The first dir defines training
        seeds; extras are treated as mutated/exploratory candidates.

    --show-trajectory: overlay per-seed beam-to-beam arrows onto Panel B.
    --trajectory-seeds-subset: comma-separated seed_ids to highlight at full
        alpha (others fade to 0.18).
    """
    from sklearn.decomposition import PCA
    from design_search.landscape.layout import resolve

    if getattr(args, "run_root", None):
        if args.landscape_dir is None:
            args.landscape_dir = resolve(args.run_root, "walk")
        if args.extra_landscape_dirs is None:
            args.extra_landscape_dirs = [resolve(args.run_root, "novelty")]
        if args.consistency_file is None:
            args.consistency_file = resolve(args.run_root, "consistency_jsonl")
        if args.ood_pred_file is None:
            args.ood_pred_file = resolve(args.run_root, "ood_predictions_full")
        if args.heldout_latents_file is None:
            args.heldout_latents_file = resolve(args.run_root, "heldout_latents")

    primary = args.landscape_dir
    extras = args.extra_landscape_dirs or []

    if extras:
        leia, latents, seeds_mask, metric_a, metric_b, fem_paintable = (
            _load_data_and_metrics_multi([primary, *extras], args.metric_b)
        )
    else:
        leia, latents, seeds_mask, metric_a, metric_b, fem_paintable = (
            _load_data_and_metrics(primary, args.metric_b)
        )

    X = latents
    space_label = "tokenizer-latent (mean-pooled prevq, 1024-d)"
    suffix = ""

    pca = PCA(n_components=3)
    pca.fit(X[seeds_mask])
    proj = pca.transform(X)
    var_ratios = pca.explained_variance_ratio_
    print(f"Explained variance: PC1={var_ratios[0]:.2%} "
          f"PC2={var_ratios[1]:.2%} PC3={var_ratios[2]:.2%} "
          f"total={sum(var_ratios):.2%}")

    # Dynamics consistency metric (Panel C). Loaded as JSONL keyed by id.
    metric_c = None
    c_paintable = None
    metric_c_label = "Dynamics consistency"
    if args.consistency_file:
        c_by_id = {}
        with open(args.consistency_file) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("status") != "ok":
                    continue
                cs = rec.get("c_per_frame")
                if not cs:
                    continue
                if args.consistency_metric == "mean":
                    c_by_id[rec["id"]] = float(np.mean(cs))
                elif args.consistency_metric == "t3":
                    c_by_id[rec["id"]] = float(cs[2]) if len(cs) >= 3 else float("nan")
                    metric_c_label = "Dynamics consistency"
                elif args.consistency_metric == "t10":
                    c_by_id[rec["id"]] = float(cs[-1])
                    metric_c_label = "Dynamics consistency"
                elif args.consistency_metric == "slope":
                    if len(cs) >= 3:
                        c_by_id[rec["id"]] = float(cs[-1] - cs[1])
                    else:
                        c_by_id[rec["id"]] = float("nan")
                    metric_c_label = "Dynamics consistency"
        metric_c = np.full(len(leia), np.nan, dtype=np.float64)
        for i, e in enumerate(leia):
            v = c_by_id.get(e["id"])
            if v is not None and np.isfinite(v):
                metric_c[i] = v
        c_paintable = np.isfinite(metric_c)
        print(f"Panel C paintable ({args.consistency_metric}): "
              f"{int(c_paintable.sum())}/{len(leia)}")

    # OOD-head predicted vm_corr (Panel D). predictions.jsonl from `ood.pipeline predict`.
    metric_d = None
    d_paintable = None
    if args.ood_pred_file:
        pred_by_id = {}
        with open(args.ood_pred_file) as f:
            for line in f:
                rec = json.loads(line)
                v = rec.get("predicted_vm_corr")
                if v is not None and np.isfinite(v):
                    pred_by_id[rec["id"]] = float(v)
        metric_d = np.full(len(leia), np.nan, dtype=np.float64)
        for i, e in enumerate(leia):
            v = pred_by_id.get(e["id"])
            if v is not None and np.isfinite(v):
                metric_d[i] = v
        d_paintable = np.isfinite(metric_d)
        print(f"Panel D paintable (predicted vm_corr): "
              f"{int(d_paintable.sum())}/{len(leia)}")

    # Held-out plates — project their pre-encoded latents onto the same basis.
    heldout_proj = None
    heldout_ids = []
    if args.heldout_latents_file:
        with open(args.heldout_latents_file) as f:
            ho = json.load(f)
        heldout_ids = [p["id"] for p in ho["plates"]]
        ho_lat = np.array([p["latent"] for p in ho["plates"]], dtype=np.float64)
        heldout_proj = pca.transform(ho_lat)
        print(f"Loaded {len(heldout_ids)} held-out plate latents; "
              f"projected onto PCA basis.")

    # Output dir: when --run-root is set route to <root>/figures/; else combined
    # mode uses outputs/landscape_v2_combined/, single-dir mode uses primary.
    if getattr(args, "run_root", None):
        out_dir = resolve(args.run_root, "figures_dir")
        os.makedirs(out_dir, exist_ok=True)
        if extras:
            suffix = f"_combined{suffix}"
    elif extras:
        out_dir = os.path.join(
            os.path.dirname(primary.rstrip("/")), "landscape_v2_combined")
        os.makedirs(out_dir, exist_ok=True)
        suffix = f"_combined{suffix}"
    else:
        out_dir = primary

    pca_dump = {
        "method": "pca",
        "feature_space": "latent",
        "fit_on": "training",
        "primary_dir": primary,
        "extra_dirs": extras,
        "n_train": int(seeds_mask.sum()),
        "n_total": int(len(leia)),
        "explained_variance_ratio": var_ratios.tolist(),
        "components": pca.components_.tolist(),
        "mean": pca.mean_.tolist(),
        "ids": [e["id"] for e in leia],
        "depths": [e["depth"] for e in leia],
        "landscape_dir_idx": [e.get("landscape_dir_idx", 0) for e in leia],
        "is_training_seed": seeds_mask.tolist(),
        "proj": proj.tolist(),
        "metric_a": "recon_error",
        "metric_b": args.metric_b,
    }
    pca_path = os.path.join(out_dir, f"pca{suffix}.json")
    with open(pca_path, "w") as f:
        json.dump(pca_dump, f)
    print(f"Saved {pca_path}")

    suptitle = (f"Design landscape, {space_label} "
                f"({len(leia)} candidates from {int(seeds_mask.sum())} "
                f"training plates; PCA fit on training only)")
    pc_label = lambda i: f"PC{i + 1} ({var_ratios[i]:.0%} var.)"

    # Trajectory subset parsing
    traj_subset = None
    if args.trajectory_seeds_subset:
        traj_subset = [s.strip() for s in args.trajectory_seeds_subset.split(",")
                       if s.strip()]
    traj_dir = args.show_trajectory

    # In minimal/paper mode, override the consistency colorbar to the
    # paper-precise per-frame-mean expression.
    if args.minimal:
        metric_c_label = r"$\langle \|z - \hat{z}\| / \|z\| \rangle$"

    # Compute Spearman annotations for A / C / D against vm_corr_mean ground
    # truth on the FEM-paintable mask (independent of train/val split).
    from scipy.stats import spearmanr as _sr
    def _safe_spearman(x, y, mask):
        m = np.asarray(mask, dtype=bool)
        if m.sum() < 3:
            return None
        try:
            v = float(_sr(np.asarray(x)[m], np.asarray(y)[m])[0])
            return v if np.isfinite(v) else None
        except Exception:
            return None

    spearman_a = _safe_spearman(metric_a, metric_b, fem_paintable)
    spearman_c = (None if metric_c is None
                  else _safe_spearman(metric_c, metric_b,
                                      fem_paintable & c_paintable))
    spearman_d = (None if metric_d is None
                  else _safe_spearman(metric_d, metric_b,
                                      fem_paintable & d_paintable))
    if spearman_a is not None:
        print(f"Recon-error Spearman vs FEM: {spearman_a:+.4f}")
    if spearman_c is not None:
        print(f"Consistency Spearman vs FEM: {spearman_c:+.4f}")
    if spearman_d is not None:
        print(f"Predicted   Spearman vs FEM: {spearman_d:+.4f}")

    # Held-out predictions (Panel D triangle colours).
    heldout_metric_d = None
    if metric_d is not None and heldout_proj is not None and heldout_ids:
        # `pred_by_id` was built earlier; look up each held-out plate.
        ho_metric_list = []
        for hid in heldout_ids:
            ho_metric_list.append(pred_by_id.get(hid, np.nan))
        ho_metric_arr = np.array(ho_metric_list, dtype=np.float32)
        if np.isfinite(ho_metric_arr).all():
            heldout_metric_d = ho_metric_arr
            print(f"Held-out predictions for triangle colours: "
                  f"{heldout_metric_d.tolist()}")

    common_kwargs = dict(
        trajectory_dir=traj_dir,
        trajectory_subset=traj_subset,
        leia_all=leia if traj_dir else None,
        clean_panel_b=args.clean_panel_b,
        heldout_proj=heldout_proj,
        show_suptitle=not args.no_suptitle,
        metric_c=metric_c,
        c_paintable=c_paintable,
        metric_c_label=metric_c_label,
        metric_d=metric_d,
        d_paintable=d_paintable,
        heldout_metric_d=heldout_metric_d,
        spearman_a=spearman_a,
        spearman_c=spearman_c,
        spearman_d=spearman_d,
        grid_2x2=args.grid_2x2,
        panel_order=args.panel_order,
        minimal=args.minimal,
        metric_a_label=(r"$\langle \|\hat{u}_0\|^2 \rangle$"
                        if args.minimal else "Tokenizer reconstruction error"),
    )
    # Filename suffix includes the model-error metric so different runs don't
    # overwrite each other. total_work_err keeps the empty default for
    # backward compat with earlier outputs.
    metric_suffix = "" if args.metric_b == "total_work_err" else f"_{args.metric_b}"
    file_suffix = (
        suffix + metric_suffix
        + ("_clean" if args.clean_panel_b else "")
        + ("_heldout" if heldout_proj is not None else "")
        + (f"_consistency-{args.consistency_metric}" if metric_c is not None else "")
        + ("_panelD" if metric_d is not None else "")
        + ("_valonly" if args.val_only else "")
        + ("_2x2" if args.grid_2x2 else "")
        + ("_truth-first" if (args.panel_order == "truth-first"
                              and metric_d is not None
                              and not args.grid_2x2) else "")
        + ("_minimal" if args.minimal else "")
    )

    # --val-only: mask the rendering to OOD-head val-split candidates only.
    # Requires --ood-pred-file (the predictions.jsonl carries the in_train_split
    # flag that defines train vs val membership).
    if args.val_only:
        if not args.ood_pred_file:
            raise ValueError("--val-only requires --ood-pred-file (predictions.jsonl).")
        val_ids = set()
        val_seeds = set()
        with open(args.ood_pred_file) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("in_train_split") is False:
                    val_ids.add(rec["id"])
                    val_seeds.add(rec.get("seed_plate"))
        keep_mask = np.zeros(len(leia), dtype=bool)
        for i, e in enumerate(leia):
            cid = e["id"]
            sp = e.get("seed_plate")
            # Keep if (a) the candidate is a val mutant, or
            #         (b) the candidate is a depth=0 seed of a val-pool seed.
            if cid in val_ids or (sp in val_seeds and e["depth"] == 0):
                keep_mask[i] = True
        n_kept = int(keep_mask.sum())
        n_seed_kept = int((keep_mask & seeds_mask).sum())
        print(f"--val-only: kept {n_kept}/{len(leia)} candidates, "
              f"{n_seed_kept} val-pool seeds (diamonds).")
        # Mask all per-candidate arrays in lockstep.
        proj = proj[keep_mask]
        seeds_mask = seeds_mask[keep_mask]
        metric_a = metric_a[keep_mask]
        metric_b = metric_b[keep_mask]
        fem_paintable = fem_paintable[keep_mask]
        if metric_c is not None:
            metric_c = metric_c[keep_mask]
            c_paintable = c_paintable[keep_mask]
        if metric_d is not None:
            metric_d = metric_d[keep_mask]
            d_paintable = d_paintable[keep_mask]
        leia = [e for i, e in enumerate(leia) if keep_mask[i]]
        # Rebuild common_kwargs with the masked metric_c / metric_d / c_paintable / d_paintable.
        common_kwargs = dict(
            trajectory_dir=traj_dir,
            trajectory_subset=traj_subset,
            leia_all=leia if traj_dir else None,
            clean_panel_b=args.clean_panel_b,
            heldout_proj=heldout_proj,
            show_suptitle=not args.no_suptitle,
            metric_c=metric_c,
            c_paintable=c_paintable,
            metric_c_label=metric_c_label,
            metric_d=metric_d,
            d_paintable=d_paintable,
            heldout_metric_d=heldout_metric_d,
            spearman_a=spearman_a,
            spearman_c=spearman_c,
            spearman_d=spearman_d,
            grid_2x2=args.grid_2x2,
            panel_order=args.panel_order,
            minimal=args.minimal,
            metric_a_label=(r"$\langle \|\hat{u}_0\|^2 \rangle$"
                        if args.minimal else "Tokenizer reconstruction error"),
        )

    # 1×2 figure: PC1×PC2 only
    _plot_panels_grid(
        proj, seeds_mask, metric_a, metric_b, fem_paintable, args.metric_b,
        pc_pairs=[(0, 1, pc_label(0), pc_label(1))],
        suptitle=suptitle,
        out_pdf=os.path.join(out_dir, f"landscape_panels{file_suffix}.pdf"),
        out_png=os.path.join(out_dir, f"landscape_panels{file_suffix}.png"),
        **common_kwargs,
    )

    # 2×2 figure: PC1×PC2 row + PC2×PC3 row
    _plot_panels_grid(
        proj, seeds_mask, metric_a, metric_b, fem_paintable, args.metric_b,
        pc_pairs=[
            (0, 1, pc_label(0), pc_label(1)),
            (1, 2, pc_label(1), pc_label(2)),
        ],
        suptitle=suptitle,
        out_pdf=os.path.join(out_dir, f"landscape_panels{file_suffix}_4.pdf"),
        out_png=os.path.join(out_dir, f"landscape_panels{file_suffix}_4.png"),
        **common_kwargs,
    )



def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pca = sub.add_parser("pca", help="Fit PCA + render two scatter panels")
    p_pca.add_argument("--landscape-dir", type=str, default=None,
                       help="Primary landscape dir; if --run-root is set, "
                            "defaults to <run-root>/walk.")
    p_pca.add_argument("--run-root", type=str, default=None,
                       help="Reproduction run root; sets landscape-dir, "
                            "extra-landscape-dirs, consistency-file, "
                            "ood-pred-file, heldout-latents-file, and routes "
                            "outputs to <root>/figures/.")
    p_pca.add_argument(
        "--metric-b", type=str, default="total_work_err",
        choices=["total_work_err", "disp_l2_err_mean", "vm_corr_mean"],
        help="Which LEIA-vs-FEM error metric to color Panel B by.",
    )
    p_pca.add_argument(
        "--extra-landscape-dirs", type=str, nargs="+", default=None,
        help="Additional landscape dirs to merge into the figure. Their "
             "depth=0 entries are NOT marked as training seeds; only the "
             "primary --landscape-dir contributes the seed mask.",
    )
    p_pca.add_argument(
        "--show-trajectory", type=str, default=None,
        help="Path to a landscape dir containing novelty_meta.json + "
             "candidates.json. Enables per-seed beam-to-beam arrow overlay "
             "on Panel B.",
    )
    p_pca.add_argument(
        "--trajectory-seeds-subset", type=str, default=None,
        help="Comma-separated seed_ids to highlight at full alpha "
             "(other seeds fade to alpha=0.18). E.g. 'mg_007,mg_415,mg_421'.",
    )
    p_pca.add_argument(
        "--clean-panel-b", action="store_true",
        help="Strip the model-error column to only colored paintable scatter "
             "+ training-plate diamonds (drops No-FEM, FEM-failed, final "
             "iter-D beam markers, and trajectory arrows). Output filename "
             "gets a '_clean' suffix.",
    )
    p_pca.add_argument(
        "--heldout-latents-file", type=str, default=None,
        help="Path to a JSON file with pre-encoded held-out plate latents "
             "(e.g. design_search/outputs/heldout_latents.json). When set, "
             "the held-out plates are projected onto the fitted PCA basis "
             "and overlaid as red triangles.",
    )
    p_pca.add_argument(
        "--no-suptitle", action="store_true",
        help="Suppress the figure-wide suptitle.",
    )
    p_pca.add_argument(
        "--consistency-file", type=str, default=None,
        help="Path to a consistency.jsonl produced by "
             "`landscape.score --metric=consistency`. Adds a third column (Panel C) "
             "colored by the chosen consistency metric.",
    )
    p_pca.add_argument(
        "--consistency-metric", type=str, default="mean",
        choices=["mean", "t3", "t10", "slope"],
        help="Which scalar to extract per candidate from c_per_frame: "
             "mean (over all 10 frames), t3 (peak discriminator), t10 (final), "
             "or slope (c_t=10 − c_t=2).",
    )
    p_pca.add_argument(
        "--ood-pred-file", type=str, default=None,
        help="Path to predictions.jsonl produced by `ood.pipeline predict`. Adds a "
             "Panel D colored by predicted vm_corr (RdYlGn — green=healthy).",
    )
    p_pca.add_argument(
        "--val-only", action="store_true",
        help="Mask the rendering to only the OOD-head val-split candidates "
             "(the 1385 mutants from 11 unseen seeds) plus their val-pool "
             "seed diamonds. Held-out triangles are still drawn. "
             "Requires --ood-pred-file.",
    )
    p_pca.add_argument(
        "--grid-2x2", action="store_true",
        help="When all four panels (recon + true + consistency + predicted) "
             "are present and there's a single PC pair, arrange them in a "
             "2x2 grid instead of 1x4. Better for one-column paper layout.",
    )
    p_pca.add_argument(
        "--panel-order", type=str, default="truth-first",
        choices=["truth-first", "recon-first"],
        help="Column order for 1×N layouts. "
             "'truth-first' (default): A=model error, B=recon, C=consistency, "
             "D=predicted. 'recon-first' (legacy): A=recon, B=model error, "
             "C=consistency, D=predicted. Has no effect in --grid-2x2 mode.",
    )
    p_pca.add_argument(
        "--minimal", action="store_true",
        help="Paper-figure mode: suppress all subplot legends and corr-vs-FEM "
             "insets, and use compact single-letter colorbar labels (e for "
             "recon, c for dynamics consistency). Reader is expected to learn "
             "the mapping from the figure caption.",
    )
    p_pca.set_defaults(func=cmd_pca)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
