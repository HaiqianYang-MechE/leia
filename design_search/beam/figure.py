#!/usr/bin/env python3
"""Generate Figure 4: Surrogate-guided beam search over lattice topologies.

Panels in the paper figure:
  (a) Topology evolution filmstrip — best design at iterations 0, 3, 5, 7, 9
  (b) Convergence — beam search vs random search score over iterations
  (c) Search landscape scatter — stretch_work vs shear_work colored by vol_frac
  (d) Per-candidate cost — LEIA inference vs FEM solve

Two-step workflow:
  1. Prepare filmstrip (GPU + xvfb-run required):
     xvfb-run -a python -m design_search.beam.figure prepare-filmstrip \
         --beam-results design_search/outputs/full_001/results.json

  2. Generate figure (no GPU needed):
     python -m design_search.beam.figure figure \
         --beam-results design_search/outputs/full_001/results.json \
         --output design_search/outputs/figure4.pdf
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

from design_search.core.objectives import compute_work, volume_fraction
from design_search.beam.fem_validate import best_candidate_at_iteration


# Paper-style typography for the figure.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ========== HELPERS ==========

def best_per_iteration(all_candidates):
    """Extract the best score seen so far at each iteration."""
    iters = sorted(set(c["iteration"] for c in all_candidates))
    best_score = 0
    convergence = []
    for it in iters:
        iter_cands = [c for c in all_candidates if c["iteration"] == it]
        iter_best = max(c["score"] for c in iter_cands)
        best_score = max(best_score, iter_best)
        convergence.append((it, best_score))
    return convergence


# ========== FILMSTRIP RENDERING (pyvista + xvfb-run) ==========

SNAPSHOT_ITERS = [-1, 3, 5, 7, 9]   # -1 = raw seed, then best-so-far at these iterations


def snapshot_iters(all_candidates):
    """Iterations drawn in the filmstrip: SNAPSHOT_ITERS for the paper's
    10-iteration search; a shorter run shows the seed plus up to four of the
    iterations it actually reached (never a label for one that did not run)."""
    max_it = max((c["iteration"] for c in all_candidates), default=-1)
    if max_it >= SNAPSHOT_ITERS[-1]:
        return list(SNAPSHOT_ITERS)
    if max_it < 0:
        return [-1]
    n = min(len(SNAPSHOT_ITERS) - 1, max_it + 1)
    return [-1] + sorted({int(round(x)) for x in np.linspace(0, max_it, n)})


def compute_camera_position(coords, topo, panel_size=1200):
    """Compute a fixed isometric camera for a plate mesh with generous framing."""
    import pyvista as pv

    grid = pv.UnstructuredGrid({pv.CellType.TETRA: topo}, coords)
    p = pv.Plotter(off_screen=True, window_size=(panel_size, panel_size))
    p.add_mesh(grid)
    p.view_isometric()
    p.camera.zoom(0.6)  # pull back to avoid clipping deformed plates
    cam = p.camera_position
    p.close()
    return cam


def prepare_filmstrip(beam_data, output_dir):
    """Mesh and render undeformed plate snapshots for the filmstrip.

    Renders each plate at rest with a uniform color to show topology.
    Requires xvfb-run for pyvista offscreen. No GPU inference needed.
    Saves PNG images to output_dir/filmstrip_iter{N}.png.
    """
    import pyvista as pv
    pv.OFF_SCREEN = True

    from design_search.core.config import ServingConfig
    from design_search.core.meshing import generate_plate_mesh, MeshingError

    config = ServingConfig()
    all_cands = beam_data["all_candidates"]

    os.makedirs(output_dir, exist_ok=True)

    # Collect snapshot candidates
    snapshots = []
    for it in snapshot_iters(all_cands):
        cand = best_candidate_at_iteration(all_cands, it)
        if cand is not None:
            snapshots.append((it, cand))

    # Mesh all snapshots — skip frames whose mesh fails
    snapshot_data = []
    for it, cand in snapshots:
        print(f"  Filmstrip iter {it}: meshing...")
        try:
            mesh = generate_plate_mesh(cand["graph"], config)
        except MeshingError as e:
            print(f"    SKIP frame {it}: mesh failed ({e})")
            continue
        snapshot_data.append({
            "iteration": it,
            "score": cand["score"],
            "vol_frac": cand["vol_frac"],
            "coords": mesh.points,
            "topo": mesh.cells,
        })

    # Compute camera from first snapshot (seed) — shared across all
    cam = compute_camera_position(snapshot_data[0]["coords"],
                                  snapshot_data[0]["topo"])

    # Render each snapshot: undeformed, uniform color
    for s in snapshot_data:
        grid = pv.UnstructuredGrid({pv.CellType.TETRA: s["topo"]}, s["coords"])

        p = pv.Plotter(off_screen=True, window_size=(1200, 1200))
        p.add_mesh(grid, color="#7DB4DD", show_edges=False, opacity=1.0)
        p.set_background("white")
        p.camera_position = cam
        img = p.screenshot(return_img=True)
        p.close()

        path = os.path.join(output_dir, f"filmstrip_iter{s['iteration']:03d}.png")
        Image.fromarray(img).save(path)
        print(f"  Saved {path} (score={s['score']:.0f}, "
              f"vf={s['vol_frac']:.4f})")

    # Save metadata
    meta = {
        "snapshot_iters": [s["iteration"] for s in snapshot_data],
        "scores": [s["score"] for s in snapshot_data],
        "vol_fracs": [s["vol_frac"] for s in snapshot_data],
    }
    with open(os.path.join(output_dir, "filmstrip_meta.json"), "w") as f:
        json.dump(meta, f)

    print(f"  Filmstrip ready: {len(snapshot_data)} snapshots")


# ========== PANEL RENDERERS ==========

def panel_a_filmstrip(fig, gs_row, beam_data, filmstrip_dir):
    """(a) Topology evolution filmstrip — 5 plate mesh snapshots."""
    meta_path = os.path.join(filmstrip_dir, "filmstrip_meta.json")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

        for col_idx, (it, score) in enumerate(
                zip(meta["snapshot_iters"], meta["scores"])):
            ax = fig.add_subplot(gs_row[col_idx])
            img_path = os.path.join(filmstrip_dir,
                                     f"filmstrip_iter{it:03d}.png")
            if os.path.exists(img_path):
                img = np.array(Image.open(img_path).convert("RGB"))
                # Auto-crop whitespace
                mask = (img < 250).any(axis=2)
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                if rows.any() and cols.any():
                    r0, r1 = np.where(rows)[0][[0, -1]]
                    c0, c1 = np.where(cols)[0][[0, -1]]
                    pad = 10
                    r0 = max(0, r0 - pad)
                    r1 = min(img.shape[0], r1 + pad)
                    c0 = max(0, c0 - pad)
                    c1 = min(img.shape[1], c1 + pad)
                    img = img[r0:r1, c0:c1]
                ax.imshow(img)

            label = "seed" if it == -1 else f"iter {it}"
            ax.set_title(f"{label}  (score {score:.0f})",
                         fontsize=FONT_LABEL, pad=4)
            ax.axis("off")
    else:
        all_cands = beam_data["all_candidates"]
        for col_idx, it in enumerate(snapshot_iters(all_cands)):
            ax = fig.add_subplot(gs_row[col_idx], projection="3d")
            cand = best_candidate_at_iteration(all_cands, it)
            if cand is None:
                continue
            label = "seed" if it == -1 else f"iter {it}"
            _render_graph_wireframe(ax, cand["graph"],
                                    title=f"{label}  (score {cand['score']:.0f})")


# ========== FONT SIZES (consistent hierarchy, paper-ready) ==========

FONT_PANEL = 22       # panel labels: (a), (b), ...
FONT_LABEL = 20       # axis labels, filmstrip titles
FONT_TICK = 18        # tick labels
FONT_LEGEND = 18      # legend entries
FONT_ANNOT = 18       # annotations


def _render_graph_wireframe(ax, graph, title=""):
    """Fallback: render RVE graph as 3D wireframe."""
    nodes = np.array(graph["nodes"])
    for beam in graph["beams"]:
        i, j = beam["idx"]
        r = (beam.get("r_start", 0.05) + beam.get("r_end", 0.05)) / 2
        ax.plot([nodes[i, 0], nodes[j, 0]],
                [nodes[i, 1], nodes[j, 1]],
                [nodes[i, 2], nodes[j, 2]],
                color="steelblue", linewidth=max(0.5, min(4, r * 80)), alpha=0.8)
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2],
               color="steelblue", s=20, zorder=5)
    ax.set_title(title, fontsize=FONT_LABEL, pad=2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=25, azim=135)


def _style_ax(ax, xlabel="", ylabel="", title=""):
    """Apply consistent styling to an axis."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    if title:
        ax.set_title(title, fontsize=FONT_PANEL, fontweight="bold")
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(True, alpha=0.3)


def panel_b_convergence(ax, beam_data, random_data=None, fem_data=None):
    """(b) Convergence — best score vs iteration, with FEM validation line."""
    cands = beam_data["all_candidates"]

    # LEIA convergence: best-so-far score per iteration.
    conv = best_per_iteration(cands)
    iters = [c[0] for c in conv]
    scores = [c[1] for c in conv]

    ax.plot(iters, scores, "o-", color="#2196F3", linewidth=2.5, markersize=5,
            label="LEIA", zorder=3)

    # FEM validation line (skip failed points with score=0).
    # iter -1 (raw seed) is naturally included in fem_points by fem_validate.py.
    if fem_data and "fem_points" in fem_data:
        fem_iters = [p["iteration"] for p in fem_data["fem_points"]
                     if p["fem_score"] > 0]
        fem_scores = [p["fem_score"] for p in fem_data["fem_points"]
                      if p["fem_score"] > 0]
        ax.plot(fem_iters, fem_scores, "o--", color="#2196F3", linewidth=2,
                markersize=5, alpha=0.5, label="FEM", zorder=4)

    # Seed diamond: explicit lookup of the iteration=-1 candidate.
    seed_cand = next((c for c in cands if c["iteration"] == -1), None)
    if seed_cand is not None:
        ax.plot(-1, seed_cand["score"], "D", color="#FF9800", markersize=14,
                zorder=5, markeredgecolor="black", markeredgewidth=0.8)

    # Winner star: global best across all candidates (not just iter D-1 best).
    # Robust to non-monotonic searches; agrees with the LEIA convergence line.
    winner_cand = max(cands, key=lambda c: c["score"])
    ax.plot(winner_cand["iteration"], winner_cand["score"], "*",
            color="#FF9800", markersize=22, zorder=5,
            markeredgecolor="black", markeredgewidth=0.8)

    ax.legend(fontsize=FONT_LEGEND - 1, loc="upper left")
    _style_ax(ax, "Iteration", "Score")
    ax.text(-0.14, 1.04, "(b)", transform=ax.transAxes, fontsize=FONT_PANEL,
            fontweight="bold", va="bottom", ha="left")


def panel_c_scatter(ax, beam_data, trajectory_mode="best"):
    """(c) Search landscape — stretch_work vs shear_work with trajectory.

    trajectory_mode:
      "best"     — position of the best-so-far candidate at each iteration
                   (default; trajectory ends exactly on the winner star but can
                   jump non-monotonically in (sw, shw) when score is dominated
                   by vol_frac).
      "centroid" — mean (sw, shw) over all candidates per iteration. Smoother
                   trajectory through the search cloud, but the endpoint is
                   the cloud's centroid, not the winner.
    """
    cands = beam_data["all_candidates"]
    sw = [c["stretch_work"] for c in cands]
    shw = [c["shear_work"] for c in cands]
    vf = [c["vol_frac"] for c in cands]
    scores = [c["score"] for c in cands]

    sizes = np.array(scores)
    sizes = 10 + 60 * (sizes - min(sizes)) / (max(sizes) - min(sizes) + 1e-8)

    sc = ax.scatter(sw, shw, c=vf, s=sizes * 2, cmap="RdYlBu_r", alpha=0.5,
                    edgecolors="gray", linewidths=0.3, zorder=2)

    # Iteration trajectory.
    traj_sw, traj_shw = [], []
    if trajectory_mode == "centroid":
        iters = sorted(set(c["iteration"] for c in cands if c["iteration"] >= 0))
        for it in iters:
            ic = [c for c in cands if c["iteration"] == it]
            traj_sw.append(np.mean([c["stretch_work"] for c in ic]))
            traj_shw.append(np.mean([c["shear_work"] for c in ic]))
    else:  # "best" — best-so-far candidate position
        iters = sorted(set(c["iteration"] for c in cands))
        best_score_so_far = -np.inf
        best_cand_so_far = None
        for it in iters:
            ic = [c for c in cands if c["iteration"] == it]
            iter_best = max(ic, key=lambda c: c["score"])
            if iter_best["score"] > best_score_so_far:
                best_score_so_far = iter_best["score"]
                best_cand_so_far = iter_best
            traj_sw.append(best_cand_so_far["stretch_work"])
            traj_shw.append(best_cand_so_far["shear_work"])

    # Draw trajectory with arrows
    for i in range(len(traj_sw) - 1):
        ax.annotate("", xy=(traj_sw[i+1], traj_shw[i+1]),
                     xytext=(traj_sw[i], traj_shw[i]),
                     arrowprops=dict(arrowstyle="-|>", color="black",
                                     lw=2.5, mutation_scale=18),
                     zorder=4)
    ax.plot(traj_sw, traj_shw, "o", color="black", markersize=5, zorder=4)

    # Highlight seed (iter=-1) and global-best winner.
    seed = next((c for c in cands if c["iteration"] == -1), cands[0])
    winner = max(cands, key=lambda c: c["score"])
    ax.plot(seed["stretch_work"], seed["shear_work"], "D", color="#FF9800",
            markersize=14, zorder=5, markeredgecolor="black",
            markeredgewidth=0.8, label="Seed")
    ax.plot(winner["stretch_work"], winner["shear_work"], "*", color="#FF9800",
            markersize=22, zorder=5, markeredgecolor="black",
            markeredgewidth=0.8, label="Winner")

    cb = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Volume fraction", fontsize=FONT_LABEL)
    cb.ax.tick_params(labelsize=FONT_TICK)

    # Tighter axis limits to show detail (exclude outliers)
    sw_arr = np.array(sw)
    shw_arr = np.array(shw)
    ax.set_xlim(0, np.percentile(sw_arr, 98) * 1.15)
    ax.set_ylim(0, np.percentile(shw_arr, 98) * 1.15)

    ax.legend(fontsize=FONT_LEGEND, loc="lower right")
    _style_ax(ax, "Stretch work", "Shear work")
    ax.text(-0.14, 1.04, "(c)", transform=ax.transAxes, fontsize=FONT_PANEL,
            fontweight="bold", va="bottom", ha="left")


def panel_d_per_candidate_cost(ax, bench_data):
    """(d) Per-candidate cost: LEIA inference vs FEM solve.

    Both bars exclude meshing (shared by both methods) and, with the bench's
    default settings, are measured serially on the same set of plates with no
    concurrent jobs, at one mesh resolution (the paper plots the same
    measurement as lines across resolutions). If the bench copied the
    validation stage's parallel-pool FEM times instead (`fem_source` other
    than `serial_direct`), the panel says so. Sample sizes (n) live in the
    figure caption, not on the plot.
    """
    if not bench_data:
        ax.text(0.5, 0.5, "stage 3 (bench) not run\n— per-candidate cost unavailable",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=FONT_LABEL, color="#888888")
        ax.set_xticks([]); ax.set_yticks([])
        _style_ax(ax, "", "")
        return

    plates = bench_data["plates"]
    leia_times = [p["leia_time_s"] for p in plates
                  if p.get("leia_time_s") is not None]
    # Include 'ok' and 'partial' (real solver wall-clock); exclude only 'error'
    # status (worker crash). Partial = some Newton directions diverged, but
    # the timing is still a real measurement of solver work.
    fem_times = [p["fem_time_s"] for p in plates
                 if p.get("fem_status") in ("ok", "partial")
                 and p.get("fem_time_s", 0) > 0]

    leia_mean = float(np.mean(leia_times)) if leia_times else 0.0
    leia_std = float(np.std(leia_times)) if leia_times else 0.0
    fem_mean = float(np.mean(fem_times)) if fem_times else 0.0
    fem_std = float(np.std(fem_times)) if fem_times else 0.0

    means = [leia_mean, fem_mean]
    stds = [leia_std, fem_std]
    # Match the LEIA/FEM colors used in fig_fps_scaling (blue + dark red).
    colors = ["#2196F3", "#B22222"]

    x_positions = np.array([0, 1])
    bar_width = 0.55
    bplot = ax.bar(
        x_positions, means, width=bar_width,
        color=colors, edgecolor="black", linewidth=1.2,
        yerr=stds,
        error_kw=dict(ecolor="black", capsize=8, capthick=2.0, lw=2.0),
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(["LEIA", "FEM"])
    ax.set_xlim(-0.6, 1.6)

    y_top = (max(means) + max(stds)) * 1.22 if max(means) > 0 else 1.0
    ax.set_ylim(0, y_top)

    for bar, m, s in zip(bplot, means, stds):
        label = f"{m:.1f}±{s:.1f}s"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + y_top * 0.025,
                label, ha="center", va="bottom",
                fontsize=FONT_ANNOT - 2, fontweight="bold")

    fem_source = str(bench_data.get("fem_source", ""))
    if fem_source and fem_source != "serial_direct":
        ax.text(0.5, 0.97, "FEM times partly from the\nparallel validation pool",
                ha="center", va="top", transform=ax.transAxes,
                fontsize=FONT_ANNOT - 3, color="#B22222")

    _style_ax(ax, "", "Time per candidate (s)")


def _save_fig(fig, path):
    """Save figure as PDF + PNG."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    png = path.replace(".pdf", ".png")
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ========== MAIN FIGURE: (a) filmstrip, (b) convergence, (c) scatter, (d) cost ==========

def generate_figure(beam_data, random_data=None, output_path="figure4.pdf",
                    filmstrip_dir=None, fem_data=None, bench_data=None,
                    trajectory_mode="best"):
    """Generate the main Figure 4: (a) filmstrip, (b) convergence, (c) scatter, (d) cost.

    Panel (d) shows per-candidate cost bars from `bench_data`.
    """
    if filmstrip_dir is None:
        filmstrip_dir = os.path.join(os.path.dirname(output_path), "filmstrip")

    fig = plt.figure(figsize=(18, 9))

    # 2 rows:
    #   Row 0: filmstrip (5 panels, full width)
    #   Row 1: (b) | (c) | (d)  — 3 equal panels
    gs_outer = gridspec.GridSpec(2, 1, figure=fig,
                                 height_ratios=[0.55, 1.0],
                                 hspace=0.15,
                                 left=0.06, right=0.97,
                                 top=0.91, bottom=0.08)

    # --- Row 0: filmstrip ---
    gs_film = gridspec.GridSpecFromSubplotSpec(
        1, 5, subplot_spec=gs_outer[0], wspace=0.03)
    panel_a_filmstrip(fig, gs_film, beam_data, filmstrip_dir)

    fig.text(0.03, 0.99, "(a)", fontsize=FONT_PANEL,
             fontweight="bold", va="top")

    # Arrow + label above filmstrip
    from matplotlib.patches import FancyArrowPatch
    arrow = FancyArrowPatch(
        (0.18, 0.95), (0.82, 0.95),
        transform=fig.transFigure, figure=fig,
        arrowstyle="-|>", mutation_scale=20,
        color="black", linewidth=2.0,
    )
    fig.patches.append(arrow)
    fig.text(0.50, 0.955, "optimization", ha="center", va="bottom",
             fontsize=FONT_LABEL, style="italic",
             transform=fig.transFigure)

    # --- Row 1: b | c | d ---
    gs_row = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_outer[1],
        width_ratios=[1.0, 1.0, 1.1],
        wspace=0.40)

    ax_b = fig.add_subplot(gs_row[0])
    panel_b_convergence(ax_b, beam_data, random_data, fem_data=fem_data)

    ax_c = fig.add_subplot(gs_row[1])
    panel_c_scatter(ax_c, beam_data, trajectory_mode=trajectory_mode)

    ax_d = fig.add_subplot(gs_row[2])
    panel_d_per_candidate_cost(ax_d, bench_data)
    ax_d.text(-0.14, 1.04, "(d)", transform=ax_d.transAxes, fontsize=FONT_PANEL,
              fontweight="bold", va="bottom", ha="left")

    # Save
    _save_fig(fig, output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 4")
    subparsers = parser.add_subparsers(dest="command")

    # Step 1: prepare filmstrip (GPU + xvfb-run required)
    p_film = subparsers.add_parser("prepare-filmstrip",
                                    help="Render plate mesh snapshots (GPU + xvfb-run)")
    p_film.add_argument("--beam-results", type=str, required=True)
    p_film.add_argument("--output-dir", type=str, default=None,
                        help="Dir for filmstrip images (default: alongside results)")

    # Step 2: generate figure (no GPU needed)
    p_fig = subparsers.add_parser("figure", help="Generate Figure 4")
    p_fig.add_argument("--beam-results", type=str, required=True)
    p_fig.add_argument("--fem-results", type=str, default=None)
    p_fig.add_argument("--bench-results", type=str, default=None,
                        help="Path to bench.json from design_search.beam.bench "
                             "(figure4.sh stage 3); required for panel (d).")
    p_fig.add_argument("--output", type=str,
                        default="design_search/outputs/figure4.pdf")
    p_fig.add_argument("--filmstrip-dir", type=str, default=None)
    p_fig.add_argument("--trajectory-mode", type=str, default="best",
                        choices=["best", "centroid"],
                        help="Panel (c) trajectory: 'best' (best-so-far per "
                             "iter, ends on winner star) or 'centroid' "
                             "(mean of all per-iter candidates, smoother but "
                             "endpoint != winner). Default 'best'.")

    args = parser.parse_args()

    if args.command == "prepare-filmstrip":
        print("Loading beam search results...")
        with open(args.beam_results) as f:
            beam_data = json.load(f)

        filmstrip_dir = args.output_dir
        if filmstrip_dir is None:
            filmstrip_dir = os.path.join(
                os.path.dirname(args.beam_results), "filmstrip")

        print(f"Preparing filmstrip ({len(snapshot_iters(beam_data['all_candidates']))} snapshots)...")
        prepare_filmstrip(beam_data, filmstrip_dir)

    elif args.command == "figure":
        print("Loading beam search results...")
        with open(args.beam_results) as f:
            beam_data = json.load(f)
        print(f"  {beam_data['n_candidates']} candidates, "
              f"winner score={beam_data['winner']['score']:.2f}")

        fem_data = None
        if args.fem_results and os.path.exists(args.fem_results):
            print("Loading FEM validation results...")
            with open(args.fem_results) as f:
                fem_data = json.load(f)
            print(f"  FEM time: {fem_data['fem_time_per_plate_s']:.1f}s per plate, "
                  f"{len(fem_data.get('fem_points', []))} validation points")

        bench_data = None
        if args.bench_results and os.path.exists(args.bench_results):
            print("Loading per-candidate bench results...")
            with open(args.bench_results) as f:
                bench_data = json.load(f)
            s = bench_data["summary"]
            print(f"  LEIA: {s['leia_mean_s']:.2f}±{s['leia_std_s']:.2f}s "
                  f"(n={s['leia_n']}); "
                  f"FEM: {s['fem_mean_s']:.1f}±{s['fem_std_s']:.1f}s "
                  f"(n={s['fem_n']})")

        generate_figure(beam_data, None, args.output,
                        filmstrip_dir=args.filmstrip_dir,
                        fem_data=fem_data, bench_data=bench_data,
                        trajectory_mode=args.trajectory_mode)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
