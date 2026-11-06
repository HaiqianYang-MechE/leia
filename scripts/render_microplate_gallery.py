#!/usr/bin/env python
"""MicroPlate gallery — produces two 2x4 videos in a single run.

Output 1 (dynamics): production LEIA rolls out 8 plates in parallel under
their own boundary conditions. Each panel = deformed mesh colored by von
Mises stress; per-step action label (stretch / twist / shear1 / shear2 in
{-1, 0, +1}) overlaid in the upper-left corner. Per-plate vmax.

Output 2 (unit cells): rotating-camera 3D view of the same 8 plates'
underlying unit cells (graph -> tet mesh via physics.meshing.mesh_graph).
Same n_frames as the dynamics video so the two read as a paired demo.

Both videos use a pale top→bottom gradient background, multi-light PBR-ish
shading, and SSAA, for a paper-figure look that holds up at submission
quality.

Usage:
    python scripts/render_microplate_gallery.py \\
        --tokenizer checkpoints/tokenizer/production_v2/best.pt \\
        --dynamics  checkpoints/dynamics/production_v2/best_ar.pt \\
        --npz-dir   data/microplate_lattice/npz \\
        --fps-dir   data/microplate_lattice/fps \\
        --mesh-dir  data/meshes/microplate_lattice \\
        --rve-bank  data/rve_bank
"""

import os
import sys
import argparse
import json
import time as time_mod
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pyvista as pv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leia.engine import LeiaEngine
from leia.render import (build_node_adjacency, laplacian_smooth,
                         load_npz_plate_data, composite_overlay,
                         compose_grid, open_writer, fit_camera_to_envelope)
from leia.tokenizer import decode_full_mesh, subsample_fps
from leia.stress import (precompute_stress_geometry, stress_from_disps,
                         cauchy_from_P, von_mises, neo_hookean_P,
                         compute_cumulative_bc)
from physics.meshing import mesh_graph

pv.OFF_SCREEN = True

# 3rd column = lowest-vf training plates after mg_111: mg_007 (22.9%) and
# mg_209 (25.6%) — true sparse lattices, in-distribution for production v2.
DEFAULT_PLATES = ['mg_111', 'mg_036', 'mg_007', 'mg_141',
                  'mg_213', 'mg_084', 'mg_209', 'mg_265']

# --- light / paper palette --------------------------------------------------
# Subtle top-to-bottom gradient (matches the old dark gradient direction —
# darker top, lighter bottom — but in pale tones).
BG_TOP    = (0.870, 0.895, 0.920)  # pale cool gray-blue at top
BG_BOTTOM = (0.985, 0.990, 0.995)  # near-white at bottom
TEXT_COLOR  = '#1A1A1A'
CELL_COLOR  = '#FFEDCE'  # warm cream
WIRE_COLOR  = '#666666'
PAPER_FONT  = 'serif'

STRESS_CMAP = 'RdPu'  # built-in: pale cream → pink → deep magenta-purple

ACTION_LABELS = ('stretch', 'twist', 'shear1', 'shear2')


def parse_args():
    p = argparse.ArgumentParser(
        description="MicroPlate gallery (dynamics + rotating unit cells)")
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--dynamics', required=True)
    p.add_argument('--npz-dir', required=True)
    p.add_argument('--fps-dir', required=True)
    p.add_argument('--mesh-dir', required=True)
    p.add_argument('--rve-bank', default='data/rve_bank',
                   help='Directory containing {mg_xxx}/graph.json')
    p.add_argument('--plates', type=str, default=','.join(DEFAULT_PLATES),
                   help='Comma-separated 8 plate IDs')
    p.add_argument('--traj-idx', type=int, default=0)
    p.add_argument('--subsample-n', type=int, default=10240)
    p.add_argument('--fps-core-n', type=int, default=8192)
    p.add_argument('--stress-mu', type=float, default=1.0)
    p.add_argument('--stress-lmbda', type=float, default=10.0)
    p.add_argument('--panel-size', type=int, default=720)
    p.add_argument('--text-font-size', type=int, default=10,
                   help='matplotlib font size (pt) for action label + colorbar')
    p.add_argument('--fps', type=int, default=8)
    p.add_argument('--smooth-iters', type=int, default=3)
    p.add_argument('--smooth-alpha', type=float, default=0.5)
    p.add_argument('--unit-cell-res', type=int, default=50,
                   help='SDF res_per_cell for unit-cell meshing (visualization)')
    p.add_argument('--rotation-degrees', type=float, default=360.0,
                   help='Total rotation arc for the unit-cell video')
    p.add_argument('--skip-dynamics', action='store_true',
                   help='Skip rollout precompute + dynamics video render; '
                        'render only the unit-cell rotating video')
    p.add_argument('--n-frames', type=int, default=31,
                   help='Frames for the unit-cell rotation video (used '
                        'when --skip-dynamics; otherwise inferred from data)')
    p.add_argument('--output-dir', default='outputs/microplate_gallery/')
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dynamics rollout (per plate)
# ---------------------------------------------------------------------------

def precompute_plate_rollout(plate_id, args, engine, device):
    pdata = load_npz_plate_data(args.npz_dir, args.fps_dir, args.mesh_dir,
                                plate_id, max_trajs=args.traj_idx + 1)
    coords = pdata['coords']
    topo = pdata['topology']
    outputs = pdata['outputs']
    fps_order = pdata['fps_order']
    inputs_all = pdata['inputs']  # (n_traj, n_inc, 4) raw {-1,0,+1}

    n_frames = outputs.shape[1]
    n_nodes = coords.shape[0]
    n_inc = n_frames - 1

    coords_f32 = coords.astype(np.float32)
    c_min = coords_f32.min(axis=0)
    c_max = coords_f32.max(axis=0)
    scale = np.maximum(c_max - c_min, 1e-8)

    rng = np.random.default_rng(42)
    sub_idx = subsample_fps(fps_order, args.subsample_n,
                            args.fps_core_n, rng)
    coords_sub = coords_f32[sub_idx]
    ref_sub_norm = ((coords_sub - c_min) / scale * 2 - 1).astype(np.float32)
    ref_full_norm = ((coords_f32 - c_min) / scale * 2 - 1).astype(np.float32)
    ref_full_t = torch.tensor(ref_full_norm, dtype=torch.float32,
                              device=device).unsqueeze(0)
    coord_scale_half = (scale / 2.0).astype(np.float32)

    dNdx, volumes, good = precompute_stress_geometry(coords, topo)
    avg_op = (build_node_adjacency(topo, n_nodes)
              if args.smooth_iters > 0 else None)

    inputs_traj = inputs_all[args.traj_idx]                    # (n_inc, 4) raw
    cum_bc = compute_cumulative_bc(inputs_all)[args.traj_idx]  # (n_inc, 4) scaled

    disp_0_sub = outputs[args.traj_idx, 0, sub_idx]
    prevq = engine.encode_frame(disp_0_sub, ref_sub_norm)
    prevq_prev = prevq.clone() if engine.velocity_input else None

    disps_per_frame, vm_per_frame, action_per_frame = [], [], []

    for fi in range(n_frames):
        e2e_d, sym6_head, vm_head, grad_u_phys = decode_full_mesh(
            engine.tokenizer, prevq, ref_full_t, engine.disp_scale, device,
            coord_scale=coord_scale_half)
        if avg_op is not None:
            e2e_d = laplacian_smooth(e2e_d, avg_op, args.smooth_iters,
                                     args.smooth_alpha)

        if vm_head is not None:
            vm_node = vm_head
        elif grad_u_phys is not None:
            F = np.eye(3) + grad_u_phys
            P = neo_hookean_P(F, mu=args.stress_mu, lmbda=args.stress_lmbda)
            vm_node = von_mises(cauchy_from_P(P, F)).astype(np.float32)
        else:
            _, _, _, vm_node = stress_from_disps(
                topo, e2e_d, dNdx, volumes, good,
                mu=args.stress_mu, lmbda=args.stress_lmbda)

        disps_per_frame.append(e2e_d)
        vm_per_frame.append(vm_node)
        action_per_frame.append(np.zeros(4, dtype=int) if fi == 0
                                else inputs_traj[fi - 1].astype(int))

        if fi < n_inc:
            bc = torch.tensor(cum_bc[fi], dtype=torch.float32,
                              device=device).unsqueeze(0)
            prev_state_arg = (prevq_prev.unsqueeze(0)
                              if engine.velocity_input else None)
            next_prevq, _ = engine.step_dynamics(
                prevq.unsqueeze(0), bc, prev_state=prev_state_arg)
            if engine.velocity_input:
                prevq_prev = prevq
            prevq = next_prevq.squeeze(0)

    all_vm = np.concatenate(vm_per_frame)
    pos = all_vm[all_vm > 0]
    vmax = float(np.percentile(pos, 99)) if len(pos) > 0 else 1.0

    # Deformation envelope (for camera fitting that survives peak stretch)
    env_min = coords.min(axis=0).astype(np.float64)
    env_max = coords.max(axis=0).astype(np.float64)
    for d in disps_per_frame:
        df = coords + d
        env_min = np.minimum(env_min, df.min(axis=0))
        env_max = np.maximum(env_max, df.max(axis=0))

    return {
        'plate_id': plate_id, 'coords': coords, 'topo': topo,
        'disps': disps_per_frame, 'vm_nodes': vm_per_frame,
        'action': action_per_frame, 'vmax': vmax, 'n_frames': n_frames,
        'env_min': env_min, 'env_max': env_max,
    }


# ---------------------------------------------------------------------------
# Game-engine plotter helpers
# ---------------------------------------------------------------------------

def _setup_cinematic_plotter(width, height):
    """Plotter with light gradient bg, soft three-light setup, SSAA."""
    p = pv.Plotter(off_screen=True, window_size=(width, height),
                   lighting=None)
    p.set_background(BG_BOTTOM, top=BG_TOP)
    p.enable_anti_aliasing('ssaa')
    # Softer lighting on light bg; key from upper-right, gentle fill, faint rim.
    p.add_light(pv.Light(position=(3, 3, 4), focal_point=(0, 0, 0),
                         color='white', intensity=0.85))
    p.add_light(pv.Light(position=(-3, -2, 2), focal_point=(0, 0, 0),
                         color=(0.85, 0.90, 1.0), intensity=0.35))
    p.add_light(pv.Light(position=(0, -3, -1), focal_point=(0, 0, 0),
                         color=(1.0, 0.95, 0.85), intensity=0.2))
    return p


def _render_action_buttons(action_step, width_px, height_px, font_size):
    """Game-controller-style 4-button grid encoding per-step action.

    Layout (2x2):
        [S] [T]       ← stretch, twist
        [1] [2]       ← shear1, shear2

    Fill color encodes action state in {-1, 0, +1}:
        -1 → orange (unloading)
         0 → light gray (rest)
        +1 → teal     (loading)
    """
    dpi = 200
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()

    state_face = {-1: '#E07A5F', 0: '#EDEDED', +1: '#2E86AB'}  # orange/gray/blue (colorblind-safe)
    edge_color = '#404040'
    label_color = '#1A1A1A'

    positions_labels = [
        (0.30, 0.72, 'S'),   # stretch
        (0.70, 0.72, 'T'),   # twist
        (0.30, 0.28, '1'),   # shear1
        (0.70, 0.28, '2'),   # shear2
    ]
    radius = 0.18
    for (x, y, lab), state in zip(positions_labels, action_step):
        circ = plt.Circle((x, y), radius,
                          facecolor=state_face[int(state)],
                          edgecolor=edge_color, linewidth=1.2)
        ax.add_patch(circ)
        ax.text(x, y, lab, ha='center', va='center',
                fontsize=font_size, family=PAPER_FONT,
                color=label_color, fontweight='bold')
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def _render_text_image(text, width_px, height_px, font_size,
                       color=TEXT_COLOR):
    """Render text → transparent RGBA image (matplotlib, paper-style font)."""
    dpi = 200
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(0.04, 0.96, text, ha='left', va='top',
            fontsize=font_size, family=PAPER_FONT, color=color,
            linespacing=1.3)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def _render_colorbar_strip(vmax, width_px, height_px, font_size):
    """Vertical colorbar with σ title on top, gradient bg matching the panel.

    Layout inside the strip:
      [ small left margin | bar (20% wide) | tick labels | right margin ]
    Background: same top→bottom pale gradient as pyvista, so the strip blends
    seamlessly with the dynamics panel rather than reading as a white patch.
    """
    dpi = 200
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_facecolor(BG_BOTTOM)  # matches the bottom of the gradient

    # Bg gradient (BG_TOP at top, BG_BOTTOM at bottom) — vertical column of
    # interpolated colors, drawn full-strip via imshow under the colorbar.
    n = 64
    grad = np.zeros((n, 1, 3), dtype=np.float32)
    for i in range(n):
        t = i / (n - 1)                                              # 0=bot, 1=top
        grad[i, 0] = (1 - t) * np.array(BG_BOTTOM) + t * np.array(BG_TOP)
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.imshow(grad, extent=[0, 1, 0, 1], aspect='auto',
                 interpolation='bilinear', origin='lower')
    ax_bg.set_axis_off()

    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
    ax = fig.add_axes([0.10, 0.12, 0.20, 0.76])
    cbar = matplotlib.colorbar.ColorbarBase(
        ax, cmap=STRESS_CMAP, norm=norm, orientation='vertical')
    ax.set_title(r'$\sigma$', fontsize=font_size + 2, family=PAPER_FONT,
                 color=TEXT_COLOR, pad=4)
    cbar.outline.set_edgecolor(TEXT_COLOR)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=font_size - 2,
                        width=0.6)
    for t in cbar.ax.get_yticklabels():
        t.set_family(PAPER_FONT)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    img = rgba[..., :3].copy()
    plt.close(fig)
    return img




def _fmt_signed(v):
    """-1 -> '-1', 0 -> ' 0', +1 -> '+1'  (constant-width)."""
    return '+1' if v > 0 else ('-1' if v < 0 else ' 0')


def format_action(action_step):
    s = action_step
    return (f"{ACTION_LABELS[0]}={_fmt_signed(s[0])}   "
            f"{ACTION_LABELS[1]}={_fmt_signed(s[1])}\n"
            f"{ACTION_LABELS[2]}={_fmt_signed(s[2])}   "
            f"{ACTION_LABELS[3]}={_fmt_signed(s[3])}")




def render_dynamics_panel(coords, topo, disp, vm_nodes, vmax, cam,
                          panel_size, action_step, font_size):
    """3D mesh (pyvista) + matplotlib colorbar + matplotlib text overlay.

    Layout (panel_size x panel_size):  [pyvista_mesh | colorbar_strip].
    Both overlays are matplotlib-rendered (paper-style serif font, fixed
    pixel size — no VTK auto-scaling, so text doesn't jump between frames).
    """
    ps = panel_size
    cb_w = max(110, ps // 6)        # ~17% reserved so tick labels don't clip
    pv_w = ps - cb_w

    deformed = coords + disp
    grid = pv.UnstructuredGrid({pv.CellType.TETRA: topo}, deformed)
    grid.point_data['vm'] = np.clip(vm_nodes, 0, vmax * 1.5)

    p = _setup_cinematic_plotter(pv_w, ps)
    p.add_mesh(grid, scalars='vm', clim=[0, vmax], cmap=STRESS_CMAP,
               smooth_shading=True, show_edges=False, show_scalar_bar=False,
               specular=0.3, specular_power=15,
               ambient=0.20, diffuse=0.65)
    p.camera_position = cam
    pv_img = p.screenshot(return_img=True)
    p.close()

    cb_img = _render_colorbar_strip(vmax, cb_w, ps, font_size)
    panel = np.concatenate([pv_img, cb_img], axis=1)

    btn_size = int(ps * 0.20)
    btn_img = _render_action_buttons(action_step, btn_size, btn_size,
                                     font_size + 2)
    panel = composite_overlay(panel, btn_img,
                               x=int(ps * 0.02), y=int(ps * 0.02))
    return panel


def render_unit_cell_panel(mesh_result, theta_deg, panel_size):
    grid = pv.UnstructuredGrid({pv.CellType.TETRA: mesh_result.cells},
                               mesh_result.points)
    p = _setup_cinematic_plotter(panel_size, panel_size)
    p.add_mesh(grid, color=CELL_COLOR, smooth_shading=True,
               specular=0.35, specular_power=12,
               ambient=0.45, diffuse=0.60)
    p.add_mesh(pv.Cube(x_length=1.96, y_length=1.96, z_length=1.96),
               style='wireframe', color=WIRE_COLOR, opacity=0.40,
               line_width=1.2)
    p.enable_eye_dome_lighting()
    p.hide_axes()
    p.camera_position = 'iso'
    p.camera.zoom(0.85)
    p.camera.Azimuth(theta_deg)
    img = p.screenshot(return_img=True)
    p.close()
    return img


# ---------------------------------------------------------------------------
# Unit cell meshing
# ---------------------------------------------------------------------------

def precompute_unit_cells(plate_ids, rve_bank_dir, res_per_cell):
    cells = {}
    for pid in plate_ids:
        gpath = os.path.join(rve_bank_dir, pid, 'graph.json')
        with open(gpath) as f:
            graph_data = json.load(f)
        t0 = time_mod.time()
        result = mesh_graph(graph_data, tiles=(1, 1, 1),
                            res_per_cell=res_per_cell)
        print(f"  unit cell {pid}: {len(result.points)} pts, "
              f"{len(result.cells)} tets [{time_mod.time()-t0:.0f}s]")
        cells[pid] = result
    return cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    plates = args.plates.split(',')
    if len(plates) != 8:
        sys.exit(f"Need exactly 8 plates; got {len(plates)}: {plates}")

    print("=" * 60)
    print("MicroPlate gallery — dynamics + rotating unit cells")
    print(f"Plates: {plates}")
    print("=" * 60)

    ps = args.panel_size

    if not args.skip_dynamics:
        engine = LeiaEngine(tokenizer_ckpt=args.tokenizer,
                            dynamics_ckpt=args.dynamics,
                            device=str(device))

        # 1) Dynamics rollouts
        print("\n--- dynamics rollouts ---")
        plate_data = []
        for pid in plates:
            print(f"\n[plate] {pid}")
            t0 = time_mod.time()
            pd = precompute_plate_rollout(pid, args, engine, device)
            plate_data.append(pd)
            print(f"  done in {time_mod.time()-t0:.0f}s, "
                  f"vmax={pd['vmax']:.3f}, n_frames={pd['n_frames']}")
        n_frames = min(p['n_frames'] for p in plate_data)

        # 2) Cameras
        print("\n--- camera fitting ---")
        cams = [fit_camera_to_envelope(pd['env_min'], pd['env_max'],
                                       args.panel_size, zoom=0.85)
                for pd in plate_data]

        # 3) Render dynamics video FIRST (so unit-cell meshing failure can't
        #    wipe out the dynamics video).
        dyn_path = os.path.join(args.output_dir,
                                f"{ts}_microplate_gallery.mp4")
        print(f"\n--- dynamics render -> {dyn_path} ---")
        writer = open_writer(dyn_path, args.fps)
        t_render = time_mod.time()
        for fi in range(n_frames):
            panels = []
            for pi, pd in enumerate(plate_data):
                panels.append(render_dynamics_panel(
                    pd['coords'], pd['topo'], pd['disps'][fi],
                    pd['vm_nodes'][fi], pd['vmax'], cams[pi], ps,
                    pd['action'][fi], args.text_font_size))
            writer.append_data(compose_grid(panels, 2, 4))
            if (fi + 1) % 5 == 0 or fi == 0:
                print(f"  frame {fi+1}/{n_frames}  "
                      f"[{time_mod.time()-t_render:.0f}s]")
        writer.close()
        print(f"Saved: {dyn_path}  [{time_mod.time()-t_render:.0f}s]")
    else:
        n_frames = args.n_frames
        print(f"\n--- skipping dynamics; unit-cell only, n_frames={n_frames} ---")

    # 4) Unit cell meshes (one per plate; reused across rotation frames)
    print("\n--- unit-cell meshing ---")
    try:
        cells = precompute_unit_cells(plates, args.rve_bank, args.unit_cell_res)
    except Exception as e:
        print(f"\nUnit-cell meshing failed: {e}")
        print("Skipping unit-cell video. Dynamics video already saved.")
        return

    # 5) Render unit-cell rotating video
    cell_path = os.path.join(args.output_dir,
                             f"{ts}_unit_cell_gallery.mp4")
    print(f"\n--- unit-cell render -> {cell_path} ---")
    writer = open_writer(cell_path, args.fps)
    t_render = time_mod.time()
    for fi in range(n_frames):
        theta = args.rotation_degrees * fi / max(n_frames - 1, 1)
        panels = []
        for pid in plates:
            panels.append(render_unit_cell_panel(cells[pid], theta, ps))
        writer.append_data(compose_grid(panels, 2, 4))
        if (fi + 1) % 5 == 0 or fi == 0:
            print(f"  frame {fi+1}/{n_frames}  "
                  f"[{time_mod.time()-t_render:.0f}s]")
    writer.close()
    print(f"Saved: {cell_path}  [{time_mod.time()-t_render:.0f}s]")

    print("\nAll done.")


if __name__ == '__main__':
    main()
