#!/usr/bin/env python
"""E2E dashboard video: GT vs Recon stress fields + F-D line plots.

Layout per trajectory row (3 columns):
  | GT stress field | Recon stress field | [Fx vs dx] [Mx vs θ] |
  |                 |                    | [Fy vs dy] [Fz vs dz] |

Stress panels: deformed tet mesh colored by von Mises (inferno, shared vmax).
  GT uses FEM vm_nodes. Recon uses the tokenizer's stress-head output.
F-D plots: progressive GT and Recon curves with marker on current frame.

Usage:
    python scripts/render_e2e_dashboard.py \
        --tokenizer checkpoints/tokenizer/production_v2/best.pt \
        --dynamics checkpoints/dynamics/production_v2/best_ar.pt \
        --npz-dir   data/microplate_lattice/npz \
        --fps-dir   data/microplate_lattice/fps \
        --mesh-dir  data/meshes/microplate_lattice \
        --plate-id mg_023 --n-trajs 2
"""

import os
import sys
import argparse
import time as time_mod
from datetime import datetime

import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyvista as pv
import imageio

from leia.engine import LeiaEngine
from leia.tokenizer import decode_full_mesh, subsample_fps
from leia.stress import (precompute_stress_geometry, stress_from_disps,
                         cauchy_from_P, von_mises, neo_hookean_P,
                         identify_right_boundary_nodes,
                         precompute_boundary_geometry,
                         boundary_reaction_force,
                         boundary_force_from_grad_head,
                         boundary_force_from_stress_head,
                         compute_gen_disp,
                         compute_cumulative_bc,
                         MICROPLATE_LATTICE_SPEEDS as SPEEDS)

pv.OFF_SCREEN = True

FD_LABELS = [('$F_x$', '$d_x$'), ('$M_x$', r'$\theta$'),
             ('$F_y$', '$d_y$'), ('$F_z$', '$d_z$')]


def parse_args():
    parser = argparse.ArgumentParser(
        description="E2E dashboard: GT vs Recon stress + F-D curves")
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--dynamics', required=True)
    parser.add_argument('--npz-dir', type=str, required=True,
                        help='Directory with {plate_id}/traj_*.npz files')
    parser.add_argument('--fps-dir', type=str, required=True,
                        help='Directory with {plate_id}_fps_order.npy files')
    parser.add_argument('--mesh-dir', type=str, required=True,
                        help='Directory with {plate_id}/mesh.npz files (for topology)')
    parser.add_argument('--curated-plates', type=str, default=None,
                        help='JSON file with list of plate IDs')
    parser.add_argument('--max-trajs', type=int, default=5,
                        help='Max trajectories per plate')
    # Common options
    parser.add_argument('--plate-id', type=str, default=None)
    parser.add_argument('--subsample-n', type=int, default=10240)
    parser.add_argument('--fps-core-n', type=int, default=8192)
    parser.add_argument('--action-scale', type=float, default=1.0)
    parser.add_argument('--stress-mu', type=float, default=1.0)
    parser.add_argument('--stress-lmbda', type=float, default=10.0)
    parser.add_argument('--n-trajs', type=int, default=2)
    parser.add_argument('--traj-indices', type=str, default=None)
    parser.add_argument('--output-dir', default='outputs/e2e_dashboard/')
    parser.add_argument('--panel-size', type=int, default=512)
    parser.add_argument('--fps', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--smooth-iters', type=int, default=3,
                        help='Laplacian-smoothing iterations on decoded '
                             'displacement (cosmetic; removes isolated-node '
                             'spikes at high-stress junctions where the '
                             'cross-attention decoder is less reliable at '
                             'full-mesh queries). 0 disables smoothing.')
    parser.add_argument('--smooth-alpha', type=float, default=0.5,
                        help='Smoothing mixing factor per iteration '
                             '(u_new = (1-alpha)*u + alpha*mean(u_neighbors))')
    parser.add_argument('--teacher-force', action='store_true',
                        help='Re-encode GT displacement at every frame '
                             'instead of using dynamics-predicted prevq. '
                             'Isolates tokenizer reconstruction quality from '
                             'dynamics rollout drift. Use to diagnose whether '
                             'spikes come from the tokenizer or from dynamics.')
    parser.add_argument('--save-fd-h5', action='store_true',
                        help='After rendering, also dump the per-traj F-D '
                             'arrays (gt_gen_force/disp, e2e_gen_force/disp) '
                             'to an HDF5 next to the MP4. Enables downstream '
                             'replotting and numerical comparison without '
                             're-rendering.')
    return parser.parse_args()




def render_stress_panel(coords, topo, disp, vm_nodes, vmax, cam, panel_size,
                        label=None):
    """Render deformed mesh colored by von Mises with label and colorbar."""
    deformed = coords + disp
    grid = pv.UnstructuredGrid({pv.CellType.TETRA: topo}, deformed)
    grid.point_data['VM'] = np.clip(vm_nodes, 0, vmax * 1.5)

    p = pv.Plotter(off_screen=True, window_size=(panel_size, panel_size))
    p.add_mesh(grid, scalars='VM', clim=[0, vmax], cmap='inferno',
               show_edges=False, show_scalar_bar=True,
               scalar_bar_args=dict(
                   title='stress',
                   title_font_size=14,
                   label_font_size=14,
                   width=0.50, height=0.06,
                   position_x=0.25, position_y=0.02,
                   fmt='%.2f', n_labels=5,
                   vertical=False,
               ))
    p.camera_position = cam
    if label:
        p.add_text(label, position='upper_left', font_size=18,
                   color='white', shadow=True)
    img = p.screenshot(return_img=True)
    p.close()
    return img


def render_fd_panel(gt_gen_force, gt_gen_disp, recon_gen_force, recon_gen_disp,
                    current_frame, panel_size):
    """Render 2x2 F-D line plots as an image array.

    Both GT and Recon curves are progressive (drawn up to current_frame).
    A marker indicates the current step.
    """
    dpi = 150
    fig_size = panel_size / dpi
    fig, axes = plt.subplots(2, 2, figsize=(fig_size, fig_size), dpi=dpi)
    fig.patch.set_facecolor('black')
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.95,
                        wspace=0.45, hspace=0.45)

    n_frames = gt_gen_force.shape[0]

    for idx, (ax, (f_label, d_label)) in enumerate(zip(axes.flat, FD_LABELS)):
        ax.set_facecolor('#1a1a1a')

        # Full range for axis limits (from GT)
        d_full = gt_gen_disp[:, idx]
        f_full = gt_gen_force[:, idx]
        d_range = d_full.max() - d_full.min()
        f_range = f_full.max() - f_full.min()
        if d_range < 1e-10:
            d_range = 1.0
        if f_range < 1e-10:
            f_range = 1.0
        ax.set_xlim(d_full.min() - 0.1 * d_range, d_full.max() + 0.1 * d_range)
        ax.set_ylim(f_full.min() - 0.15 * f_range, f_full.max() + 0.15 * f_range)

        fi = current_frame  # 0..n_frames-1

        # Progressive GT curve
        if fi > 0:
            ax.plot(gt_gen_disp[:fi+1, idx], gt_gen_force[:fi+1, idx],
                    '-', color='#4fc3f7', linewidth=2.0, label='GT')
            ax.plot(gt_gen_disp[fi, idx], gt_gen_force[fi, idx],
                    'o', color='#4fc3f7', markersize=6, zorder=5)

        # Progressive Recon curve
        if fi > 0:
            ax.plot(recon_gen_disp[:fi+1, idx], recon_gen_force[:fi+1, idx],
                    '--', color='#ff8a65', linewidth=2.0, label='Recon')
            ax.plot(recon_gen_disp[fi, idx], recon_gen_force[fi, idx],
                    's', color='#ff8a65', markersize=5, zorder=5)

        ax.set_xlabel(d_label, color='white', fontsize=10, labelpad=2)
        ax.set_ylabel(f_label, color='white', fontsize=10, labelpad=2)
        ax.tick_params(colors='white', labelsize=8, pad=2,
                       length=3, width=0.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        for spine in ax.spines.values():
            spine.set_color('#555555')
            spine.set_linewidth(0.5)
        if idx == 0 and fi > 0:
            ax.legend(fontsize=7, loc='upper right',
                      facecolor='#1a1a1a', edgecolor='#555555',
                      labelcolor='white', handlelength=1.2,
                      borderpad=0.3, handletextpad=0.4)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return img


from leia.render import (build_node_adjacency, laplacian_smooth,
                         load_npz_plate_data, discover_npz_plates)


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 60)
    print("E2E Dashboard: GT vs Recon stress + F-D curves")
    print("=" * 60)

    engine = LeiaEngine(tokenizer_ckpt=args.tokenizer,
                        dynamics_ckpt=args.dynamics, device=device)
    tokenizer = engine.tokenizer        # for decode_full_mesh chunked path
    disp_scale = engine.disp_scale
    velocity_input = engine.velocity_input
    print(f"  K={engine.K}, H={engine.H}, disp_scale={disp_scale:.4f}")
    print(f"  Stress head: {engine.has_stress_head}, "
          f"Gradient head: {engine.has_grad_head}, "
          f"n_input_frames={engine.n_input_frames}")
    print(f"  Dynamics: velocity_input={velocity_input}")
    if abs(args.action_scale - engine.action_scale) > 1e-9:
        sys.exit(f"--action-scale {args.action_scale} does not match the dynamics "
                 f"checkpoint's training value {engine.action_scale}; pass "
                 f"--action-scale {engine.action_scale} (or drop the flag).")

    plate_ids = discover_npz_plates(args.npz_dir, args.curated_plates)
    if args.plate_id:
        plate_ids = [args.plate_id]

    for pid in plate_ids:
        print(f"\n{'='*60}")
        print(f"Plate: {pid}")
        print(f"{'='*60}")

        pdata = load_npz_plate_data(
            args.npz_dir, args.fps_dir, args.mesh_dir,
            pid, args.max_trajs)
        coords = pdata['coords']
        topo = pdata['topology']
        outputs = pdata['outputs']
        fps_order = pdata['fps_order']
        inputs_all = pdata['inputs']

        n_traj_avail, n_frames, n_nodes, _ = outputs.shape
        n_inc = n_frames - 1

        # Coord normalization: MUST use the FULL-MESH bounding box so subset
        # and full-mesh queries share the same coordinate frame as during
        # training (MmapMultiStructureDataset normalizes full coords, then
        # indexes the FPS subset). Using the subset's own bbox pushes full-mesh
        # outer nodes outside [-1, 1], producing spike-like displacement
        # artifacts at boundary nodes and ruining F-D reaction forces.
        coords_f32 = coords.astype(np.float32)
        c_min_full = coords_f32.min(axis=0)
        c_max_full = coords_f32.max(axis=0)
        scale_full = np.maximum(c_max_full - c_min_full, 1e-8)

        # FPS subsampling — select rows from the full-mesh-normalized coords
        rng = np.random.default_rng(42)
        sub_idx = subsample_fps(fps_order, args.subsample_n, args.fps_core_n, rng)
        coords_sub = coords_f32[sub_idx]
        ref_sub_norm = ((coords_sub - c_min_full) / scale_full * 2 - 1).astype(np.float32)
        ref_sub_t = torch.tensor(ref_sub_norm, dtype=torch.float32,
                                 device=device).unsqueeze(0)

        ref_full_norm = ((coords_f32 - c_min_full) / scale_full * 2 - 1).astype(np.float32)
        ref_full_t = torch.tensor(ref_full_norm, dtype=torch.float32,
                                  device=device).unsqueeze(0)
        coord_scale_half = (scale_full / 2.0).astype(np.float32)

        # Stress geometry
        dNdx, volumes, good = precompute_stress_geometry(coords, topo)

        # Node-adjacency averaging operator for optional Laplacian smoothing
        # of decoded displacement (removes spike outliers at high-stress
        # junctions). Built once per plate; cheap sparse matvec per application.
        avg_op = None
        if args.smooth_iters > 0:
            avg_op = build_node_adjacency(topo, n_nodes)
            print(f"  Laplacian smoothing: {args.smooth_iters} iters, "
                  f"alpha={args.smooth_alpha}")

        # Boundary geometry for F-D
        right_node_ids, x_max = identify_right_boundary_nodes(coords, eps=1e-2)
        y_coords, z_coords = coords[:, 1], coords[:, 2]
        cy = (y_coords.min() + y_coords.max()) / 2.0
        cz = (z_coords.min() + z_coords.max()) / 2.0
        bdy_geom = precompute_boundary_geometry(topo, dNdx, volumes, right_node_ids)

        # Cumulative BCs
        cum_bc = compute_cumulative_bc(inputs_all)

        # Select trajectories
        if args.traj_indices:
            traj_ids = [int(x) for x in args.traj_indices.split(',')]
        else:
            traj_ids = np.linspace(0, n_traj_avail - 1, args.n_trajs,
                                   dtype=int).tolist()
        n_rows = len(traj_ids)
        print(f"  {n_nodes} nodes, {n_frames} frames, trajs: {traj_ids}")

        # Precompute all data
        print("  Precomputing GT + E2E (full mesh decode + F-D)...")
        all_traj_data = []
        t0 = time_mod.time()

        for ti in traj_ids:
            traj_data = {
                'gt_disps': [], 'gt_vm_nodes': [],
                'e2e_disps': [], 'e2e_vm_nodes': [],
                'gt_gen_force': np.zeros((n_frames, 4), dtype=np.float64),
                'gt_gen_disp': np.zeros((n_frames, 4), dtype=np.float64),
                'e2e_gen_force': np.zeros((n_frames, 4), dtype=np.float64),
                'e2e_gen_disp': np.zeros((n_frames, 4), dtype=np.float64),
            }

            # Generalized displacement (same for GT and Recon — from actions)
            traj_inputs = inputs_all[ti]  # (n_inc, 4)
            gen_disp = compute_gen_disp(traj_inputs, SPEEDS)  # (n_frames, 4)
            traj_data['gt_gen_disp'] = gen_disp
            traj_data['e2e_gen_disp'] = gen_disp  # same BC-driven displacement

            # Encode frame 0
            disp_0_sub = outputs[ti, 0, sub_idx]
            prevq = engine.encode_frame(
                disp_0_sub.astype(np.float32), ref_sub_norm)
            prevq_prev = prevq.clone() if velocity_input else None

            for fi in range(n_frames):
                # Teacher-forcing diagnostic: re-encode GT instead of using
                # dynamics-predicted prevq. Isolates tokenizer quality.
                if args.teacher_force:
                    disp_fi_sub = outputs[ti, fi, sub_idx]
                    prevq_te = engine.encode_frame(
                        disp_fi_sub.astype(np.float32), ref_sub_norm)
                else:
                    prevq_te = prevq
                # GT stress
                gt_d = outputs[ti, fi].astype(np.float64)
                _, _, _, gt_vm_node = stress_from_disps(
                    topo, gt_d, dNdx, volumes, good,
                    mu=args.stress_mu, lmbda=args.stress_lmbda)
                traj_data['gt_disps'].append(gt_d)
                traj_data['gt_vm_nodes'].append(gt_vm_node)

                # GT boundary reaction force
                if fi > 0:
                    traj_data['gt_gen_force'][fi] = boundary_reaction_force(
                        gt_d, coords, n_nodes, right_node_ids, cy, cz,
                        bdy_geom, mu=args.stress_mu, lmbda=args.stress_lmbda)

                # E2E decode + stress (use prevq_te: either dynamics-predicted
                # or GT-re-encoded depending on --teacher-force)
                e2e_d, sym6_head, vm_head, grad_u_phys = decode_full_mesh(
                    tokenizer, prevq_te, ref_full_t, disp_scale, device,
                    coord_scale=coord_scale_half)

                # Optional Laplacian smoothing of decoded displacement — only
                # affects mesh-deformation rendering. F-D force (integrated via
                # stress head Cauchy) uses the raw e2e_d_raw for F^{-T}, which
                # is insensitive to a few outliers, so --smooth-* never changes
                # the F-D curves or --save-fd-h5.
                e2e_d_raw = e2e_d
                if avg_op is not None:
                    e2e_d = laplacian_smooth(
                        e2e_d, avg_op, args.smooth_iters, args.smooth_alpha)

                if vm_head is not None:
                    # Stress-head path: VM computed directly from Cauchy sym6
                    e2e_vm_node = vm_head
                elif grad_u_phys is not None:
                    F_head = np.eye(3) + grad_u_phys
                    P_head = neo_hookean_P(F_head, mu=args.stress_mu,
                                           lmbda=args.stress_lmbda)
                    sigma_head = cauchy_from_P(P_head, F_head)
                    e2e_vm_node = von_mises(sigma_head).astype(np.float32)
                else:
                    _, _, _, e2e_vm_node = stress_from_disps(
                        topo, e2e_d, dNdx, volumes, good,
                        mu=args.stress_mu, lmbda=args.stress_lmbda)

                traj_data['e2e_disps'].append(e2e_d)
                traj_data['e2e_vm_nodes'].append(e2e_vm_node)

                # E2E boundary reaction force
                if fi > 0:
                    if sym6_head is not None:
                        # Stress-head path: use trained Cauchy sym6 directly.
                        # Tet-integrated stress from decoded u gives garbage
                        # forces (sign flips, 30× magnitude errors) because
                        # L2-trained u has noisy spatial gradients.
                        traj_data['e2e_gen_force'][fi] = \
                            boundary_force_from_stress_head(
                                sym6_head, e2e_d_raw, coords, n_nodes,
                                right_node_ids, cy, cz, bdy_geom)
                    elif grad_u_phys is not None:
                        traj_data['e2e_gen_force'][fi] = \
                            boundary_force_from_grad_head(
                                grad_u_phys, coords, n_nodes,
                                right_node_ids, cy, cz, bdy_geom,
                                mu=args.stress_mu, lmbda=args.stress_lmbda)
                    else:
                        traj_data['e2e_gen_force'][fi] = \
                            boundary_reaction_force(
                                e2e_d_raw, coords, n_nodes, right_node_ids,
                                cy, cz, bdy_geom,
                                mu=args.stress_mu, lmbda=args.stress_lmbda)

                # Dynamics step. velocity_input requires prev_state so the
                # model sees latent velocity z_t - z_{t-1}; without it the
                # velocity collapses to `state` and rollout diverges.
                if fi < n_inc:
                    bc = torch.tensor(
                        cum_bc[ti, fi] * args.action_scale,
                        dtype=torch.float32, device=device).unsqueeze(0)
                    prev_state_arg = (prevq_prev.unsqueeze(0)
                                      if velocity_input else None)
                    pred, _ = engine.step_dynamics(
                        prevq.unsqueeze(0), bc, prev_state=prev_state_arg)
                    if velocity_input:
                        prevq_prev = prevq
                    prevq = pred.squeeze(0)

            # Shared vmax from GT
            all_gt = np.concatenate(traj_data['gt_vm_nodes'])
            pos = all_gt[all_gt > 0]
            traj_data['vmax'] = float(np.percentile(pos, 99)) if len(pos) > 0 else 1.0
            all_traj_data.append(traj_data)
            elapsed = time_mod.time() - t0
            print(f"  Traj {ti}: vmax={traj_data['vmax']:.3f} [{elapsed:.0f}s]")

        # Camera from undeformed mesh
        grid = pv.UnstructuredGrid({pv.CellType.TETRA: topo}, coords)
        p = pv.Plotter(off_screen=True, window_size=(args.panel_size, args.panel_size))
        p.add_mesh(grid)
        p.view_isometric()
        p.camera.zoom(0.8)
        cam = p.camera_position
        p.close()

        # Render video: 3 columns x n_rows
        ps = args.panel_size
        n_cols = 3
        frame_w = ((n_cols * ps + 15) // 16) * 16
        frame_h = ((n_rows * ps + 15) // 16) * 16

        short_pid = pid.replace('desc_', '')
        ckpt_name = os.path.basename(os.path.dirname(args.dynamics))
        out_path = os.path.join(args.output_dir,
                                f"{ts}_{ckpt_name}_{short_pid}_dashboard.mp4")
        writer = imageio.get_writer(out_path, fps=args.fps, codec='libx264',
                                    quality=8, macro_block_size=16)

        print(f"  Rendering {n_rows}x{n_cols} dashboard, {n_frames} frames...")
        t_render = time_mod.time()

        for fi in range(n_frames):
            row_imgs = []
            for td in all_traj_data:
                vmax = td['vmax']

                # Column 1: GT stress panel
                gt_img = render_stress_panel(
                    coords, topo, td['gt_disps'][fi],
                    td['gt_vm_nodes'][fi], vmax, cam, ps, label='GT')

                # Column 2: Recon stress panel
                recon_img = render_stress_panel(
                    coords, topo, td['e2e_disps'][fi],
                    td['e2e_vm_nodes'][fi], vmax, cam, ps, label='Recon')

                # Column 3: 2x2 F-D plots
                fd_img = render_fd_panel(
                    td['gt_gen_force'], td['gt_gen_disp'],
                    td['e2e_gen_force'], td['e2e_gen_disp'],
                    fi, ps)

                row_img = np.concatenate([gt_img, recon_img, fd_img], axis=1)
                row_imgs.append(row_img)

            frame = np.concatenate(row_imgs, axis=0)

            # Pad to macroblock alignment
            if frame.shape[1] < frame_w or frame.shape[0] < frame_h:
                padded = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
                padded[:frame.shape[0], :frame.shape[1]] = frame
                frame = padded

            writer.append_data(frame)
            if (fi + 1) % 5 == 0 or fi == 0:
                print(f"    Frame {fi+1}/{n_frames} [{time_mod.time()-t_render:.0f}s]")

        writer.close()
        render_time = time_mod.time() - t_render
        total_time = time_mod.time() - t0
        print(f"  Saved: {out_path} ({render_time:.0f}s render, {total_time:.0f}s total)")

        if args.save_fd_h5:
            fd_path = out_path.replace('.mp4', '_fd.h5')
            with h5py.File(fd_path, 'w') as hf:
                hf.attrs['plate_id'] = pid
                hf.attrs['tokenizer'] = str(args.tokenizer)
                hf.attrs['dynamics'] = str(args.dynamics)
                hf.attrs['teacher_force'] = bool(args.teacher_force)
                for ti, td in enumerate(all_traj_data):
                    g = hf.create_group(f'traj_{ti}')
                    g.create_dataset('gt_gen_force', data=td['gt_gen_force'])
                    g.create_dataset('gt_gen_disp',  data=td['gt_gen_disp'])
                    g.create_dataset('e2e_gen_force', data=td['e2e_gen_force'])
                    g.create_dataset('e2e_gen_disp',  data=td['e2e_gen_disp'])
            print(f"  Saved F-D: {fd_path}")

    print(f"\n{'='*60}\nDone!\n{'='*60}")


if __name__ == '__main__':
    main()
