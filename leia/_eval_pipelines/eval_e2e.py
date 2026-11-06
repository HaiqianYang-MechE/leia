#!/usr/bin/env python
"""
End-to-end evaluation of the whole-plate dynamics pipeline.

Rollout: encode frame 0 → dynamics rollout with cumulative BCs → decode →
compute von Mises stress correlation vs GT.

Usage:
    python scripts/eval.py e2e \\
        --tokenizer checkpoints/tokenizer/production_v2/best.pt \\
        --dynamics  checkpoints/dynamics/production_v2/best_ar.pt \\
        --npz-dir   data/microplate_lattice/npz \\
        --fps-dir   data/microplate_lattice/fps \\
        --grad-u-dir data/microplate_lattice/grad_u \\
        --split val --full-mesh --compute-fd

`--split val` resolves the plate set from MICROPLATE_LATTICE.splits.
"""

import os
import argparse
import glob
import time
import json
from datetime import datetime

import numpy as np
import torch

from leia.engine import LeiaEngine
from leia.regime import MICROPLATE_LATTICE
from leia.tokenizer import decode_full_mesh, subsample_fps
from leia.stress import (
    precompute_stress_geometry, stress_from_disps,
    von_mises as vm_from_sigma,
    neohookean_cauchy_sym6_from_gradu,
    boundary_force_from_stress_head,
    boundary_force_from_grad_head,
    boundary_reaction_force,
    precompute_boundary_geometry,
    identify_right_boundary_nodes,
    compute_cumulative_bc,
)
from leia.metrics import pearson_corr, vm_from_sym6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--dynamics', required=True)
    parser.add_argument('--npz-dir', type=str, required=True,
                        help='Directory with {plate_id}/traj_*.npz files')
    parser.add_argument('--fps-dir', type=str, required=True,
                        help='Directory with {plate_id}_fps_order.npy files')
    parser.add_argument('--curated-plates', type=str, default=None,
                        help='JSON file with list of plate IDs (defaults to '
                             'all subdirectories of --npz-dir)')
    parser.add_argument('--grad-u-dir', type=str, default=None,
                        help='Directory with {plate_id}_grad_u.npy files for GT stress')
    parser.add_argument('--max-trajs', type=int, default=5,
                        help='Max trajectories per plate')
    # Common args
    parser.add_argument('--subsample-n', type=int, default=10240,
                        help='Total subsampled nodes per frame (paper: 8192 FPS core + 2048 random)')
    parser.add_argument('--fps-core-n', type=int, default=8192,
                        help='FPS core set size, deterministic (paper: 8192)')
    parser.add_argument('--action-scale', type=float, default=1.0,
                        help='Must match dynamics training')
    parser.add_argument('--stress-mu', type=float, default=1.0)
    parser.add_argument('--stress-lmbda', type=float, default=10.0)
    parser.add_argument('--n-traj', type=int, default=3,
                        help='Trajectories to evaluate per plate')
    parser.add_argument('--traj-indices', type=str, default=None,
                        help='Comma-separated trajectory indices (into the '
                             'sorted [0:max_trajs] slice). Overrides --n-traj. '
                             'For the in-dist val split (trajs beyond '
                             'training): set --max-trajs >= 101 and pass '
                             '"--traj-indices 100" to pick the first '
                             'out-of-training trajectory per plate.')
    parser.add_argument('--eval-frames', type=str, default='5,10,15,20,25,30',
                        help='Comma-separated frame indices to evaluate stress')
    parser.add_argument('--output-dir', type=str,
                        default='outputs/whole_plate_e2e')
    parser.add_argument('--out-json', type=str, default=None,
                        help='Override the auto-generated JSON filename '
                             '(default: {output_dir}/{ts}_e2e_results.json).')
    parser.add_argument('--mesh-dir', type=str, default=None,
                        help='Directory with {plate_id}/mesh.npz (topology) '
                             'for the --full-mesh, --compute-fd and --disp-vm-corr '
                             'paths (default: data/meshes/microplate_lattice).')
    # Eval-mode switches — all default to OFF, preserving original behavior.
    parser.add_argument('--teacher-force', action='store_true',
                        help='Re-encode GT displacement at every frame '
                             'instead of AR-rolling the dynamics model. '
                             'Isolates tokenizer reconstruction from dynamics drift.')
    parser.add_argument('--full-mesh', action='store_true',
                        help='Decode at all plate nodes (full mesh) instead '
                             'of the FPS-subsampled points. Required for '
                             'disp_vm_corr and F-D boundary integration.')
    parser.add_argument('--compute-fd', action='store_true',
                        help='Also integrate boundary reaction force per '
                             'frame and compare to GT F-D (Fx, Mx, Fy, Fz). '
                             'Requires --full-mesh + --mesh-dir.')
    parser.add_argument('--split', type=str, default=None,
                        choices=['train', 'val', 'ood'],
                        help='Sugar for split selection. train -> --traj-indices '
                             '"0,2,4" (sample of training trajectories); '
                             'val -> --traj-indices = MICROPLATE_LATTICE.splits.'
                             'training_cutoff_index (first post-training trajectory); '
                             'ood -> --traj-indices "0" on '
                             'MICROPLATE_LATTICE.splits.heldout_plates. '
                             'Explicit --traj-indices / --curated-plates still '
                             'override.')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--grad-head-convention', type=str, default='B',
                        choices=['A', 'B'],
                        help='coord_scale inversion convention for the gradient '
                             'head (only matters when the tokenizer has grad_proj). '
                             'B (default, j-axis) is the convention of the released '
                             'recipe; A (i-axis) is kept for older gradient-head '
                             'checkpoints. Mixing them scrambles the VM correlation '
                             '(anisotropic per-component, not just amplitude).')
    args = parser.parse_args()

    # --split sugar. train-set eval uses trajs 0, 2, 4 (a sample of training
    # trajectories). train/val default to the training plates; ood to heldout —
    # both sourced from MICROPLATE_LATTICE.splits. Explicit --curated-plates
    # still wins (e.g. for smoke tests on a 3-plate subset).
    args._curated_plates_list = None
    if args.split is not None:
        cutoff = MICROPLATE_LATTICE.splits.training_cutoff_index
        if args.split == 'train':
            if not args.traj_indices:
                args.traj_indices = '0,2,4'
            if not args.curated_plates:
                args._curated_plates_list = list(MICROPLATE_LATTICE.splits.training_plates)
        elif args.split == 'val':
            if not args.traj_indices:
                args.traj_indices = str(cutoff)
            if args.max_trajs < cutoff + 1:
                args.max_trajs = cutoff + 1
            if not args.curated_plates:
                args._curated_plates_list = list(MICROPLATE_LATTICE.splits.training_plates)
        elif args.split == 'ood':
            if not args.traj_indices:
                args.traj_indices = '0'
            if not args.curated_plates:
                args._curated_plates_list = list(MICROPLATE_LATTICE.splits.heldout_plates)

    if args.compute_fd and not args.full_mesh:
        parser.error("--compute-fd requires --full-mesh "
                     "(boundary force needs displacement at all boundary nodes).")
    if (args.full_mesh or args.compute_fd) and not args.mesh_dir:
        # Default to the production mesh dir; users can override.
        args.mesh_dir = 'data/meshes/microplate_lattice'

    return args


def _print_dynamics_metadata(dynamics_ckpt_path, device):
    """Print epoch + best metrics from the dynamics ckpt payload."""
    from leia.training import load_checkpoint
    _state, _opt, payload = load_checkpoint(dynamics_ckpt_path, map_location=device)
    epoch = payload.get('epoch', '?')
    best_val_loss = payload.get('best_val_loss', '?')
    best_ar_disp_l2 = payload.get('best_ar_disp_l2', '?')
    print(f"  Dynamics: epoch={epoch}, best_val_loss={best_val_loss}, "
          f"best_ar_disp_l2={best_ar_disp_l2}")






def load_plate_data_npz(npz_dir, fps_dir, plate_id, max_trajs):
    """Load plate data from NPZ files.

    Returns dict with keys: coords, outputs, fps_order, inputs, vm_gt_all.
    vm_gt_all is None (not available in NPZ format).
    """
    plate_dir = os.path.join(npz_dir, plate_id)
    traj_files = sorted(glob.glob(os.path.join(plate_dir, 'traj_*.npz')))
    if not traj_files:
        return None
    traj_files = traj_files[:max_trajs]

    # Load coords and fps_order from first trajectory / fps_dir
    first = np.load(traj_files[0])
    coords = first['coords'].astype(np.float32)
    fps_path = os.path.join(fps_dir, f'{plate_id}_fps_order.npy')
    if not os.path.exists(fps_path):
        raise FileNotFoundError(
            f"{fps_path} not found: the eval subsamples with the same FPS order the "
            f"tokenizer trained on. Run scripts/precompute_fps.py for {plate_id} "
            f"(README § 2 builds it for every plate).")
    fps_order = np.load(fps_path)

    # Stack outputs and inputs across trajectories
    outputs_list = []
    inputs_list = []
    for tf in traj_files:
        d = np.load(tf)
        outputs_list.append(d['outputs'].astype(np.float32))  # (n_frames, N, 3)
        inputs_list.append(d['inputs'].astype(np.float32))    # (n_inc, 4)
    outputs = np.stack(outputs_list, axis=0)  # (n_traj, n_frames, N, 3)
    inputs = np.stack(inputs_list, axis=0)    # (n_traj, n_inc, 4)

    return {
        'coords': coords,
        'outputs': outputs,
        'fps_order': fps_order,
        'inputs': inputs,
        'vm_gt_all': None,
    }


def encode_frame(engine: LeiaEngine, disp_sub, ref_norm,
                 disp_m1=None, disp_m2=None):
    """Encode one frame -> prevq embedding (K, H).

    Thin wrapper around `engine.encode_frame` (AR-init / single-frame) and
    `engine.encode_history` (TF case with n_input_frames > 1). Channel-order
    convention matches `MmapMultiStructureDataset.__getitem__`:
    [disp_{t-2}, disp_{t-1}, disp_t]. `disp_m1`/`disp_m2` default to the
    current frame to replay the legacy "pad with current frame" clamping.
    """
    n_input_frames = engine.n_input_frames
    if n_input_frames > 1:
        m1 = disp_m1 if disp_m1 is not None else disp_sub
        m2 = disp_m2 if disp_m2 is not None else disp_sub
        # Stack chronologically: [t-2, t-1, t]; engine takes the last
        # n_input_frames slice.
        history = np.stack([m2, m1, disp_sub], axis=0).astype(np.float32)
        return engine.encode_history(history, ref_norm)
    return engine.encode_frame(disp_sub.astype(np.float32), ref_norm)



def main():
    args = parse_args()
    device = torch.device(args.device)
    eval_frames = [int(x) for x in args.eval_frames.split(',')]

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load models via canonical engine API.
    print("Loading models...")
    engine = LeiaEngine(tokenizer_ckpt=args.tokenizer,
                        dynamics_ckpt=args.dynamics, device=device)
    tokenizer = engine.tokenizer       # for decode_full_mesh chunked path
    tok_cfg = engine.tok_cfg
    disp_scale = engine.disp_scale
    K, H = engine.K, engine.H
    velocity_input = engine.velocity_input
    if abs(args.action_scale - engine.action_scale) > 1e-9:
        raise SystemExit(
            f"--action-scale {args.action_scale} does not match the dynamics "
            f"checkpoint's training value {engine.action_scale}; pass "
            f"--action-scale {engine.action_scale} (or drop the flag).")
    print(f"  Tokenizer: K={K}, H={H}, disp_scale={disp_scale:.4f}, "
          f"stress_head={engine.has_stress_head}, "
          f"n_input_frames={engine.n_input_frames}")
    _print_dynamics_metadata(args.dynamics, device)
    print(f"  velocity_input={velocity_input}")

    # Discover plate IDs from curated list or subdirectories. Resolution
    # order: explicit --curated-plates path > sugar-resolved list (from
    # MICROPLATE_LATTICE.splits) > all subdirs of --npz-dir.
    if args.curated_plates is not None:
        with open(args.curated_plates, 'r') as f:
            plate_ids = sorted(json.load(f)['plates'])
    elif getattr(args, '_curated_plates_list', None):
        plate_ids = sorted(args._curated_plates_list)
    else:
        plate_ids = sorted([
            d for d in os.listdir(args.npz_dir)
            if os.path.isdir(os.path.join(args.npz_dir, d))
        ])
    print(f"\nPlates: {plate_ids}")

    all_results = {}
    # Collect scatter data: {pid: [(vm_gt, vm_pred, corr), ...]} — one entry per frame
    scatter_data = {}

    for pid in plate_ids:
        print(f"\n{'='*60}")
        print(f"Plate: {pid}")
        print(f"{'='*60}")

        pdata = load_plate_data_npz(
            args.npz_dir, args.fps_dir, pid, args.max_trajs)
        if pdata is None:
            print(f"  WARNING: no traj_*.npz found for {pid}, skipping")
            continue
        coords = pdata['coords']
        outputs = pdata['outputs']
        fps_order = pdata['fps_order']
        vm_gt_all = pdata['vm_gt_all']  # None for NPZ
        inputs = pdata['inputs']
        # Load grad_u mmap for GT stress (if available)
        grad_u_gt_mmap = None
        if args.grad_u_dir:
            gu_path = os.path.join(args.grad_u_dir, f'{pid}_grad_u.npy')
            if os.path.exists(gu_path):
                grad_u_gt_mmap = np.load(os.path.realpath(gu_path), mmap_mode='r')

        n_traj_avail, n_frames, n_nodes, _ = outputs.shape
        n_inc = n_frames - 1
        # Select which trajectories to evaluate. --traj-indices overrides
        # --n-traj; skip plates that can't provide the requested index (e.g.
        # plates with <101 trajectories when asking for index 100).
        if args.traj_indices:
            requested = [int(x) for x in args.traj_indices.split(',')]
            valid_traj_ids = [i for i in requested if i < n_traj_avail]
            if not valid_traj_ids:
                print(f"  Skipping {pid}: none of --traj-indices "
                      f"{requested} < n_traj_avail={n_traj_avail}")
                continue
            if len(valid_traj_ids) < len(requested):
                print(f"  {pid}: dropping indices "
                      f"{sorted(set(requested) - set(valid_traj_ids))} "
                      f"(>= n_traj_avail={n_traj_avail})")
            traj_ids = valid_traj_ids
            n_traj = len(traj_ids)
        else:
            n_traj = min(args.n_traj, n_traj_avail)
            traj_ids = list(range(n_traj))

        # Subsample (deterministic — same as tokenizer training)
        rng = np.random.default_rng(42)
        sub_idx = subsample_fps(fps_order, args.subsample_n, args.fps_core_n, rng)
        coords_sub = coords[sub_idx]

        # Normalize to [-1, 1] using FULL mesh range (matches tokenizer training)
        c_min = coords.min(axis=0)
        c_max = coords.max(axis=0)
        coord_range = np.maximum(c_max - c_min, 1e-8)
        coord_scale = (coord_range / 2.0).astype(np.float32)  # half range, matches training
        coords_norm = ((coords - c_min) / coord_range * 2 - 1).astype(np.float32)
        ref_norm = coords_norm[sub_idx]

        # Cumulative BCs
        cum_bc = compute_cumulative_bc(inputs)

        # Topology + full-mesh geometry (only built when needed). Used for
        # disp_vm_corr (FEM on decoded u) and F-D boundary integration.
        topo = None
        dNdx = volumes = good = None
        right_node_ids = bdy_geom = None
        ref_norm_full_t = None
        if args.full_mesh or args.compute_fd:
            mesh_path = os.path.join(args.mesh_dir, pid, 'mesh.npz')
            if not os.path.exists(mesh_path):
                print(f"  WARNING: mesh.npz missing for {pid}; "
                      f"skipping plate entirely (full-mesh/F-D requires topology).")
                continue
            else:
                with np.load(mesh_path) as m:
                    # mesh.npz keys observed across datasets: 'cells' (lattice
                    # production), 'topology' (some legacy), 'tets' (other
                    # legacy). All store (n_cells, 4) int tet connectivity.
                    if 'cells' in m:
                        topo = m['cells'].astype(np.int64)
                    elif 'topology' in m:
                        topo = m['topology'].astype(np.int64)
                    elif 'tets' in m:
                        topo = m['tets'].astype(np.int64)
                    else:
                        raise KeyError(
                            f"No topology array in {mesh_path}; "
                            f"expected one of {{cells, topology, tets}}, "
                            f"got {list(m.keys())}")
                dNdx, volumes, good = precompute_stress_geometry(
                    coords.astype(np.float64), topo)
                ref_norm_full_t = torch.tensor(
                    coords_norm, dtype=torch.float32, device=device).unsqueeze(0)
                if args.compute_fd:
                    right_node_ids, _ = identify_right_boundary_nodes(
                        coords.astype(np.float64), eps=1e-2)
                    bdy_geom = precompute_boundary_geometry(
                        topo, dNdx, volumes, right_node_ids)

        ref_norm_sub_t = torch.tensor(
            ref_norm, dtype=torch.float32, device=device).unsqueeze(0)
        n_nodes_full = coords.shape[0]

        mode_label = "TF" if args.teacher_force else "AR"
        mesh_label = "full-mesh" if args.full_mesh else "sub-mesh"
        print(f"  {n_traj} traj, {n_frames} frames, "
              f"{n_nodes_full if args.full_mesh else len(sub_idx)} eval-nodes "
              f"[{mode_label} / {mesh_label}{' / F-D' if args.compute_fd else ''}]")

        # Accumulators used for scatter visualization (kept for compatibility).
        plate_corrs = []
        plate_scatter = []
        # New per-plate metric accumulators.
        plate_disp_rel_l2 = []
        plate_head_vm = []
        plate_disp_vm = []
        plate_grad_head_vm = []
        plate_fd_overall = []
        plate_fd_per_channel = []

        for traj_idx in traj_ids:
            t0 = time.time()

            # --- Build the latent sequence used at each frame. ---
            # AR: encode frame 0, roll out dynamics model (velocity_input-aware).
            # TF: encode GT at each frame.
            prevq0 = encode_frame(engine, outputs[traj_idx, 0, sub_idx], ref_norm)

            if args.teacher_force:
                # Tokenizer was trained on (disp_{t-2}, disp_{t-1}, disp_t)
                # windows clamped to the current trajectory; frame 0 is rest.
                # Match that clamping here so TF inputs are in distribution.
                z_at_frame = {0: prevq0}
                for f in range(1, n_frames):
                    disp_f  = outputs[traj_idx, f,             sub_idx]
                    disp_m1 = outputs[traj_idx, max(f - 1, 0), sub_idx]
                    disp_m2 = outputs[traj_idx, max(f - 2, 0), sub_idx]
                    z_at_frame[f] = encode_frame(
                        engine, disp_f, ref_norm,
                        disp_m1=disp_m1, disp_m2=disp_m2)
            else:
                # Autoregressive rollout. velocity_input threads prev_state;
                # skipping it collapses velocity to state and diverges.
                z_at_frame = {0: prevq0.clone()}
                prevq = prevq0.clone()
                prevq_prev = prevq.clone() if velocity_input else None
                for t in range(n_inc):
                    bc = torch.tensor(cum_bc[traj_idx, t] * args.action_scale,
                                      dtype=torch.float32, device=device).unsqueeze(0)
                    prev_state_arg = (prevq_prev.unsqueeze(0)
                                      if velocity_input else None)
                    pred, _ = engine.step_dynamics(
                        prevq.unsqueeze(0), bc, prev_state=prev_state_arg)
                    if velocity_input:
                        prevq_prev = prevq
                    prevq = pred.squeeze(0)
                    z_at_frame[t + 1] = prevq.clone()

            # --- Iterate over frames.
            # F-D needs every frame (continuous curve); eval metrics only at
            # eval_frames. When F-D is off we skip non-eval frames entirely.
            compute_frames = (list(range(n_frames)) if args.compute_fd
                              else sorted(set(f for f in eval_frames if f <= n_inc)))

            fd_gt = np.zeros((n_frames, 4), dtype=np.float64)
            fd_pred = np.zeros((n_frames, 4), dtype=np.float64)

            traj_corrs = []
            traj_disp_rel_l2 = []
            traj_head_vm = []
            traj_disp_vm = []
            traj_grad_head_vm = []

            for frame in compute_frames:
                z_enc = z_at_frame[frame]

                # Decode at sub or full mesh, picking up stress-head sym6 if
                # present. Chunked to keep 300K-node plates within memory.
                ref_t = ref_norm_full_t if args.full_mesh else ref_norm_sub_t
                u_phys, sym6_node, vm_head, grad_u_phys = decode_full_mesh(
                    tokenizer, z_enc, ref_t, disp_scale, device,
                    chunk_size=50000, coord_scale=coord_scale,
                    grad_head_convention=args.grad_head_convention)

                # F-D contribution (full-mesh only).
                if args.compute_fd and topo is not None:
                    if frame > 0:
                        # GT F-D: use precomputed grad_u when available;
                        # fall back to stress_from_disps on GT displacement.
                        if grad_u_gt_mmap is not None:
                            flat_idx = traj_idx * n_frames + frame
                            gu_gt_full = np.array(grad_u_gt_mmap[flat_idx])
                            # boundary_force_from_grad_head expects (N, 3, 3)
                            fd_gt[frame] = boundary_force_from_grad_head(
                                gu_gt_full, coords.astype(np.float64),
                                n_nodes_full, right_node_ids,
                                (coords[:, 1].min() + coords[:, 1].max()) / 2.0,
                                (coords[:, 2].min() + coords[:, 2].max()) / 2.0,
                                bdy_geom, mu=args.stress_mu, lmbda=args.stress_lmbda)
                        else:
                            # FEM-from-GT-displacement path.
                            fd_gt[frame] = boundary_reaction_force(
                                outputs[traj_idx, frame].astype(np.float64),
                                coords.astype(np.float64), n_nodes_full,
                                right_node_ids,
                                (coords[:, 1].min() + coords[:, 1].max()) / 2.0,
                                (coords[:, 2].min() + coords[:, 2].max()) / 2.0,
                                bdy_geom, mu=args.stress_mu, lmbda=args.stress_lmbda)
                        # Pred F-D: stress-head path when sym6 present, else
                        # gradient-head, else FEM-from-decoded-u (fallback).
                        u_phys_f64 = u_phys.astype(np.float64)
                        cy = (coords[:, 1].min() + coords[:, 1].max()) / 2.0
                        cz = (coords[:, 2].min() + coords[:, 2].max()) / 2.0
                        if sym6_node is not None:
                            fd_pred[frame] = boundary_force_from_stress_head(
                                sym6_node, u_phys_f64,
                                coords.astype(np.float64), n_nodes_full,
                                right_node_ids, cy, cz, bdy_geom)
                        elif grad_u_phys is not None:
                            fd_pred[frame] = boundary_force_from_grad_head(
                                grad_u_phys, coords.astype(np.float64),
                                n_nodes_full, right_node_ids, cy, cz,
                                bdy_geom, mu=args.stress_mu, lmbda=args.stress_lmbda)
                        else:
                            fd_pred[frame] = boundary_reaction_force(
                                u_phys_f64, coords.astype(np.float64),
                                n_nodes_full, right_node_ids, cy, cz, bdy_geom,
                                mu=args.stress_mu, lmbda=args.stress_lmbda)

                # Eval-frame metrics (disp_rel_l2, head_vm, disp_vm).
                if frame not in eval_frames:
                    continue

                query_idx = slice(None) if args.full_mesh else sub_idx
                u_gt = outputs[traj_idx, frame, query_idx].astype(np.float64)
                u_pred = u_phys[query_idx] if args.full_mesh else u_phys

                # Displacement relative L2.
                num = float(np.linalg.norm(u_pred - u_gt))
                den = float(np.linalg.norm(u_gt) + 1e-30)
                traj_disp_rel_l2.append(num / den)

                # GT von Mises at the queried points.
                vm_gt = None
                if vm_gt_all is not None:
                    vm_gt = vm_gt_all[traj_idx, frame, query_idx].astype(np.float64)
                elif grad_u_gt_mmap is not None:
                    flat_idx = traj_idx * n_frames + frame
                    gu_gt = np.array(grad_u_gt_mmap[flat_idx])
                    # Compute per-node GT VM from grad_u (CPU float64, matches
                    # leia.stress.von_mises conventions used by stress_from_disps).
                    gu_gt_f = gu_gt.astype(np.float64)
                    I3 = np.eye(3, dtype=np.float64)
                    F_gt = I3 + gu_gt_f
                    J = np.linalg.det(F_gt)
                    Js = np.maximum(J, 1e-30)
                    b = np.einsum('...ij,...kj->...ik', F_gt, F_gt)
                    sigma = (args.stress_mu / Js)[..., None, None] * (b - I3) \
                        + (args.stress_lmbda / Js * np.log(Js))[..., None, None] * I3
                    vm_full = vm_from_sigma(sigma)
                    vm_gt = vm_full[query_idx]
                elif topo is not None:
                    # No grad_u on disk — compute GT VM from GT displacement via FEM.
                    _, _, _, vm_full = stress_from_disps(
                        topo, outputs[traj_idx, frame].astype(np.float64),
                        dNdx, volumes, good,
                        mu=args.stress_mu, lmbda=args.stress_lmbda)
                    vm_gt = vm_full[query_idx]

                # head_vm_corr (stress-head path). vm_head lives at all decoded
                # points; same query_idx applies when full-mesh, no slicing in sub.
                if vm_head is not None and vm_gt is not None:
                    vm_head_query = vm_head[query_idx] if args.full_mesh else vm_head
                    hv = pearson_corr(vm_head_query, vm_gt)
                    traj_head_vm.append(hv)
                else:
                    hv = float('nan')

                # disp_vm_corr via FEM on decoded-u. Needs topology; ONLY
                # meaningful on full mesh (we need tet connectivity).
                if args.full_mesh and topo is not None and vm_gt is not None:
                    _, _, _, vm_disp_full = stress_from_disps(
                        topo, u_phys.astype(np.float64),
                        dNdx, volumes, good,
                        mu=args.stress_mu, lmbda=args.stress_lmbda)
                    dv = pearson_corr(vm_disp_full, vm_gt)
                    traj_disp_vm.append(dv)
                else:
                    dv = float('nan')

                # grad_head_vm_corr (gradient-head path). grad_u_phys is at the
                # same resolution as u_phys (full or sub). Closed-form Neo-Hookean
                # Cauchy → VM; no topology needed, so this works on sub mesh too.
                if grad_u_phys is not None and vm_gt is not None:
                    sym6 = neohookean_cauchy_sym6_from_gradu(
                        grad_u_phys, mu=args.stress_mu, lmbda=args.stress_lmbda)
                    vm_grad_full = vm_from_sym6(sym6)
                    vm_grad_query = vm_grad_full[query_idx] if args.full_mesh \
                        else vm_grad_full
                    gv = pearson_corr(vm_grad_query, vm_gt)
                    traj_grad_head_vm.append(gv)
                else:
                    gv = float('nan')

                # Back-compat: traj_corrs prefers stress-head VM, falls back to
                # gradient-head VM, then disp-VM (FEM on decoded u).
                corr = hv if not np.isnan(hv) else (
                    gv if not np.isnan(gv) else dv)
                traj_corrs.append(corr)

                # Scatter data — keep one frame per traj (last eval frame).
                if vm_gt is not None and len(plate_scatter) < n_traj \
                        and frame == eval_frames[-1] and not args.full_mesh:
                    vm_pred_scatter = vm_head if vm_head is not None else np.zeros_like(vm_gt)
                    plate_scatter.append(
                        (vm_gt.copy(), np.asarray(vm_pred_scatter).copy(), corr))

            # Per-traj F-D summary (overall scaled rel_L2 + per-channel).
            fd_rel_l2_overall = float('nan')
            fd_rel_l2_per_channel = [float('nan')] * 4
            if args.compute_fd:
                gt_tail = fd_gt[1:n_frames]
                pr_tail = fd_pred[1:n_frames]
                scales = np.abs(gt_tail).max(axis=0) + 1e-12
                ref_scaled = gt_tail / scales
                err_scaled = (pr_tail - gt_tail) / scales
                ref_norm_val = float(np.linalg.norm(ref_scaled))
                if ref_norm_val > 1e-12:
                    fd_rel_l2_overall = float(
                        np.linalg.norm(err_scaled) / ref_norm_val)
                for c in range(4):
                    num = float(np.linalg.norm(pr_tail[:, c] - gt_tail[:, c]))
                    den = float(np.linalg.norm(gt_tail[:, c]) + 1e-12)
                    fd_rel_l2_per_channel[c] = (
                        num / den if den > 1e-12 else float('nan'))
                plate_fd_overall.append(fd_rel_l2_overall)
                plate_fd_per_channel.append(fd_rel_l2_per_channel)

            dt = time.time() - t0
            plate_corrs.extend(traj_corrs)
            plate_disp_rel_l2.extend(traj_disp_rel_l2)
            plate_head_vm.extend(traj_head_vm)
            plate_disp_vm.extend(traj_disp_vm)
            plate_grad_head_vm.extend(traj_grad_head_vm)

            def _fmt(x): return f"{x:.4f}" if not np.isnan(x) else "N/A"
            hv_med = float(np.median([c for c in traj_head_vm
                                      if not np.isnan(c)])) if traj_head_vm else float('nan')
            dv_med = float(np.median([c for c in traj_disp_vm
                                      if not np.isnan(c)])) if traj_disp_vm else float('nan')
            gv_med = float(np.median([c for c in traj_grad_head_vm
                                      if not np.isnan(c)])) if traj_grad_head_vm else float('nan')
            d_med = float(np.median(traj_disp_rel_l2)) if traj_disp_rel_l2 else float('nan')
            extra = ""
            if args.compute_fd:
                extra = f" | fd={_fmt(fd_rel_l2_overall)}"
            print(f"  Traj {traj_idx}: disp_rel_l2={_fmt(d_med)} "
                  f"head_vm={_fmt(hv_med)} grad_head_vm={_fmt(gv_med)} "
                  f"disp_vm={_fmt(dv_med)}{extra} "
                  f"({len(traj_disp_rel_l2)} eval frames, {dt:.1f}s)")

        def _med(xs):
            xs = [x for x in xs if not np.isnan(x)]
            return float(np.median(xs)) if xs else float('nan')

        plate_head_vm_med = _med(plate_head_vm)
        plate_disp_vm_med = _med(plate_disp_vm)
        plate_grad_head_vm_med = _med(plate_grad_head_vm)
        plate_disp_rel_l2_med = _med(plate_disp_rel_l2)
        plate_fd_overall_med = _med(plate_fd_overall)
        plate_fd_per_channel_med = [
            _med([row[c] for row in plate_fd_per_channel])
            for c in range(4)
        ] if plate_fd_per_channel else [float('nan')] * 4
        # Back-compat summary: median_vm_corr prefers stress-head VM, falls
        # back to gradient-head VM, then disp-VM (FEM on decoded u). This
        # matches the per-frame priority used for `traj_corrs`.
        if not np.isnan(plate_head_vm_med):
            plate_median = plate_head_vm_med
        elif not np.isnan(plate_grad_head_vm_med):
            plate_median = plate_grad_head_vm_med
        else:
            plate_median = plate_disp_vm_med
        all_results[pid] = {
            'median_vm_corr': plate_median,  # legacy key: row's native VM
            'head_vm_corr_median': plate_head_vm_med,
            'grad_head_vm_corr_median': plate_grad_head_vm_med,
            'disp_vm_corr_median': plate_disp_vm_med,
            'disp_rel_l2_median': plate_disp_rel_l2_med,
            'fd_rel_l2_overall_median': plate_fd_overall_med,
            'fd_rel_l2_per_channel_median': plate_fd_per_channel_med,
            'n_frames_eval': len(plate_disp_rel_l2),
            'all_corrs': [float(c) for c in plate_corrs],
            'mode': 'teacher_force' if args.teacher_force else 'ar',
            'mesh': 'full' if args.full_mesh else 'sub',
        }
        print(f"  => {pid}: disp_rel_l2={_fmt(plate_disp_rel_l2_med)} "
              f"head_vm={_fmt(plate_head_vm_med)} "
              f"grad_head_vm={_fmt(plate_grad_head_vm_med)} "
              f"disp_vm={_fmt(plate_disp_vm_med)}"
              + (f" fd={_fmt(plate_fd_overall_med)}"
                 if args.compute_fd else ""))
        scatter_data[pid] = plate_scatter

    # Overall summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    def _med_across(key):
        xs = [all_results[p][key] for p in all_results
              if not np.isnan(all_results[p].get(key, float('nan')))]
        return float(np.median(xs)) if xs else float('nan')

    def _fmt(x): return f"{x:.4f}" if not np.isnan(x) else "N/A"

    overall_head_vm = _med_across('head_vm_corr_median')
    overall_grad_head_vm = _med_across('grad_head_vm_corr_median')
    overall_disp_vm = _med_across('disp_vm_corr_median')
    overall_disp_rel_l2 = _med_across('disp_rel_l2_median')
    overall_fd = _med_across('fd_rel_l2_overall_median')
    # Legacy alias — row's native VM (stress > grad > disp).
    if not np.isnan(overall_head_vm):
        overall = overall_head_vm
    elif not np.isnan(overall_grad_head_vm):
        overall = overall_grad_head_vm
    else:
        overall = overall_disp_vm

    for pid, res in sorted(all_results.items()):
        print(f"  {pid}: disp_rel_l2={_fmt(res['disp_rel_l2_median'])} "
              f"head_vm={_fmt(res['head_vm_corr_median'])} "
              f"grad_head_vm={_fmt(res.get('grad_head_vm_corr_median', float('nan')))} "
              f"disp_vm={_fmt(res['disp_vm_corr_median'])}"
              + (f" fd={_fmt(res['fd_rel_l2_overall_median'])}"
                 if args.compute_fd else ""))

    print(f"\n  Overall medians:"
          f" disp_rel_l2={_fmt(overall_disp_rel_l2)}"
          f"  head_vm={_fmt(overall_head_vm)}"
          f"  grad_head_vm={_fmt(overall_grad_head_vm)}"
          f"  disp_vm={_fmt(overall_disp_vm)}"
          + (f"  F-D rel_L2={_fmt(overall_fd)}" if args.compute_fd else ""))

    # Save results
    results = {
        'timestamp': ts,
        'tokenizer': args.tokenizer,
        'dynamics': args.dynamics,
        'mode': 'teacher_force' if args.teacher_force else 'ar',
        'mesh': 'full' if args.full_mesh else 'sub',
        'split': args.split,
        'curated_plates': args.curated_plates,
        'traj_indices': args.traj_indices,
        'eval_frames': eval_frames,
        'subsample_n': args.subsample_n,
        'fps_core_n': args.fps_core_n,
        'stress_mu': args.stress_mu,
        'stress_lmbda': args.stress_lmbda,
        'n_traj_per_plate': args.n_traj,
        'action_scale': args.action_scale,
        'overall': {
            'disp_rel_l2_median': overall_disp_rel_l2,
            'head_vm_corr_median': overall_head_vm,
            'grad_head_vm_corr_median': overall_grad_head_vm,
            'disp_vm_corr_median': overall_disp_vm,
            'fd_rel_l2_overall_median': overall_fd,
        },
        'overall_median_vm_corr': overall,  # legacy
        'per_plate': all_results,
    }
    if args.out_json:
        out_path = args.out_json
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    else:
        out_path = os.path.join(args.output_dir, f'{ts}_e2e_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # --- Scatter and histogram plots ---
    plates_with_scatter = [pid for pid in sorted(scatter_data) if scatter_data[pid]]
    if plates_with_scatter:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Scatter: 2x5 grid of per-plate GT vs pred VM
        n_scatter = min(10, len(plates_with_scatter))
        fig, axes = plt.subplots(2, 5, figsize=(20, 8), squeeze=False)
        for si in range(n_scatter):
            pid = plates_with_scatter[si]
            ax = axes[si // 5][si % 5]
            # Concatenate all scatter entries for this plate
            all_gt = np.concatenate([s[0] for s in scatter_data[pid]])
            all_pred = np.concatenate([s[1] for s in scatter_data[pid]])
            vmax = np.percentile(np.concatenate([all_gt, all_pred]), 99)
            if vmax < 1e-10:
                vmax = 1.0
            ax.scatter(all_gt, all_pred, s=0.3, alpha=0.3, c='steelblue', rasterized=True)
            ax.plot([0, vmax], [0, vmax], 'r--', lw=1)
            ax.set_xlim(0, vmax)
            ax.set_ylim(0, vmax)
            ax.set_aspect('equal')
            r = pearson_corr(all_gt, all_pred)
            med = all_results[pid]['median_vm_corr']
            ax.set_title(f'{pid}\nr={r:.3f} (med={med:.3f})', fontsize=9)
            ax.set_xlabel('GT VM', fontsize=7)
            ax.set_ylabel('E2E Pred VM', fontsize=7)
            ax.tick_params(labelsize=6)
        for si in range(n_scatter, 10):
            axes[si // 5][si % 5].set_visible(False)
        fig.suptitle(f'E2E von Mises: GT vs Predicted (overall median={overall:.4f})',
                     fontsize=12)
        fig.tight_layout()
        scatter_path = os.path.join(args.output_dir, f'{ts}_e2e_vm_scatter.png')
        fig.savefig(scatter_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Scatter: {scatter_path}")

        # Histogram of per-plate median vm_corr
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: histogram of per-frame vm_corr (pooled across all plates)
        ax = axes[0]
        pooled_corrs = []
        for pid in all_results:
            pooled_corrs.extend(all_results[pid].get('all_corrs', []))
        valid_corrs = [c for c in pooled_corrs if not np.isnan(c)]
        ax.hist(valid_corrs, bins=40, color='steelblue', alpha=0.7, edgecolor='white')
        ax.axvline(overall, color='red', ls='--', lw=1.5,
                   label=f'median={overall:.3f}')
        ax.set_xlabel('Per-frame von Mises correlation')
        ax.set_ylabel('Count')
        ax.set_title(f'E2E VM correlation distribution ({len(valid_corrs)} frames)')
        ax.legend()

        # Right: per-plate median bar chart
        ax = axes[1]
        sorted_plates = sorted(all_results.keys(),
                               key=lambda p: all_results[p]['median_vm_corr'])
        medians = [all_results[p]['median_vm_corr'] for p in sorted_plates]
        colors = ['#d32f2f' if m < 0.7 else '#ff9800' if m < 0.85
                  else '#4caf50' for m in medians]
        ax.barh(range(len(sorted_plates)), medians, color=colors, height=0.7)
        ax.set_yticks(range(len(sorted_plates)))
        ax.set_yticklabels(sorted_plates, fontsize=6)
        ax.axvline(overall, color='red', ls='--', lw=1)
        ax.set_xlabel('Median von Mises correlation')
        ax.set_title(f'Per-plate E2E VM correlation (n={len(sorted_plates)})')
        ax.set_xlim(0, 1)
        fig.tight_layout()
        hist_path = os.path.join(args.output_dir, f'{ts}_e2e_vm_histogram.png')
        fig.savefig(hist_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Histogram: {hist_path}")


if __name__ == '__main__':
    main()
