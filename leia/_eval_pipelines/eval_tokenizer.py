#!/usr/bin/env python
"""
Evaluate the MicroPlate lattice tokenizer's stress reconstruction at subsampled points.

Supports two stress computation modes:
  1. Stress head (stress_proj) — Cauchy sym6 directly, universal
  2. Gradient head (grad_proj) → autograd Jacobian fallback when no stress head

Reads from the canonical mmap directory (the same format used by tokenizer
training): {plate_id}_{coords,outputs,fps_order,grad_u}.npy.

Reports per-frame Pearson correlation (median across frames, per-plate and overall).

Usage:
    python scripts/eval.py tokenizer \\
        --checkpoint checkpoints/tokenizer/production_v2/best.pt \\
        --mmap-dir data/microplate_lattice/plate_dataset_curated_mmap_100k/ \\
        --subsample-n 10240 --fps-core-n 8192 --n-frames 100
"""

import os
import time
import argparse
import json
from datetime import datetime

import numpy as np
import torch

from leia.engine import LeiaEngine
from leia.tokenizer import subsample_fps
from leia.stress import neohookean_cauchy_sym6_from_gradu
from leia.metrics import pearson_corr, vm_from_sym6

# `num_decoder_layers` fallback for older ckpts that omit it from cfg
# (used in autograd-Jacobian fallback warning).
TOKENIZER_NUM_DECODER_LAYERS = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--mmap-dir', type=str, required=True,
                        help='Mmap directory with {plate_id}_{coords,outputs,fps_order,grad_u}.npy')
    parser.add_argument('--subsample-n', type=int, default=10240,
                        help='Total subsampled nodes per frame (paper: 8192 FPS core + 2048 random)')
    parser.add_argument('--fps-core-n', type=int, default=8192,
                        help='FPS core set size, deterministic (paper: 8192)')
    parser.add_argument('--n-frames', type=int, default=0,
                        help='Frames to evaluate per plate (0=all)')
    parser.add_argument('--stress-mu', type=float, default=1.0,
                        help='Neo-Hookean mu')
    parser.add_argument('--stress-lmbda', type=float, default=10.0,
                        help='Neo-Hookean lambda')
    parser.add_argument('--output-dir', type=str,
                        default='outputs/lattice_stress_eval')
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


def _grad_u_to_vm(grad_u_phys, mu, lmbda, device=None):
    """grad_u (torch [1, N, 3, 3] or numpy [..., 3, 3]) → von Mises (numpy [N]).

    Wraps `leia.stress.neohookean_cauchy_sym6_from_gradu` + `leia.metrics.vm_from_sym6`.
    `device` is accepted for backward-compatible call sites and ignored.
    """
    if hasattr(grad_u_phys, 'detach'):
        grad_u_np = grad_u_phys.detach().cpu().numpy()
    else:
        grad_u_np = np.asarray(grad_u_phys)
    if grad_u_np.ndim == 4 and grad_u_np.shape[0] == 1:
        grad_u_np = grad_u_np[0]  # squeeze leading batch
    sym6 = neohookean_cauchy_sym6_from_gradu(grad_u_np, mu=mu, lmbda=lmbda)
    return vm_from_sym6(sym6).astype(np.float32)


def compute_predicted_vm(engine: LeiaEngine, ref_norm, x_sub, coord_scale,
                         mu, lmbda, device, *,
                         disp_m1=None, disp_m2=None):
    """Encode → decode → stress → von Mises at subsampled points.

    Head priority: stress head (universal) > grad head (Neo-Hookean only) >
    autograd Jacobian (fallback). Engine handles the head dispatch in
    `decode`; the autograd fallback fires only when the loaded tokenizer
    has neither head (legacy continuous-bottleneck checkpoints).

    `disp_m1`/`disp_m2` carry the previous two frames when the tokenizer
    was trained with `n_input_frames=3`; they default to the current frame
    (matches the legacy "pad with current frame" behaviour).
    """
    n_input_frames = engine.n_input_frames
    # Build history input as physical-space displacement; engine normalizes
    # by disp_scale internally.
    if n_input_frames == 3:
        m1 = disp_m1 if disp_m1 is not None else x_sub
        m2 = disp_m2 if disp_m2 is not None else x_sub
        # Stack chronologically: oldest → newest (m2, m1, current).
        disp_history = np.stack([m2, m1, x_sub], axis=0).astype(np.float32)
    else:
        disp_history = x_sub[None].astype(np.float32)

    if engine.has_stress_head or engine.has_grad_head:
        z = engine.encode_history(disp_history, ref_norm)
        ref_t = torch.tensor(ref_norm, dtype=torch.float32,
                             device=device).unsqueeze(0)
        out = engine.decode(z[None], ref_t, compute_stress=True)
        u_recon_phys = out["disp"].squeeze(0).cpu().numpy()
        if engine.has_stress_head:
            sym6 = out["cauchy_sym6"].squeeze(0).cpu().numpy()
            pred_vm = vm_from_sym6(sym6).astype(np.float32)
            return pred_vm, u_recon_phys
        # grad-head: convert raw normalized grad_u → physical via coord_scale.
        cs = torch.tensor(coord_scale, dtype=torch.float32, device=device)
        grad_pred = out["grad_u"].reshape(1, -1, 3, 3)
        scale = (engine.disp_scale / cs).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        grad_u_phys = grad_pred * scale
        pred_vm = _grad_u_to_vm(grad_u_phys, mu, lmbda, device)
        return pred_vm, u_recon_phys

    # Autograd Jacobian fallback — for legacy continuous-bottleneck ckpts
    # with neither head. Reaches into tokenizer internals; rare path.
    tokenizer = engine.tokenizer
    disp_scale = engine.disp_scale
    x_now = (x_sub / disp_scale).astype(np.float32)
    if n_input_frames == 3:
        m1n = (disp_m1 / disp_scale).astype(np.float32) if disp_m1 is not None else x_now
        m2n = (disp_m2 / disp_scale).astype(np.float32) if disp_m2 is not None else x_now
        x_in = np.concatenate([m2n, m1n, x_now], axis=-1)
    else:
        x_in = x_now
    x_t = torch.tensor(x_in, dtype=torch.float32, device=device).unsqueeze(0)
    ref_t = torch.tensor(ref_norm, dtype=torch.float32, device=device).unsqueeze(0)
    cs = torch.tensor(coord_scale, dtype=torch.float32, device=device)

    with torch.no_grad():
        z_enc = tokenizer.encode(x_t, ref_t)
    ref_g = ref_t.detach().requires_grad_(True)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        queries = tokenizer.pos_encoding(ref_g)
        features = tokenizer.cross_attn_decoder(query=queries, key=z_enc)
        if tokenizer.decoder_self_attn is not None:
            features = tokenizer.decoder_self_attn(features)
        u_recon = tokenizer.final_proj(tokenizer.final_norm(features))
    grad_u = []
    for i in range(3):
        g = torch.autograd.grad(u_recon[..., i].sum(), ref_g,
                                create_graph=False, retain_graph=True)[0]
        grad_u.append(g)
    grad_u = torch.stack(grad_u, dim=-2)
    scale_factor = disp_scale / cs
    grad_u_phys = grad_u * scale_factor[None, None, None, :]
    pred_vm = _grad_u_to_vm(grad_u_phys, mu, lmbda, device)
    u_recon_phys = (u_recon.detach() * disp_scale).squeeze(0).cpu().numpy()
    return pred_vm, u_recon_phys




def load_mmap_plates(mmap_dir):
    """Discover plates and load metadata from mmap directory."""
    import glob as _glob
    coord_files = sorted(_glob.glob(os.path.join(mmap_dir, '*_coords.npy')))
    plates = {}
    for cf in coord_files:
        pid = os.path.basename(cf).replace('_coords.npy', '')
        coords = np.load(cf)
        fps_path = os.path.join(mmap_dir, f'{pid}_fps_order.npy')
        fps_order = np.load(fps_path).astype(np.int64)
        outputs = np.load(os.path.join(mmap_dir, f'{pid}_outputs.npy'), mmap_mode='r')
        grad_u_path = os.path.join(mmap_dir, f'{pid}_grad_u.npy')
        grad_u = np.load(os.path.realpath(grad_u_path), mmap_mode='r') if os.path.exists(grad_u_path) else None
        plates[pid] = {
            'coords': coords,
            'fps_order': fps_order,
            'outputs': outputs,
            'grad_u': grad_u,
            'n_nodes': coords.shape[0],
            'n_frames': outputs.shape[0],
        }
    return plates


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading tokenizer from {args.checkpoint}")
    engine = LeiaEngine(tokenizer_ckpt=args.checkpoint, dynamics_ckpt=None,
                        device=device)
    cfg = engine.tok_cfg
    disp_scale = engine.disp_scale
    num_latents = engine.K
    hidden_dim = engine.H
    num_dec = cfg.get('num_decoder_layers', TOKENIZER_NUM_DECODER_LAYERS)
    n_input_frames = engine.n_input_frames

    if engine.has_stress_head:
        mode = "stress head"
    elif engine.has_grad_head:
        mode = "gradient head"
    else:
        mode = "autograd"
    print(f"  hidden_dim: {hidden_dim}, latents: {num_latents}, "
          f"dec: {num_dec}, disp_scale: {disp_scale:.4f}, "
          f"n_input_frames: {n_input_frames}, stress: {mode}")
    if num_dec > 0 and mode == "autograd":
        print(f"  WARNING: dec={num_dec} > 0 — autograd Jacobian gives exact "
              f"per-point gradients only for dec=0. Results may be incorrect.")

    print(f"Loading mmap data from {args.mmap_dir}")
    mmap_plates = load_mmap_plates(args.mmap_dir)
    plate_ids = sorted(mmap_plates.keys())
    print(f"  Found {len(plate_ids)} plates")

    print(f"  Subsampling: {args.subsample_n} points "
          f"({args.fps_core_n} FPS core + {args.subsample_n - args.fps_core_n} random)")

    t0 = time.time()
    all_vm_corrs = []
    all_disp_errs = []
    all_pk1_amps = []
    per_plate_results = {}

    for plate_id in plate_ids:
        p = mmap_plates[plate_id]
        coords = p['coords'].astype(np.float64)
        fps_order = p['fps_order']
        n_nodes = p['n_nodes']
        n_total_frames = p['n_frames']
        outputs_arr = p['outputs']
        grad_u_arr = p['grad_u']
        has_grad_u = grad_u_arr is not None

        print(f"\n  {plate_id}: {n_nodes:,} nodes, {n_total_frames} frames, "
              f"grad_u={has_grad_u}")

        # Coordinate normalization (matches training)
        c_min = coords.min(axis=0)
        c_max = coords.max(axis=0)
        coord_range = np.maximum(c_max - c_min, 1e-8)
        coord_scale = (coord_range / 2.0).astype(np.float32)
        coords_norm = ((coords - c_min) / coord_range * 2 - 1).astype(np.float32)

        # Build frame list (skip frame 0 = zero displacement in each trajectory).
        # Mmap layout: flat [T, N, 3] where T = n_traj * n_frames_per_traj.
        # Frames 0, 31, 62, ... are zero-displacement starts.
        n_frames_per_traj = 31  # from FEM config
        all_frame_indices = []
        for start in range(0, n_total_frames, n_frames_per_traj):
            for fi in range(1, min(n_frames_per_traj, n_total_frames - start)):
                all_frame_indices.append(start + fi)

        total_available = len(all_frame_indices)
        if args.n_frames > 0 and args.n_frames < total_available:
            rng = np.random.default_rng(42)
            sel = rng.choice(total_available, size=args.n_frames, replace=False)
            sel.sort()
            eval_frames = [all_frame_indices[i] for i in sel]
        else:
            eval_frames = all_frame_indices

        vm_corrs, disp_errs, pk1_amps = [], [], []
        for idx, frame_ref in enumerate(eval_frames):
            selected = subsample_fps(fps_order, args.subsample_n, args.fps_core_n,
                                     rng=np.random.default_rng(42))

            disp_m1_sub = None
            disp_m2_sub = None
            fi_flat = frame_ref
            gt_disps = np.array(outputs_arr[fi_flat]).astype(np.float64)
            gt_disps_sub = gt_disps[selected].astype(np.float32)
            ref_sub = coords_norm[selected]
            if n_input_frames == 3:
                traj_start = fi_flat - (fi_flat % n_frames_per_traj)
                m1_idx = max(fi_flat - 1, traj_start)
                m2_idx = max(fi_flat - 2, traj_start)
                disp_m1_sub = np.array(outputs_arr[m1_idx])[selected].astype(np.float32)
                disp_m2_sub = np.array(outputs_arr[m2_idx])[selected].astype(np.float32)
            if not has_grad_u:
                continue
            gt_grad_u_sub = np.array(grad_u_arr[fi_flat])[selected].astype(np.float64)
            # GT grad_u is already in physical space (∂u_phys/∂x_phys)
            gt_grad_u_t = torch.tensor(gt_grad_u_sub, dtype=torch.float32,
                                       device=device).unsqueeze(0)
            gt_vm_sub = _grad_u_to_vm(gt_grad_u_t, args.stress_mu,
                                      args.stress_lmbda, device)

            # Predicted stress
            pred_vm, u_recon = compute_predicted_vm(
                engine, ref_sub, gt_disps_sub, coord_scale,
                args.stress_mu, args.stress_lmbda, device,
                disp_m1=disp_m1_sub, disp_m2=disp_m2_sub)

            # Per-frame metrics
            r = pearson_corr(gt_vm_sub, pred_vm)
            vm_corrs.append(r)

            disp_err = float(np.linalg.norm(u_recon - gt_disps_sub) /
                             max(np.linalg.norm(gt_disps_sub), 1e-30))
            disp_errs.append(disp_err)

            safe = gt_vm_sub > 1e-10
            if safe.any():
                pk1_amps.append(float(np.median(pred_vm[safe] / gt_vm_sub[safe])))

            if (idx + 1) % 50 == 0:
                print(f"    {plate_id}: {idx+1}/{len(eval_frames)} frames, "
                      f"median vm_corr={np.median(vm_corrs):.4f}")

        med_vm = float(np.median(vm_corrs)) if vm_corrs else float('nan')
        med_disp = float(np.median(disp_errs)) if disp_errs else float('nan')
        med_pk1 = float(np.median(pk1_amps)) if pk1_amps else float('nan')
        print(f"    {plate_id}: {len(vm_corrs)} frames, median vm_corr={med_vm:.4f}, "
              f"disp_err={med_disp:.4f}, vm_amp={med_pk1:.4f}")

        per_plate_results[plate_id] = {
            'n_frames': len(vm_corrs),
            'n_nodes': n_nodes,
            'median_vm_corr': med_vm,
            'median_disp_err': med_disp,
            'median_vm_amp': med_pk1,
            'vm_corrs': vm_corrs,
            'disp_errs': disp_errs,
        }
        all_vm_corrs.extend(vm_corrs)
        all_disp_errs.extend(disp_errs)
        all_pk1_amps.extend(pk1_amps)

    elapsed = time.time() - t0

    # Overall summary
    median_vm_corr = float(np.median(all_vm_corrs)) if all_vm_corrs else float('nan')
    mean_vm_corr = float(np.mean(all_vm_corrs)) if all_vm_corrs else float('nan')
    median_disp_err = float(np.median(all_disp_errs)) if all_disp_errs else float('nan')
    median_vm_amp = float(np.median(all_pk1_amps)) if all_pk1_amps else float('nan')
    total_frames = len(all_vm_corrs)
    if total_frames == 0:
        raise SystemExit(
            "eval tokenizer: no frame was scored. The stress metrics need the "
            "per-plate *_grad_u.npy in the mmap (a mmap built with "
            "--allow-missing-caches has none) and at least one plate with frames; "
            "see README § 2.")

    print(f"\n{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mode: {mode}, hidden_dim: {hidden_dim}, latents: {num_latents}")
    print(f"Subsampling: {args.subsample_n} ({args.fps_core_n} FPS + "
          f"{args.subsample_n - args.fps_core_n} random)")
    print(f"Material: mu={args.stress_mu}, lambda={args.stress_lmbda}")
    print(f"Plates: {len(plate_ids)}, Frames evaluated: {total_frames}")
    print(f"Time: {elapsed:.0f}s")
    print(f"{'='*60}")
    print(f"OVERALL median von Mises correlation:  {median_vm_corr:.4f}")
    print(f"OVERALL mean von Mises correlation:    {mean_vm_corr:.4f}")
    print(f"OVERALL median displacement rel error: {median_disp_err:.6f}")
    print(f"OVERALL median VM amplification:       {median_vm_amp:.4f}")
    print(f"{'='*60}")
    print(f"\nPer-plate breakdown:")
    for pid in plate_ids:
        r = per_plate_results[pid]
        print(f"  {pid}: vm_corr={r['median_vm_corr']:.4f}, "
              f"disp_err={r['median_disp_err']:.4f}, "
              f"vm_amp={r['median_vm_amp']:.4f} "
              f"({r['n_frames']} frames, {r['n_nodes']} nodes)")
    print(f"{'='*60}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_name = os.path.basename(os.path.dirname(args.checkpoint))

    # Scatter plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_plates = len(plate_ids)
    frames_per_plate = max(1, 10 // max(n_plates, 1))
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), squeeze=False)

    # Use already-evaluated per-frame vm_corrs to pick representative scatter frames
    # Re-evaluate a small sample for scatter plots
    scatter_idx = 0
    for pid in plate_ids:
        if scatter_idx >= 10:
            break
        pr = per_plate_results[pid]
        if pr['n_frames'] == 0:
            continue
        # Pick frames near median vm_corr for representative scatters
        vm_arr = np.array(pr['vm_corrs'])
        med = np.median(vm_arr)
        closest = np.argsort(np.abs(vm_arr - med))[:frames_per_plate]

        p = mmap_plates[pid]
        coords_raw = p['coords'].astype(np.float64)
        fps_order_raw = p['fps_order']
        outputs_raw = p['outputs']
        grad_u_raw = p['grad_u']

        c_min = coords_raw.min(axis=0)
        c_max = coords_raw.max(axis=0)
        coord_range = np.maximum(c_max - c_min, 1e-8)
        coord_scale = (coord_range / 2.0).astype(np.float32)
        coords_norm = ((coords_raw - c_min) / coord_range * 2 - 1).astype(np.float32)

        # Rebuild frame list (mmap layout: flat traj × frame)
        n_frames_per_traj = 31
        all_fi = []
        for start in range(0, p['n_frames'], n_frames_per_traj):
            for fi in range(1, min(n_frames_per_traj, p['n_frames'] - start)):
                all_fi.append(start + fi)

        # Apply same n_frames selection as main loop
        if args.n_frames > 0 and args.n_frames < len(all_fi):
            rng_s = np.random.default_rng(42)
            sel_s = rng_s.choice(len(all_fi), size=args.n_frames, replace=False)
            sel_s.sort()
            all_fi = [all_fi[i] for i in sel_s]

        for ci in closest:
            if scatter_idx >= 10:
                break
            frame_ref = all_fi[ci]
            selected = subsample_fps(fps_order_raw, args.subsample_n, args.fps_core_n,
                                     rng=np.random.default_rng(42))

            disp_m1_sub = None
            disp_m2_sub = None
            fi_flat = frame_ref
            gt_disps_sub = np.array(outputs_raw[fi_flat])[selected].astype(np.float32)
            ref_sub = coords_norm[selected]
            if n_input_frames == 3:
                traj_start = fi_flat - (fi_flat % n_frames_per_traj)
                m1_idx = max(fi_flat - 1, traj_start)
                m2_idx = max(fi_flat - 2, traj_start)
                disp_m1_sub = np.array(outputs_raw[m1_idx])[selected].astype(np.float32)
                disp_m2_sub = np.array(outputs_raw[m2_idx])[selected].astype(np.float32)
            if grad_u_raw is None:
                continue
            gt_gu = np.array(grad_u_raw[fi_flat])[selected].astype(np.float64)
            gt_vm = _grad_u_to_vm(
                torch.tensor(gt_gu, dtype=torch.float32, device=device).unsqueeze(0),
                args.stress_mu, args.stress_lmbda, device)

            pred_vm, _ = compute_predicted_vm(
                engine, ref_sub, gt_disps_sub, coord_scale,
                args.stress_mu, args.stress_lmbda, device,
                disp_m1=disp_m1_sub, disp_m2=disp_m2_sub)

            ax = axes[scatter_idx // 5][scatter_idx % 5]
            # Axis range from the finite values only: an untrained or diverged
            # head can produce inf/NaN stresses, and matplotlib refuses
            # non-finite limits (the metrics above already report NaN for it).
            finite = np.concatenate([gt_vm, pred_vm])
            finite = finite[np.isfinite(finite)]
            vmax = np.percentile(finite, 99) if finite.size else 1.0
            if not np.isfinite(vmax) or vmax < 1e-10:
                vmax = 1.0
            ax.scatter(gt_vm, pred_vm, s=0.3, alpha=0.3, c='steelblue', rasterized=True)
            ax.plot([0, vmax], [0, vmax], 'r--', lw=1)
            ax.set_xlim(0, vmax)
            ax.set_ylim(0, vmax)
            ax.set_aspect('equal')
            r = pearson_corr(gt_vm, pred_vm)
            ax.set_title(f'{pid} fr={frame_ref}\nr={r:.3f}', fontsize=9)
            ax.set_xlabel('GT VM', fontsize=7)
            ax.set_ylabel('Pred VM', fontsize=7)
            ax.tick_params(labelsize=6)
            scatter_idx += 1

    for idx in range(scatter_idx, 10):
        axes[idx // 5][idx % 5].set_visible(False)
    fig.suptitle(f'Lattice tokenizer stress: {ckpt_name}\n'
                 f'{mode}, h={hidden_dim}, {num_latents} latents, '
                 f'sub={args.subsample_n}, '
                 f'median vm_corr={median_vm_corr:.4f}', fontsize=12)
    fig.tight_layout()
    scatter_path = os.path.join(args.output_dir, f'{ts}_{ckpt_name}_vm_scatter.png')
    fig.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nScatter: {scatter_path}")

    # Histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_plates, 1)))
    ax = axes[0]
    for pi, pid in enumerate(plate_ids):
        vm_arr = np.array(per_plate_results[pid]['vm_corrs'])
        short_pid = pid.replace('desc_', '')
        med = per_plate_results[pid]['median_vm_corr']
        vm_arr = np.asarray(vm_arr)[np.isfinite(np.asarray(vm_arr))]   # a garbage head can give inf/NaN
        if vm_arr.size == 0:
            continue
        ax.hist(vm_arr, bins=30, alpha=0.5, color=colors[pi],
                label=f'{short_pid} (med={med:.3f})')
    ax.axvline(median_vm_corr, color='red', ls='--', lw=1.5,
               label=f'overall={median_vm_corr:.3f}')
    ax.set_xlabel('Per-frame von Mises correlation')
    ax.set_ylabel('Count')
    ax.set_title(f'{ckpt_name} ({mode})')
    ax.legend(fontsize=7)

    ax = axes[1]
    for pi, pid in enumerate(plate_ids):
        pr = per_plate_results[pid]
        short_pid = pid.replace('desc_', '')
        ax.scatter(pr['disp_errs'], pr['vm_corrs'], s=1, alpha=0.3,
                   color=colors[pi], label=short_pid)
    ax.set_xlabel('Displacement relative error')
    ax.set_ylabel('Von Mises correlation')
    ax.set_title('VM correlation vs displacement error')
    ax.legend(fontsize=7)
    fig.tight_layout()
    hist_path = os.path.join(args.output_dir, f'{ts}_{ckpt_name}_vm_histogram.png')
    fig.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Histogram: {hist_path}")

    # JSON metrics
    per_plate_json = {}
    for pid in plate_ids:
        r = per_plate_results[pid]
        per_plate_json[pid] = {
            'n_frames': r['n_frames'],
            'n_nodes': r['n_nodes'],
            'median_vm_corr': r['median_vm_corr'],
            'median_disp_err': r['median_disp_err'],
            'median_vm_amp': r['median_vm_amp'],
            'vm_corrs': r['vm_corrs'],
            'disp_errs': r['disp_errs'],
        }

    json_path = os.path.join(args.output_dir, f'{ts}_{ckpt_name}_metrics.json')
    with open(json_path, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'mode': mode,
            'hidden_dim': hidden_dim,
            'num_latents': num_latents,
            'num_decoder_layers': num_dec,
            'disp_scale': disp_scale,
            'subsample_n': args.subsample_n,
            'fps_core_n': args.fps_core_n,
            'stress_mu': args.stress_mu,
            'stress_lmbda': args.stress_lmbda,
            'n_plates': n_plates,
            'n_frames': total_frames,
            'median_vm_corr': median_vm_corr,
            'mean_vm_corr': mean_vm_corr,
            'median_disp_rel_err': median_disp_err,
            'median_vm_amplification': median_vm_amp,
            'per_plate': per_plate_json,
        }, f, indent=2)
    print(f"Metrics: {json_path}")


if __name__ == '__main__':
    main()
