#!/usr/bin/env python3
"""Evaluate LEIA on MicroPlate visco (load-reverse): teacher forcing, reconstruction, AR.

Three evaluation modes, all scored against ground-truth frames 1..100:
    teacher_forcing  — one-step prediction: encode the ground-truth history
                       window ending at frame t, take ONE dynamics step with
                       action a_t, decode, compare with ground-truth frame t+1.
    reconstruction   — tokenizer round trip only: encode the ground-truth
                       window ending at frame t, decode, compare with frame t.
                       No dynamics step; isolates tokenizer error.
    autoregressive   — encode the rest frame once, roll the dynamics model
                       forward for 100 steps on its own predictions.

Prediction files:
    predictions_tf.npz     — teacher forcing (one-step prediction)
    predictions_recon.npz  — tokenizer reconstruction
    predictions_ar.npz     — autoregressive rollout

Output arrays (n_traj trajectories, n_frames frames, n_nodes mesh nodes):
    displacement:  (n_traj, n_frames, n_nodes, 3)     float32
    cauchy_stress: (n_traj, n_frames, n_nodes, 3, 3)  float32  -- full 3x3 Cauchy tensor

Usage:
    python scripts/eval_microplate_visco.py \\
        --tokenizer-checkpoint checkpoints/tokenizer/visco-tok_*/best_recon.pt \\
        --dynamics-checkpoint  checkpoints/dynamics/visco-dyn_*/best_ar.pt \\
        --h5-path data/microplate_visco/microplate_visco.h5 \\
        --splits-json data/manifests/microplate_visco_splits.json \\
        --output-dir outputs/leia_visco
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leia.engine import LeiaEngine
from leia.metrics import pearson_corr, displacement_relative_l2
from leia.stress import sym6_to_cauchy, von_mises

_HDF5_GROUP = 'solid_plate_visco'


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    engine = LeiaEngine(
        tokenizer_ckpt=args.tokenizer_checkpoint,
        dynamics_ckpt=args.dynamics_checkpoint,
        device=device,
    )
    if not engine.has_stress_head:
        raise ValueError("Tokenizer checkpoint missing stress_mean/stress_std — "
                         "was it trained with --stress-head --cauchy-stress-weight 1.0?")
    n_input_frames = engine.n_input_frames
    velocity_input = engine.velocity_input
    print(f"Tokenizer: disp_scale={engine.disp_scale:.4f}, "
          f"n_input_frames={n_input_frames}, stress_head=True")
    print(f"Dynamics: velocity_input={velocity_input}")

    with open(args.splits_json) as f:
        splits = json.load(f)
    traj_indices = splits[args.split]
    if args.n_traj > 0:
        traj_indices = traj_indices[:args.n_traj]
    n_trajs = len(traj_indices)

    with h5py.File(args.h5_path, 'r') as f:
        g = f[_HDF5_GROUP]
        coords = g['coords'][:]   # (n_nodes, 3)
        n_nodes = coords.shape[0]
        n_frames_total = g['displacements'].shape[1]   # rest frame 0 + n_frames
        n_frames = n_frames_total - 1                  # scored frames 1..n_frames
        assert g['inputs'].shape[1] == n_frames, "one action per scored frame"

        gt_disps_all  = np.zeros((n_trajs, n_frames_total, n_nodes, 3),    dtype=np.float32)
        gt_stress_all = np.zeros((n_trajs, n_frames_total, n_nodes, 3, 3), dtype=np.float32)
        actions_all   = np.zeros((n_trajs, n_frames, 4),             dtype=np.float32)

        for i, idx in enumerate(traj_indices):
            gt_disps_all[i]  = g['displacements'][idx]              # (n_frames_total, n_nodes, 3)
            gt_stress_all[i] = g['cauchy_stress'][idx]              # (n_frames_total, n_nodes, 3, 3)
            actions_all[i]   = g['inputs'][idx].astype(np.float32)  # (n_frames, 4) int8->float32
            if (i + 1) % 20 == 0:
                print(f"  Loaded {i+1}/{n_trajs} trajectories")

    gt_disps  = gt_disps_all[:, 1:]   # (n_trajs, n_frames, n_nodes, 3)
    gt_stress = gt_stress_all[:, 1:]  # (n_trajs, n_frames, n_nodes, 3, 3)

    c_min = coords.min(axis=0)
    c_max = coords.max(axis=0)
    scale = np.maximum(c_max - c_min, 1e-8)
    ref_norm = ((coords - c_min) / scale * 2 - 1).astype(np.float32)
    ref_t = torch.tensor(ref_norm, device=device).unsqueeze(0)  # (1, n_nodes, 3)

    print(f"\n{args.split} split: {n_trajs} trajectories, {n_frames} frames, {n_nodes} nodes")

    pred_disps_tf       = np.zeros((n_trajs, n_frames, n_nodes, 3), dtype=np.float32)
    pred_stress_tf_sym6 = np.zeros((n_trajs, n_frames, n_nodes, 6), dtype=np.float32)
    pred_disps_ar       = np.zeros((n_trajs, n_frames, n_nodes, 3), dtype=np.float32)
    pred_stress_ar_sym6 = np.zeros((n_trajs, n_frames, n_nodes, 6), dtype=np.float32)
    pred_disps_rc       = np.zeros((n_trajs, n_frames, n_nodes, 3), dtype=np.float32)
    pred_stress_rc_sym6 = np.zeros((n_trajs, n_frames, n_nodes, 6), dtype=np.float32)

    for ti in range(n_trajs):
        # ------------------------------------------------------------------
        # Ground-truth latents z_t for every frame t = 0..n_frames, each encoded
        # from the clamped history window [t-F+1, ..., t] (the convention
        # used when building the dynamics training mmap). With z_{-1} := z_0,
        # this matches the AR initialisation below.
        # ------------------------------------------------------------------
        def _encode_gt(t):
            if n_input_frames > 1:
                history = np.stack(
                    [gt_disps_all[ti, max(0, t - k)]
                     for k in range(n_input_frames - 1, -1, -1)],
                    axis=0).astype(np.float32)
                return engine.encode_history(history, ref_norm)
            return engine.encode_frame(
                gt_disps_all[ti, t].astype(np.float32), ref_norm)

        z_gt = [_encode_gt(t) for t in range(n_frames_total)]   # (K, H) each

        # ------------------------------------------------------------------
        # Teacher forcing (one-step prediction): from the ground-truth latent
        # at frame fi and action a_fi, predict frame fi+1; compare with the
        # ground-truth frame fi+1 (= gt_disps[:, fi]). Same action/frame
        # indexing as the AR loop below, but the input is always ground truth.
        # ------------------------------------------------------------------
        for fi in range(n_frames):  # action index fi -> predicts frame fi+1
            act = torch.tensor(
                actions_all[ti, fi], dtype=torch.float32, device=device).unsqueeze(0)
            z_t = z_gt[fi].unsqueeze(0)                                   # (1, K, H)
            prev_state_arg = (z_gt[max(0, fi - 1)].unsqueeze(0)
                              if velocity_input else None)                # z_{t-1}
            z_next, _ = engine.step_dynamics(z_t, act, prev_state=prev_state_arg)

            out = engine.decode(z_next, ref_t, compute_stress=True)
            pred_disps_tf[ti, fi] = out["disp"].squeeze(0).cpu().numpy()
            if out["cauchy_sym6"] is not None:
                pred_stress_tf_sym6[ti, fi] = out["cauchy_sym6"].squeeze(0).cpu().numpy()

        # ------------------------------------------------------------------
        # Tokenizer reconstruction (no dynamics step): decode the ground-truth
        # latent of frame fi+1 and compare with frame fi+1. Isolates tokenizer
        # error; diagnostic only.
        # ------------------------------------------------------------------
        for fi in range(n_frames):
            out = engine.decode(z_gt[fi + 1].unsqueeze(0), ref_t, compute_stress=True)
            pred_disps_rc[ti, fi] = out["disp"].squeeze(0).cpu().numpy()
            if out["cauchy_sym6"] is not None:
                pred_stress_rc_sym6[ti, fi] = out["cauchy_sym6"].squeeze(0).cpu().numpy()

        # ------------------------------------------------------------------
        # Autoregressive rollout: start from frame 0 (rest state)
        # ------------------------------------------------------------------
        zeros = np.zeros((n_nodes, 3), dtype=np.float32)
        prevq = engine.encode_frame(zeros, ref_norm).unsqueeze(0)   # (1, K, H)
        prevq_prev = prevq.clone()                                  # z_{-1}=z_0 at rest

        for fi in range(n_frames):
            act = torch.tensor(
                actions_all[ti, fi], dtype=torch.float32, device=device).unsqueeze(0)
            prev_state_arg = prevq_prev if velocity_input else None
            prevq_new, _ = engine.step_dynamics(prevq, act, prev_state=prev_state_arg)
            prevq_prev = prevq
            prevq = prevq_new

            out = engine.decode(prevq, ref_t, compute_stress=True)
            pred_disps_ar[ti, fi] = out["disp"].squeeze(0).cpu().numpy()
            if out["cauchy_sym6"] is not None:
                pred_stress_ar_sym6[ti, fi] = out["cauchy_sym6"].squeeze(0).cpu().numpy()

        if (ti + 1) % 10 == 0 or ti == 0:
            print(f"  Evaluated trajectory {ti+1}/{n_trajs}")

    pred_stress_tf = sym6_to_cauchy(pred_stress_tf_sym6)  # (n_trajs, n_frames, n_nodes, 3, 3)
    pred_stress_ar = sym6_to_cauchy(pred_stress_ar_sym6)  # (n_trajs, n_frames, n_nodes, 3, 3)
    pred_stress_rc = sym6_to_cauchy(pred_stress_rc_sym6)  # (n_trajs, n_frames, n_nodes, 3, 3)

    # ------------------------------------------------------------------
    # Compute and print metrics
    # ------------------------------------------------------------------
    def compute_metrics(pred_d, pred_s33, gt_d, gt_s33, label):
        gt_vm   = von_mises(gt_s33.astype(np.float64))
        pred_vm = von_mises(pred_s33.astype(np.float64))

        vm_corrs_per_traj = []
        for ti in range(n_trajs):
            traj_corrs = []
            for fi in range(n_frames):
                gv = gt_vm[ti, fi]
                pv = pred_vm[ti, fi]
                if gv.std() > 1e-10 and pv.std() > 1e-10:
                    corr = pearson_corr(pv, gv)
                    traj_corrs.append(corr)
            if traj_corrs:
                vm_corrs_per_traj.append(float(np.median(traj_corrs)))

        disp_errors = displacement_relative_l2(pred_d, gt_d)
        results = {
            'disp_rel_l2_median': float(np.median(disp_errors)),
            'disp_rel_l2_mean':   float(np.mean(disp_errors)),
            'vm_corr_median':     float(np.median(vm_corrs_per_traj)),
            'vm_corr_mean':       float(np.mean(vm_corrs_per_traj)),
            'vm_corr_per_traj':   vm_corrs_per_traj,
        }
        print(f"\n{label}")
        print(f"  Disp Rel L2:    median={results['disp_rel_l2_median']:.4f}  "
              f"mean={results['disp_rel_l2_mean']:.4f}")
        print(f"  VM Stress Corr: median={results['vm_corr_median']:.4f}  "
              f"mean={results['vm_corr_mean']:.4f}")
        return results

    print("\n" + "=" * 60)
    tf_metrics = compute_metrics(pred_disps_tf, pred_stress_tf,
                                  gt_disps, gt_stress, "Teacher-Forcing (one-step prediction)")
    rc_metrics = compute_metrics(pred_disps_rc, pred_stress_rc,
                                  gt_disps, gt_stress, "Reconstruction (tokenizer only, no dynamics)")
    ar_metrics = compute_metrics(pred_disps_ar, pred_stress_ar,
                                  gt_disps, gt_stress, "Autoregressive Rollout")
    print("=" * 60)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        tf_path = os.path.join(args.output_dir, 'predictions_tf.npz')
        np.savez(tf_path,
                 displacement=pred_disps_tf,
                 cauchy_stress=pred_stress_tf,
                 traj_indices=np.array(traj_indices))
        print(f"\nTF predictions saved:  {tf_path}")

        ar_path = os.path.join(args.output_dir, 'predictions_ar.npz')
        np.savez(ar_path,
                 displacement=pred_disps_ar,
                 cauchy_stress=pred_stress_ar,
                 traj_indices=np.array(traj_indices))
        print(f"AR predictions saved:  {ar_path}")

        rc_path = os.path.join(args.output_dir, 'predictions_recon.npz')
        np.savez(rc_path,
                 displacement=pred_disps_rc,
                 cauchy_stress=pred_stress_rc,
                 traj_indices=np.array(traj_indices))
        print(f"Reconstruction saved:  {rc_path}")

        all_metrics = {
            'tokenizer_checkpoint': args.tokenizer_checkpoint,
            'dynamics_checkpoint': args.dynamics_checkpoint,
            'split': args.split,
            'n_trajs': n_trajs,
            'n_frames': n_frames,
            'n_nodes': n_nodes,
            'teacher_forcing': tf_metrics,
            'reconstruction': rc_metrics,
            'autoregressive': ar_metrics,
        }
        metrics_path = os.path.join(args.output_dir, 'metrics.json')
        with open(metrics_path, 'w') as fp:
            json.dump(all_metrics, fp, indent=2)
        print(f"Metrics saved:         {metrics_path}")

        print(f"\nMetrics in {metrics_path}: 'teacher_forcing' (one-step) and "
              f"'autoregressive' are the Table 2 modes; 'reconstruction' is diagnostic.")

    return tf_metrics, rc_metrics, ar_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate LEIA on MicroPlate visco (one-step TF, reconstruction, AR)')
    parser.add_argument('--tokenizer-checkpoint', type=str, required=True)
    parser.add_argument('--dynamics-checkpoint', type=str, required=True)
    parser.add_argument('--h5-path', type=str, required=True,
                        help='Path to microplate_visco.h5')
    parser.add_argument('--splits-json', type=str, required=True,
                        help='Path to microplate_visco_splits.json')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--n-traj', type=int, default=0,
                        help='Limit to first N trajectories (0=all)')
    parser.add_argument('--output-dir', type=str, default='outputs/leia_visco',
                        help='Where to save predictions and metrics')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
