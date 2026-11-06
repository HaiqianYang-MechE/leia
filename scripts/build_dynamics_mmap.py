#!/usr/bin/env python
"""Encode prevq tokens for dynamics training (NPZ FEM trajectories → mmap).

Reads per-plate FEM trajectory NPZ files, tokenizes each frame as a whole
plate with FPS subsampling, and writes one mmap'd .npy + .json per plate
suitable for `MmapWholePlateTrajectoryDataset`.

Output layout::

    output_mmap_dir/
        {plate_id}_prevq.npy   (n_traj*n_frames, K, H) float32  — mmap
        {plate_id}_cumbc.npy   (n_traj, n_inc, 4)       float32
        {plate_id}_meta.json   {"n_traj": ..., "n_frames": ..., "n_inc": ...}

Usage::

    python scripts/build_dynamics_mmap.py \\
        --tokenizer checkpoints/tokenizer/production_v2/best.pt \\
        --npz-dir data/microplate_lattice/npz/ \\
        --fps-dir data/microplate_lattice/fps/ \\
        --output-mmap-dir data/microplate_lattice/structure_tokens_prod_mmap/

`--curated-plates` and `--max-trajs` default to MICROPLATE_LATTICE.splits.
"""

import os
import argparse
import time

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leia.tokenizer import PerceiverTokenizer, subsample_fps
from leia.stress import compute_cumulative_bc
from leia.regime import MICROPLATE_LATTICE

# Tokenizer K fallback; real value comes from the loaded ckpt config.
TOKENIZER_NUM_LATENTS = 512


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode prevq tokens for dynamics training")
    parser.add_argument('--npz-dir', type=str, required=True,
                        help='Directory with {plate_id}/traj_*.npz files')
    parser.add_argument('--fps-dir', type=str, required=True,
                        help='Directory with {plate_id}_fps_order.npy files')
    parser.add_argument('--curated-plates', type=str, default=None,
                        help='JSON file with the plate IDs to process. '
                             'Default: MICROPLATE_LATTICE.splits.training_plates.')
    parser.add_argument('--max-trajs', type=int,
                        default=MICROPLATE_LATTICE.splits.training_cutoff_index,
                        help='Max trajectories per plate (default sourced '
                             'from MICROPLATE_LATTICE.splits.training_cutoff_index).')
    parser.add_argument('--tokenizer', required=True,
                        help='Tokenizer checkpoint (.pt)')
    parser.add_argument('--output-mmap-dir', type=str, required=True,
                        help='Output mmap directory')
    parser.add_argument('--subsample-n', type=int, default=10240,
                        help='Total subsampled nodes per frame (paper: 8192 FPS core + 2048 random)')
    parser.add_argument('--fps-core-n', type=int, default=8192,
                        help='FPS core set size, deterministic (paper: 8192)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Frames per tokenizer forward pass')
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


def load_tokenizer(ckpt_path, device):
    """Load tokenizer from checkpoint."""
    tokenizer, stats = PerceiverTokenizer.from_checkpoint(ckpt_path, device=device)
    print(f"  Tokenizer: disp_scale={stats['disp_scale']:.4f}")
    return tokenizer, stats['disp_scale'], stats['config']






@torch.no_grad()
def tokenize_plate_frames(tokenizer, coords, outputs, fps_order, disp_scale,
                          subsample_n, fps_core_n, device, batch_size=4):
    """Tokenize all frames of one plate with FPS subsampling.

    coords: (N, 3) full mesh coords
    outputs: (n_traj, n_frames, N, 3) displacements
    fps_order: (N,) FPS ordering

    Returns: prevq_embeddings (n_traj*n_frames, K, H) float32
    """
    n_traj, n_frames, N, _ = outputs.shape

    # Deterministic subsampling (same for all frames)
    rng = np.random.default_rng(42)
    sub_idx = subsample_fps(fps_order, subsample_n, fps_core_n, rng)
    coords_sub = coords[sub_idx]  # (S, 3)

    # Normalize coords to [-1, 1] using FULL mesh range (matches tokenizer training)
    c_min = coords.min(axis=0)
    c_max = coords.max(axis=0)
    scale = np.maximum(c_max - c_min, 1e-8)
    coords_norm = ((coords - c_min) / scale * 2 - 1).astype(np.float32)
    ref_norm = coords_norm[sub_idx]

    ref_t = torch.tensor(ref_norm, dtype=torch.float32, device=device).unsqueeze(0)

    K = tokenizer.latents.shape[0]
    H = tokenizer.latents.shape[1]
    n_total = n_traj * n_frames
    all_prevq = np.zeros((n_total, K, H), dtype=np.float32)

    n_input_frames = getattr(tokenizer, 'n_input_frames', 1)

    if n_input_frames == 1:
        # Flatten trajectories for batched processing
        disps_flat = outputs[:, :, sub_idx, :].reshape(n_total, len(sub_idx), 3)
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch = disps_flat[start:end]
            x = torch.tensor(batch / disp_scale, dtype=torch.float32, device=device)
            ref = ref_t.expand(x.shape[0], -1, -1)
            z_enc = tokenizer.encode(x, ref)
            all_prevq[start:end] = z_enc.cpu().numpy()
    else:
        # Build 3-frame stacked inputs respecting trajectory boundaries
        S = len(sub_idx)
        stacked = np.zeros((n_total, S, 3 * n_input_frames), dtype=np.float32)
        sub_outputs = outputs[:, :, sub_idx, :]  # (n_traj, n_frames, S, 3)
        for ti in range(n_traj):
            for fi in range(n_frames):
                fm1 = max(0, fi - 1)
                fm2 = max(0, fi - 2)
                flat_idx = ti * n_frames + fi
                stacked[flat_idx, :, :3] = sub_outputs[ti, fm2]
                stacked[flat_idx, :, 3:6] = sub_outputs[ti, fm1]
                stacked[flat_idx, :, 6:9] = sub_outputs[ti, fi]
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch = stacked[start:end]
            x = torch.tensor(batch / disp_scale, dtype=torch.float32, device=device)
            ref = ref_t.expand(x.shape[0], -1, -1)
            z_enc = tokenizer.encode(x, ref)
            all_prevq[start:end] = z_enc.cpu().numpy()

    return all_prevq


def write_mmap_plate(output_dir, pid, data, K, H, tokenizer_ckpt,
                     subsample_n, fps_core_n):
    """Write one plate's data as mmap .npy files + metadata JSON."""
    import json as _json
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f'{pid}_prevq.npy'), data['prevq_embeddings'])
    np.save(os.path.join(output_dir, f'{pid}_cumbc.npy'), data['cumulative_bc'])
    np.save(os.path.join(output_dir, f'{pid}_inputs.npy'), data['inputs'])
    meta = {
        'n_traj': int(data['n_traj']),
        'n_frames': int(data['n_frames']),
        'n_inc': int(data['n_inc']),
        'action_dim': int(data['cumulative_bc'].shape[-1]),
        'num_latents': int(K),
        'hidden_dim': int(H),
        'tokenizer_checkpoint': tokenizer_ckpt,
        'subsample_n': int(subsample_n),
        'fps_core_n': int(fps_core_n),
    }
    with open(os.path.join(output_dir, f'{pid}_meta.json'), 'w') as f:
        _json.dump(meta, f)


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}", flush=True)
    tokenizer, disp_scale, tok_cfg = load_tokenizer(args.tokenizer, device)
    K = tok_cfg.get('num_latents', TOKENIZER_NUM_LATENTS)
    H = tokenizer.latents.shape[1]
    print(f"  K={K}, H={H}, subsample_n={args.subsample_n}, fps_core_n={args.fps_core_n}",
          flush=True)

    total_t0 = time.time()
    total_traj = 0
    total_trans = 0
    n_plates_processed = 0

    import json as _json
    if args.curated_plates is not None:
        with open(args.curated_plates) as f:
            plate_ids = sorted(_json.load(f)['plates'])
    else:
        plate_ids = sorted(MICROPLATE_LATTICE.splits.training_plates)
    print(f"\nPlates ({len(plate_ids)}): {plate_ids}", flush=True)

    for pi, pid in enumerate(plate_ids):
        print(f"\n{'='*60}", flush=True)
        print(f"[{pi+1}/{len(plate_ids)}] {pid}", flush=True)
        print(f"{'='*60}", flush=True)

        # Discover trajectories
        traj_dir = os.path.join(args.npz_dir, pid)
        if not os.path.isdir(traj_dir):
            print(f"  WARNING: {traj_dir} not found, skipping", flush=True)
            continue
        traj_files = sorted(f for f in os.listdir(traj_dir) if f.endswith('.npz'))
        if args.max_trajs > 0:
            traj_files = traj_files[:args.max_trajs]
        n_traj = len(traj_files)
        if n_traj == 0:
            print(f"  WARNING: no .npz files in {traj_dir}, skipping", flush=True)
            continue

        # Load coords from first trajectory
        first = np.load(os.path.join(traj_dir, traj_files[0]))
        coords = first['coords'].astype(np.float32)

        # Load FPS ordering
        fps_path = os.path.join(args.fps_dir, f'{pid}_fps_order.npy')
        if not os.path.exists(fps_path):
            print(f"  WARNING: {fps_path} not found, skipping", flush=True)
            continue
        fps_order = np.load(fps_path)

        # Load all trajectories: inputs and outputs
        print(f"  Loading {n_traj} trajectories...", flush=True)
        all_inputs = []
        all_outputs = []
        for ti, tf in enumerate(traj_files):
            data = np.load(os.path.join(traj_dir, tf))
            all_inputs.append(data['inputs'])      # (30, 4)
            all_outputs.append(data['outputs'].astype(np.float32))  # (31, N, 3)
            if (ti + 1) % 50 == 0:
                print(f"    loaded {ti+1}/{n_traj}", flush=True)

        inputs = np.stack(all_inputs)    # (n_traj, 30, 4)
        outputs = np.stack(all_outputs)  # (n_traj, 31, N, 3)
        n_frames = outputs.shape[1]
        n_inc = n_frames - 1

        print(f"  {n_traj} traj, {n_frames} frames, {coords.shape[0]} nodes", flush=True)

        # Compute cumulative BC
        cumulative_bc = compute_cumulative_bc(inputs)

        # Tokenize all frames
        print(f"  Tokenizing ({n_traj * n_frames} frames, "
              f"subsample={args.subsample_n})...", flush=True)
        t0 = time.time()
        prevq = tokenize_plate_frames(
            tokenizer, coords, outputs, fps_order, disp_scale,
            args.subsample_n, args.fps_core_n, device, args.batch_size)
        tok_time = time.time() - t0
        print(f"  Tokenized: {tok_time:.1f}s, shape={prevq.shape}", flush=True)

        plate_data = {
            'prevq_embeddings': prevq,
            'cumulative_bc': cumulative_bc,
            'inputs': inputs,
            'n_traj': n_traj,
            'n_frames': n_frames,
            'n_inc': n_inc,
        }

        total_traj += n_traj
        total_trans += n_traj * n_inc
        n_plates_processed += 1

        write_mmap_plate(args.output_mmap_dir, pid, plate_data,
                         K, H, args.tokenizer, args.subsample_n,
                         args.fps_core_n)
        prevq_mb = plate_data['prevq_embeddings'].nbytes / 1e6
        print(f"  Wrote mmap for {pid} ({prevq_mb:.1f} MB prevq)", flush=True)
        del plate_data, prevq, cumulative_bc, inputs, outputs
        del all_inputs, all_outputs

    n_skipped = len(plate_ids) - n_plates_processed
    if n_plates_processed == 0:
        sys.exit(f"ERROR: none of the {len(plate_ids)} requested plates was processed; "
                 f"see the WARNING lines above. FPS orders come from "
                 f"scripts/precompute_fps.py (README § 2).")

    total_bytes = sum(
        os.path.getsize(os.path.join(args.output_mmap_dir, f))
        for f in os.listdir(args.output_mmap_dir) if f.endswith('.npy'))
    print(f"\nMmap output: {args.output_mmap_dir} ({total_bytes / 1e9:.1f} GB)", flush=True)
    print(f"  Plates: {n_plates_processed}"
          + (f" ({n_skipped} skipped, see WARNING lines)" if n_skipped else ""), flush=True)
    print(f"  Total trajectories: {total_traj}", flush=True)
    print(f"  Total transitions: {total_trans}", flush=True)

    total_time = time.time() - total_t0
    print(f"\nTotal: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
