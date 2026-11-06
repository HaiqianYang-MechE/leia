#!/usr/bin/env python3
"""Build train and val mmaps for visco load-reverse dynamics training.

Encodes all frames from the load-reverse HDF5 with the trained tokenizer
and writes two mmap directories (train/val) matching the split JSON exactly.

Two separate mmaps are required because MmapWholePlateDynamicsDataset's
val_ratio uses a random shuffle that cannot align with the split indices.

Output layout (one directory per split, e.g. data/microplate_visco/dynamics_mmap/{train,val}):
    visco_load_reverse_prevq.npy  (n_traj*n_frames, K, H) float32 — mmap of tokenizer latents
    visco_load_reverse_cumbc.npy  (n_traj, n_inc, 4) float32 — raw discrete actions {-1,0,+1}
    visco_load_reverse_meta.json  {"n_traj", "n_frames", "n_inc"}

Usage:
    python scripts/build_microplate_visco_mmap.py \\
        --tokenizer-checkpoint checkpoints/tokenizer/visco-tok_*/best_recon.pt \\
        --h5-path data/microplate_visco/microplate_visco.h5 \\
        --splits-json data/manifests/microplate_visco_splits.json \\
        --output-train-dir data/microplate_visco/dynamics_mmap/train \\
        --output-val-dir data/microplate_visco/dynamics_mmap/val
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leia.tokenizer import PerceiverTokenizer

_HDF5_GROUP = 'solid_plate_visco'
_N_FRAMES = 101   # 0 = rest, 1-100 = loaded (50 fwd + 50 rev)
_N_INC = 100      # action increments
_PLATE_ID = 'visco_load_reverse'


def load_tokenizer(ckpt_path, device):
    tokenizer, stats = PerceiverTokenizer.from_checkpoint(ckpt_path, device=device)
    cfg = stats['config']
    n_input_frames = cfg.get('n_input_frames', 1)
    hidden_dim = cfg.get('hidden_dim', 256)
    num_latents = cfg.get('num_latents', 128)
    print(f"Tokenizer: hidden_dim={hidden_dim}, num_latents={num_latents}, "
          f"n_input_frames={n_input_frames}, disp_scale={stats['disp_scale']:.4f}")
    return tokenizer, stats['disp_scale'], n_input_frames, num_latents, hidden_dim


def build_mmap_for_split(split_name, split_indices, h5_path, tokenizer,
                         disp_scale, n_input_frames, num_latents, hidden_dim,
                         output_dir, device):
    """Encode all frames for a split and write the mmap directory."""
    os.makedirs(output_dir, exist_ok=True)

    n_traj = len(split_indices)
    total_frames = n_traj * _N_FRAMES

    prevq_path = os.path.join(output_dir, f'{_PLATE_ID}_prevq.npy')
    cumbc_path = os.path.join(output_dir, f'{_PLATE_ID}_cumbc.npy')
    meta_path  = os.path.join(output_dir, f'{_PLATE_ID}_meta.json')

    print(f"\n[{split_name}] {n_traj} trajs, {total_frames} frames total")
    print(f"  prevq: {prevq_path}  shape=({total_frames}, {num_latents}, {hidden_dim})")

    prevq_out = np.lib.format.open_memmap(
        prevq_path, mode='w+', dtype='float32',
        shape=(total_frames, num_latents, hidden_dim))
    cumbc_out = np.zeros((n_traj, _N_INC, 4), dtype=np.float32)

    with h5py.File(h5_path, 'r') as f:
        g = f[_HDF5_GROUP]
        coords = g['coords'][:]  # (363, 3)
        n_nodes = coords.shape[0]

        c_min = coords.min(axis=0)
        c_max = coords.max(axis=0)
        scale = np.maximum(c_max - c_min, 1e-8)
        ref_norm = ((coords - c_min) / scale * 2 - 1).astype(np.float32)
        ref_t = torch.tensor(ref_norm, device=device).unsqueeze(0)  # (1, 363, 3)

        for i, traj_idx in enumerate(split_indices):
            disps = g['displacements'][traj_idx]  # (101, 363, 3) float32
            inputs = g['inputs'][traj_idx]         # (100, 4) int8

            if n_input_frames == 3:
                x_batch = np.zeros((_N_FRAMES, n_nodes, 9), dtype=np.float32)
                for t in range(_N_FRAMES):
                    fm2 = disps[max(0, t - 2)]
                    fm1 = disps[max(0, t - 1)]
                    ft  = disps[t]
                    x_batch[t] = np.concatenate([fm2, fm1, ft], axis=-1) / disp_scale
            else:
                x_batch = (disps / disp_scale).astype(np.float32)  # (101, 363, 3)

            x_t = torch.tensor(x_batch, device=device)          # (101, 363, D)
            ref_batch = ref_t.expand(_N_FRAMES, -1, -1)          # (101, 363, 3)

            with torch.no_grad():
                prevq_batch = tokenizer.encode(x_t, ref_batch)  # (101, K, H)

            flat_start = i * _N_FRAMES
            prevq_out[flat_start:flat_start + _N_FRAMES] = prevq_batch.cpu().numpy()

            cumbc_out[i] = inputs.astype(np.float32)

            if (i + 1) % 100 == 0 or i == 0:
                print(f"  [{split_name}] {i+1}/{n_traj} trajs encoded...")

    prevq_out.flush()
    del prevq_out

    np.save(cumbc_path, cumbc_out)
    meta = {'n_traj': n_traj, 'n_frames': _N_FRAMES, 'n_inc': _N_INC}
    with open(meta_path, 'w') as fp:
        json.dump(meta, fp, indent=2)

    print(f"  [{split_name}] Done.")
    print(f"    prevq: {prevq_path}")
    print(f"    cumbc: {cumbc_path}  unique={np.unique(cumbc_out)}")
    print(f"    meta:  {meta}")

    pq_verify = np.load(prevq_path, mmap_mode='r')
    print(f"    Verify prevq shape: {pq_verify.shape}")
    assert pq_verify.shape == (total_frames, num_latents, hidden_dim), \
        f"Shape mismatch: {pq_verify.shape}"


def main():
    parser = argparse.ArgumentParser(
        description='Build visco load-reverse mmaps for dynamics training')
    parser.add_argument('--tokenizer-checkpoint', type=str, required=True,
                        help='Trained visco tokenizer: <visco-tok run>/best_recon.pt (README § 3)')
    parser.add_argument('--h5-path', type=str, required=True,
                        help='Visco HDF5 from `run_visco_load_reverse.py build-hdf5` '
                             '(data/microplate_visco/microplate_visco.h5)')
    parser.add_argument('--splits-json', type=str, required=True,
                        help='Split JSON written next to it '
                             '(data/manifests/microplate_visco_splits.json)')
    parser.add_argument('--output-train-dir', type=str,
                        default='data/microplate_visco/dynamics_mmap/train',
                        help='Output directory for train mmap')
    parser.add_argument('--output-val-dir', type=str,
                        default='data/microplate_visco/dynamics_mmap/val',
                        help='Output directory for val mmap')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tokenizer, disp_scale, n_input_frames, num_latents, hidden_dim = load_tokenizer(
        args.tokenizer_checkpoint, device)

    with open(args.splits_json) as f:
        splits = json.load(f)
    train_indices = splits['train']
    val_indices   = splits['val']
    print(f"Split: {len(train_indices)} train, {len(val_indices)} val, "
          f"{len(splits['test'])} test")

    build_mmap_for_split(
        'train', train_indices, args.h5_path,
        tokenizer, disp_scale, n_input_frames, num_latents, hidden_dim,
        args.output_train_dir, device)

    build_mmap_for_split(
        'val', val_indices, args.h5_path,
        tokenizer, disp_scale, n_input_frames, num_latents, hidden_dim,
        args.output_val_dir, device)

    print(f"\nDone.")
    print(f"  Train mmap: {args.output_train_dir}")
    print(f"  Val mmap:   {args.output_val_dir}")
    print(f"\nDynamics training command (README § 3):")
    print(f"  bash scripts/launch.sh train-dynamics --gpus 8 --run-name visco-dyn -- \\")
    print(f"    --tokenizer-checkpoint {args.tokenizer_checkpoint} \\")
    print(f"    --mmap-dir     {args.output_train_dir} \\")
    print(f"    --mmap-dir-val {args.output_val_dir} \\")
    print(f"    --method continuous_pf --rollout-steps 20 --velocity-input \\")
    print(f"    --noise-std 0.0 \\")
    print(f"    --batch-size 32 --lr 3e-4 --weight-decay 1e-4 --epochs 300 \\")
    print(f"    --bf16 --ar-eval-trajs 5 --wandb")


if __name__ == '__main__':
    main()
