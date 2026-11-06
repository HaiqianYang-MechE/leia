#!/usr/bin/env python
"""
Build mmap training dataset directly from NPZ + FPS + grad_u.

Skips the intermediate HDF5 — reads per-trajectory NPZ files, stacks outputs,
copies FPS and grad_u .npy files into the mmap directory.

Usage:
    python scripts/build_tokenizer_mmap.py \
        --npz-dir data/microplate_lattice/npz \
        --fps-dir data/microplate_lattice/fps \
        --grad-u-dir data/microplate_lattice/grad_u \
        --output-dir data/microplate_lattice/plate_dataset_curated_mmap_100k

`--plate-list` and `--max-trajs` default to MICROPLATE_LATTICE.splits.
"""

import os
import json
import argparse
import time
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leia.regime import MICROPLATE_LATTICE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz-dir', type=str, required=True)
    parser.add_argument('--fps-dir', type=str, required=True)
    parser.add_argument('--grad-u-dir', type=str, required=True)
    parser.add_argument('--plate-list', type=str, default=None,
                        help='JSON file with the plate IDs to process. '
                             'Default: MICROPLATE_LATTICE.splits.training_plates.')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--max-trajs', type=int,
                        default=MICROPLATE_LATTICE.splits.training_cutoff_index,
                        help='Max trajectories per plate (default sourced '
                             'from MICROPLATE_LATTICE.splits.training_cutoff_index).')
    parser.add_argument('--allow-missing-caches', action='store_true',
                        help='Only warn when a plate has no FPS or grad_u cache. By '
                             'default that is an error, because the dataset would '
                             'silently train that plate on the full mesh / without a '
                             'stress target.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.plate_list is not None:
        with open(args.plate_list) as f:
            plates = json.load(f)['plates']
    else:
        plates = list(MICROPLATE_LATTICE.splits.training_plates)

    print(f"Building mmap for {len(plates)} plates, max_trajs={args.max_trajs}")
    t_start = time.time()

    total_frames = 0
    total_bytes = 0

    for i, plate_id in enumerate(sorted(plates)):
        t0 = time.time()

        # 1. Read and stack trajectory outputs + coords
        traj_dir = os.path.join(args.npz_dir, plate_id)
        traj_files = sorted([f for f in os.listdir(traj_dir) if f.endswith('.npz')])
        if args.max_trajs > 0:
            traj_files = traj_files[:args.max_trajs]
        n_traj = len(traj_files)

        # Load first to get shapes
        first = np.load(os.path.join(traj_dir, traj_files[0]))
        coords = first['coords'].astype(np.float32)  # (N, 3)
        n_frames_per_traj = first['outputs'].shape[0]  # 31
        n_nodes = coords.shape[0]

        # Stack all outputs: (n_traj * n_frames, N, 3)
        all_outputs = []
        for tf in traj_files:
            data = np.load(os.path.join(traj_dir, tf))
            all_outputs.append(data['outputs'].astype(np.float32))  # (31, N, 3)
        outputs_flat = np.concatenate(all_outputs, axis=0)  # (n_traj*31, N, 3)
        n_frames = outputs_flat.shape[0]
        total_frames += n_frames

        # 2. Save coords
        coords_path = os.path.join(args.output_dir, f'{plate_id}_coords.npy')
        np.save(coords_path, coords)

        # 3. Save outputs
        outputs_path = os.path.join(args.output_dir, f'{plate_id}_outputs.npy')
        np.save(outputs_path, outputs_flat)

        # 4. Copy FPS
        fps_src = os.path.join(args.fps_dir, f'{plate_id}_fps_order.npy')
        fps_dst = os.path.join(args.output_dir, f'{plate_id}_fps_order.npy')
        if os.path.exists(fps_src):
            shutil.copy2(fps_src, fps_dst)
        elif args.allow_missing_caches:
            print(f"  WARNING: no FPS for {plate_id}")
        else:
            raise FileNotFoundError(
                f"{fps_src} not found: run scripts/precompute_fps.py for {plate_id} "
                f"(README § 2) or pass --allow-missing-caches")

        # 5. Copy or link grad_u (already in correct format)
        grad_u_src = os.path.join(args.grad_u_dir, f'{plate_id}_grad_u.npy')
        grad_u_dst = os.path.join(args.output_dir, f'{plate_id}_grad_u.npy')
        if os.path.exists(grad_u_src):
            # Symlink to avoid 30 GB copy per plate
            if os.path.exists(grad_u_dst):
                os.remove(grad_u_dst)
            os.symlink(os.path.abspath(grad_u_src), grad_u_dst)
        elif args.allow_missing_caches:
            print(f"  WARNING: no grad_u for {plate_id}")
        else:
            raise FileNotFoundError(
                f"{grad_u_src} not found: run scripts/precompute_grad_u.py for {plate_id} "
                f"(README § 2) or pass --allow-missing-caches")

        # Track size (coords + outputs, not grad_u symlinks)
        total_bytes += os.path.getsize(coords_path) + os.path.getsize(outputs_path)

        dt = time.time() - t0
        print(f"  [{i+1}/{len(plates)}] {plate_id}: {n_nodes:,} nodes, "
              f"{n_traj} traj, {n_frames} frames, {dt:.1f}s")

    elapsed = time.time() - t_start
    n_files = len([f for f in os.listdir(args.output_dir)])
    print(f"\nDone: {n_files} files in {args.output_dir}/ ({elapsed:.0f}s)")
    print(f"  {len(plates)} plates, {total_frames} total frames")
    print(f"  Outputs size: {total_bytes / 1e9:.1f} GB (+ grad_u symlinks)")


if __name__ == '__main__':
    main()
