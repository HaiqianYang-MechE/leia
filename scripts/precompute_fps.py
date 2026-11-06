#!/usr/bin/env python
"""
Pre-compute FPS orderings for all plates in parallel.

Reads coords from per-plate NPZ trajectories (or a consolidated HDF5),
computes farthest-point sampling per plate via ProcessPoolExecutor, and
writes `{plate_id}_fps_order.npy` for downstream mmap builders.

Usage:
    python scripts/precompute_fps.py \
        --npz-dir data/microplate_lattice/npz \
        --output-dir data/microplate_lattice/fps \
        --workers 50

`--plate-list` defaults to None (no filter — all subdirs of `--npz-dir`);
pass an explicit JSON manifest to restrict the set.
"""

import os
import json
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import h5py


def farthest_point_sampling(points, seed=42):
    """Greedy FPS on 3D coordinates. Returns full ordering [N] int64."""
    N = len(points)
    rng = np.random.default_rng(seed)

    order = np.empty(N, dtype=np.int64)
    min_dists = np.full(N, np.inf, dtype=np.float64)
    selected = np.zeros(N, dtype=bool)

    idx = rng.integers(N)
    order[0] = idx
    selected[idx] = True

    for i in range(1, N):
        diffs = points - points[order[i - 1]]
        dists_sq = np.sum(diffs * diffs, axis=1)
        min_dists = np.minimum(min_dists, dists_sq)
        min_dists[selected] = -1.0
        idx = int(np.argmax(min_dists))
        order[i] = idx
        selected[idx] = True

    return order


def compute_fps_for_plate(plate_id, coords, output_dir, seed=42):
    """Worker: compute FPS and save .npy."""
    t0 = time.time()
    short_id = plate_id.replace('desc_', '') if plate_id.startswith('desc_') else plate_id
    fps_path = os.path.join(output_dir, f'{short_id}_fps_order.npy')

    order = farthest_point_sampling(coords.astype(np.float64), seed=seed)
    np.save(fps_path, order)

    return plate_id, len(coords), time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None,
                        help='Consolidated FEM HDF5 (legacy)')
    parser.add_argument('--mesh-dir', type=str, default=None,
                        help='Directory with {plate_id}/mesh.npz files')
    parser.add_argument('--npz-dir', type=str, default=None,
                        help='Directory with {plate_id}/traj_*.npz files (reads coords from first npz)')
    parser.add_argument('--plate-list', type=str, default=None,
                        help='JSON with "plates" list to filter')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory for FPS .npy files')
    parser.add_argument('--workers', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-existing', action='store_true', default=True)
    parser.add_argument('--no-skip-existing', dest='skip_existing',
                        action='store_false')
    args = parser.parse_args()

    if not args.input and not args.mesh_dir and not args.npz_dir:
        parser.error("Must provide --input (HDF5), --mesh-dir, or --npz-dir")

    os.makedirs(args.output_dir, exist_ok=True)

    plate_filter = None
    if args.plate_list:
        with open(args.plate_list) as f:
            plate_filter = set(json.load(f)['plates'])

    plates = {}

    if args.mesh_dir:
        # Load coords from {plate_id}/mesh.npz (key: 'points')
        import glob as _glob
        for mesh_path in sorted(_glob.glob(os.path.join(args.mesh_dir, '*/mesh.npz'))):
            pid = os.path.basename(os.path.dirname(mesh_path))
            if plate_filter and pid not in plate_filter:
                continue
            fps_path = os.path.join(args.output_dir, f'{pid}_fps_order.npy')
            if args.skip_existing and os.path.exists(fps_path):
                continue
            mesh = np.load(mesh_path)
            plates[pid] = mesh['points'].astype(np.float64)

    elif args.npz_dir:
        # Load coords from first traj npz per plate
        candidates = sorted(d for d in os.listdir(args.npz_dir)
                            if os.path.isdir(os.path.join(args.npz_dir, d)))
        for pid in candidates:
            if plate_filter and pid not in plate_filter:
                continue
            fps_path = os.path.join(args.output_dir, f'{pid}_fps_order.npy')
            if args.skip_existing and os.path.exists(fps_path):
                continue
            traj_dir = os.path.join(args.npz_dir, pid)
            traj_files = sorted(f for f in os.listdir(traj_dir) if f.endswith('.npz'))
            if not traj_files:
                continue
            first = np.load(os.path.join(traj_dir, traj_files[0]))
            plates[pid] = first['coords'].astype(np.float64)

    else:
        # Legacy HDF5 path
        with h5py.File(args.input, 'r') as fin:
            for pid in sorted(fin.keys()):
                if plate_filter and pid not in plate_filter:
                    continue
                short_id = pid.replace('desc_', '') if pid.startswith('desc_') else pid
                fps_path = os.path.join(args.output_dir, f'{short_id}_fps_order.npy')
                if args.skip_existing and os.path.exists(fps_path):
                    continue
                plates[pid] = fin[pid]['coords'][:].astype(np.float64)

    if not plates:
        print("All FPS orderings already exist. Nothing to do.")
        return

    print(f"Computing FPS for {len(plates)} plates with {args.workers} workers...")
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for pid, coords in plates.items():
            fut = pool.submit(compute_fps_for_plate, pid, coords,
                              args.output_dir, args.seed)
            futures[fut] = pid

        done = 0
        for fut in as_completed(futures):
            pid, n_nodes, dt = fut.result()
            done += 1
            print(f"  [{done}/{len(plates)}] {pid}: {n_nodes:,} nodes, {dt:.1f}s")

    elapsed = time.time() - t_start
    n_fps = len([f for f in os.listdir(args.output_dir) if f.endswith('.npy')])
    print(f"\nDone: {done} FPS orderings computed ({elapsed:.0f}s); "
          f"{n_fps} files in {args.output_dir}/")


if __name__ == '__main__':
    main()
