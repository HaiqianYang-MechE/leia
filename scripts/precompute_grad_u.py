#!/usr/bin/env python
"""
Parallel grad_u precomputation — one worker per plate, writes mmap-ready .npy.

Reads trajectory NPZ files + mesh topology, computes per-node displacement
gradients via tet shape functions, writes {plate_id}_grad_u.npy.

Usage:
    python scripts/precompute_grad_u.py \
        --npz-dir data/microplate_lattice/npz \
        --mesh-dir data/meshes/microplate_lattice \
        --output-dir data/microplate_lattice/grad_u \
        --workers 59

`--plate-list` and `--max-trajs` default to MICROPLATE_LATTICE.splits.
"""

import os
import sys
import json
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leia.stress import tet4_dNdx, tet_quality
from leia.regime import MICROPLATE_LATTICE


def compute_grad_u_for_plate(plate_id, npz_dir, mesh_dir, output_dir, max_trajs):
    """Compute grad_u for one plate, write mmap-ready .npy."""
    t0 = time.time()

    # Load mesh
    mesh_path = os.path.join(mesh_dir, plate_id, 'mesh.npz')
    mesh = np.load(mesh_path)
    coords = mesh['points'].astype(np.float64)
    topo = mesh['cells'].astype(np.int64)
    n_nodes = len(coords)
    n_tets = len(topo)

    # Precompute shape function gradients (one-time per plate)
    dNdx, volumes = tet4_dNdx(coords, topo)
    quality = tet_quality(coords, topo)
    good = quality >= 1e-4
    good_cells = topo[good]
    good_vols = volumes[good]
    good_dNdx = dNdx[good]

    # Pre-scatter volume weights to nodes
    vol_sum = np.zeros(n_nodes, dtype=np.float64)
    for k in range(4):
        np.add.at(vol_sum, good_cells[:, k], good_vols)
    safe = vol_sum > 0

    # Collect trajectory NPZ files
    traj_dir = os.path.join(npz_dir, plate_id)
    traj_files = sorted([f for f in os.listdir(traj_dir) if f.endswith('.npz')])
    if max_trajs > 0:
        traj_files = traj_files[:max_trajs]
    n_traj = len(traj_files)
    # Read frame count from the first trajectory
    first_npz = np.load(os.path.join(traj_dir, traj_files[0]))
    n_frames_per_traj = first_npz['outputs'].shape[0]
    total_frames = n_traj * n_frames_per_traj

    # Pre-allocate output as a memory-mapped file. Write to a .partial path and
    # rename at the end so an interrupted run never leaves a half-filled file
    # that --skip-existing would later accept.
    out_path = os.path.join(output_dir, f'{plate_id}_grad_u.npy')
    tmp_path = out_path + '.partial'
    out_shape = (total_frames, n_nodes, 3, 3)
    fp = np.lib.format.open_memmap(
        tmp_path, mode='w+', dtype=np.float32, shape=out_shape)

    # Process each trajectory
    frame_idx = 0
    for ti, traj_file in enumerate(traj_files):
        traj_data = np.load(os.path.join(traj_dir, traj_file))
        outputs = traj_data['outputs'].astype(np.float64)  # (31, N, 3)

        for fi in range(n_frames_per_traj):
            disps = outputs[fi]  # (N, 3)

            # Per-tet gradient: einsum over good tets only
            u_at_nodes = disps[good_cells]  # (M_good, 4, 3)
            grad_u_tets = np.einsum('mki,mkj->mij', u_at_nodes, good_dNdx)

            # Volume-weighted scatter to nodes
            grad_u_sum = np.zeros((n_nodes, 3, 3), dtype=np.float64)
            for k in range(4):
                np.add.at(grad_u_sum, good_cells[:, k],
                          grad_u_tets * good_vols[:, None, None])
            grad_u_sum[safe] /= vol_sum[safe, None, None]

            fp[frame_idx] = grad_u_sum.astype(np.float32)
            frame_idx += 1

    del fp  # flush mmap
    os.replace(tmp_path, out_path)

    dt = time.time() - t0
    return plate_id, n_nodes, n_traj, total_frames, dt


def expected_frames(npz_dir, plate_id, max_trajs):
    """Frame count a complete cache for this plate must have."""
    traj_dir = os.path.join(npz_dir, plate_id)
    traj_files = sorted(f for f in os.listdir(traj_dir) if f.endswith('.npz'))
    if max_trajs > 0:
        traj_files = traj_files[:max_trajs]
    n_frames = np.load(os.path.join(traj_dir, traj_files[0]))['outputs'].shape[0]
    return len(traj_files) * n_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz-dir', type=str, required=True)
    parser.add_argument('--mesh-dir', type=str, required=True)
    parser.add_argument('--plate-list', type=str, default=None,
                        help='JSON file with the plate IDs to process. '
                             'Default: MICROPLATE_LATTICE.splits.training_plates.')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--max-trajs', type=int,
                        default=MICROPLATE_LATTICE.splits.training_cutoff_index,
                        help='Max trajectories per plate (default sourced '
                             'from MICROPLATE_LATTICE.splits.training_cutoff_index).')
    parser.add_argument('--workers', type=int, default=59)
    parser.add_argument('--skip-existing', action='store_true', default=True)
    parser.add_argument('--no-skip-existing', dest='skip_existing',
                        action='store_false')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.plate_list is not None:
        with open(args.plate_list) as f:
            plates = json.load(f)['plates']
    else:
        plates = list(MICROPLATE_LATTICE.splits.training_plates)

    # Skip existing caches, but only if they cover the requested trajectories
    # (a cache built with a smaller --max-trajs is rebuilt).
    if args.skip_existing:
        to_process = []
        for p in plates:
            out_path = os.path.join(args.output_dir, f'{p}_grad_u.npy')
            if os.path.exists(out_path):
                have = np.load(out_path, mmap_mode='r').shape[0]
                need = expected_frames(args.npz_dir, p, args.max_trajs)
                if have >= need:
                    continue
                print(f"  {p}: existing cache has {have} frames, need {need} — rebuilding")
            to_process.append(p)
        n_skipped = len(plates) - len(to_process)
        if n_skipped:
            print(f"Skipping {n_skipped} already computed plates")
        plates = to_process

    if not plates:
        print("All grad_u files exist. Nothing to do.")
        return

    print(f"Computing grad_u for {len(plates)} plates with {args.workers} workers...")
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for p in plates:
            fut = pool.submit(compute_grad_u_for_plate,
                              p, args.npz_dir, args.mesh_dir,
                              args.output_dir, args.max_trajs)
            futures[fut] = p

        done = 0
        for fut in as_completed(futures):
            plate_id, n_nodes, n_traj, n_frames, dt = fut.result()
            done += 1
            print(f"  [{done}/{len(plates)}] {plate_id}: "
                  f"{n_nodes:,} nodes, {n_traj} traj, {n_frames} frames, "
                  f"{dt:.0f}s")

    elapsed = time.time() - t_start
    n_files = len([f for f in os.listdir(args.output_dir) if f.endswith('.npy')])
    print(f"\nDone: {n_files} grad_u files in {args.output_dir}/ ({elapsed:.0f}s)")


if __name__ == '__main__':
    main()
