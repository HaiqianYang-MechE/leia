#!/usr/bin/env python
"""Pre-compute per-plate tet geometry for whole-plate tet Sobolev training.

For each plate, loads mesh.npz, computes:
  - cells (M, 4) int64               — tet connectivity into the plate's nodes
  - dNdx (M, 4, 3) float32           — shape function gradients
  - volumes (M,) float32             — tet volumes
  - good (M,) bool                   — quality mask (>= 1e-4)
  - fps_tet_order (K,) int64         — FPS ordering of good-tet centroids, K = min(M_good, --fps-k-max)

Saves to {output-dir}/{plate_id}.npz.

This is consumed by `MmapMultiStructureDataset` when `--sobolev-method tet` is
selected during tokenizer training. The FPS ordering of tet centroids enables
deterministic spatially-uniform "core" sampling per training step, which is
combined with random remainder selection to give an unbiased Monte Carlo
estimator of the full-plate tet Sobolev loss.

Usage:
    bash scripts/launch.sh precompute-tet-geom --gpu-ids 0 -- \
        --mesh-dir data/meshes/microplate_lattice \
        --mmap-dir data/microplate_lattice/plate_dataset_curated_mmap_100k \
        --output-dir data/microplate_lattice/plate_tet_geom \
        --workers 8

`--plate-list` defaults to MICROPLATE_LATTICE.splits.training_plates.
"""

import os
import sys
import json
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leia.stress import precompute_stress_geometry
from leia.regime import MICROPLATE_LATTICE


def farthest_point_sampling(points, seed=42, k_max=None):
    """Greedy FPS on a (N, 3) coord array. Returns ordering [k_max] int64.

    If `k_max` is None, returns the full N-point ordering (O(N^2) complexity).
    For tet centroids on a ~1M-tet plate this is too slow; pass `k_max` to
    cap the selection at the first K points (O(k_max * N) complexity), which
    is enough for the deterministic FPS-core in training. Random remainder at
    training time uniformly samples from the leftover good tets directly.
    """
    N = len(points)
    if k_max is None or k_max >= N:
        k_max = N
    rng = np.random.default_rng(seed)

    order = np.empty(k_max, dtype=np.int64)
    min_dists = np.full(N, np.inf, dtype=np.float64)
    selected = np.zeros(N, dtype=bool)

    idx = rng.integers(N)
    order[0] = idx
    selected[idx] = True

    for i in range(1, k_max):
        diffs = points - points[order[i - 1]]
        dists_sq = np.sum(diffs * diffs, axis=1)
        min_dists = np.minimum(min_dists, dists_sq)
        min_dists[selected] = -1.0
        idx = int(np.argmax(min_dists))
        order[i] = idx
        selected[idx] = True

    return order


def compute_one(plate_id, mesh_path, mmap_dir, output_dir, seed, fps_k_max):
    """Worker: load full mesh, restrict tets to those entirely within the
    curated mmap nodes, recompute geometry on curated coords + remapped cells,
    FPS over curated tet centroids, and save.

    The tokenizer mmap (`<mmap_dir>/<pid>_coords.npy`) holds the full NPZ
    coords (build_tokenizer_mmap writes them unchanged), so the `mesh.npz`
    tets index it directly; the filter/remap below is a guard for mmaps built
    from a node subset and is a no-op for the documented build.
    """
    t0 = time.time()
    out_path = os.path.join(output_dir, f'{plate_id}.npz')

    with np.load(mesh_path) as m:
        if 'cells' in m:
            cells_full = m['cells'].astype(np.int64)
        elif 'topology' in m:
            cells_full = m['topology'].astype(np.int64)
        elif 'tets' in m:
            cells_full = m['tets'].astype(np.int64)
        else:
            raise KeyError(
                f"No topology array in {mesh_path}; expected one of "
                f"{{cells, topology, tets}}, got {list(m.keys())}")
        points_full = m['points'].astype(np.float64)

    # Load curated coords (the actual node set the tokenizer trains on).
    curated_path = os.path.join(mmap_dir, f'{plate_id}_coords.npy')
    if not os.path.exists(curated_path):
        raise FileNotFoundError(
            f"Curated coords not found at {curated_path}; tet geom must be "
            f"built against the same node set the dataset reads.")
    coords_curated = np.load(curated_path).astype(np.float64)  # (N_curated, 3)

    # Map curated -> full mesh indices via nearest neighbor (zero distance
    # expected when coords_curated is an exact subset of points_full).
    from scipy.spatial import cKDTree
    tree = cKDTree(points_full)
    dist, curated_to_full = tree.query(coords_curated, k=1)
    if dist.max() > 1e-6:
        raise RuntimeError(
            f"{plate_id}: curated coords are NOT a subset of full mesh points "
            f"(max dist {dist.max():.3e}). Tet geometry can't be built.")
    curated_to_full = curated_to_full.astype(np.int64)

    # Inverse map: full_to_curated[full_id] = curated index, or -1 if not in curated.
    full_to_curated = np.full(points_full.shape[0], -1, dtype=np.int64)
    full_to_curated[curated_to_full] = np.arange(coords_curated.shape[0],
                                                  dtype=np.int64)

    # Filter cells to those entirely within the curated subset.
    in_curated = (full_to_curated[cells_full] >= 0).all(axis=1)
    cells_curated = full_to_curated[cells_full[in_curated]]

    if cells_curated.shape[0] == 0:
        raise RuntimeError(f"{plate_id}: zero tets fully within curated subset")

    dNdx, volumes, good = precompute_stress_geometry(coords_curated, cells_curated)
    dNdx = dNdx.astype(np.float32)
    volumes = volumes.astype(np.float32)

    # FPS over good-tet centroids (in curated coordinates).
    centroids = coords_curated[cells_curated].mean(axis=1)
    good_idx = np.flatnonzero(good)
    if good_idx.size == 0:
        raise RuntimeError(f"{plate_id}: zero good tets after filtering")

    good_centroids = centroids[good_idx]
    fps_local = farthest_point_sampling(
        good_centroids, seed=seed, k_max=min(fps_k_max, good_idx.size))
    fps_tet_order = good_idx[fps_local].astype(np.int64)

    np.savez(
        out_path,
        cells=cells_curated.astype(np.int64),    # (M', 4) into curated coord set
        dNdx=dNdx,                                # (M', 4, 3) float32
        volumes=volumes,                          # (M',) float32
        good=good,                                # (M',) bool
        fps_tet_order=fps_tet_order,              # (M_good,) int64
        # Provenance
        n_full_nodes=np.int64(points_full.shape[0]),
        n_curated_nodes=np.int64(coords_curated.shape[0]),
        n_full_tets=np.int64(cells_full.shape[0]),
    )

    elapsed = time.time() - t0
    return plate_id, len(cells_full), len(cells_curated), int(good.sum()), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh-dir', required=True,
                        help='Directory with {plate_id}/mesh.npz files (full mesh)')
    parser.add_argument('--mmap-dir', required=True,
                        help='Tokenizer mmap dir with {plate_id}_coords.npy '
                             '(curated node set). Tet topology is filtered to '
                             'tets entirely within this curated subset.')
    parser.add_argument('--plate-list', type=str, default=None,
                        help='JSON with "plates" list to filter. Default: '
                             'MICROPLATE_LATTICE.splits.training_plates.')
    parser.add_argument('--output-dir', required=True,
                        help='Directory for {plate_id}.npz outputs')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fps-k-max', type=int, default=10000,
                        help='Cap FPS to first K_max selections per plate '
                             '(default 10000). Training-time core sampling '
                             'uses K_core ~300, so this leaves plenty of '
                             'headroom. Higher values cost FPS time linearly.')
    parser.add_argument('--skip-existing', action='store_true', default=True)
    parser.add_argument('--no-skip-existing', dest='skip_existing',
                        action='store_false')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.plate_list is not None:
        with open(args.plate_list) as f:
            plate_ids = sorted(json.load(f)['plates'])
    else:
        plate_ids = sorted(MICROPLATE_LATTICE.splits.training_plates)
    print(f"Plates: {len(plate_ids)}", flush=True)

    jobs = []
    for pid in plate_ids:
        out_path = os.path.join(args.output_dir, f'{pid}.npz')
        if args.skip_existing and os.path.exists(out_path):
            print(f"  skip {pid} (exists)", flush=True)
            continue
        mesh_path = os.path.join(args.mesh_dir, pid, 'mesh.npz')
        if not os.path.exists(mesh_path):
            print(f"  WARNING: {mesh_path} missing, skipping {pid}", flush=True)
            continue
        jobs.append((pid, mesh_path))

    if not jobs:
        print("All outputs already exist; nothing to do.")
        return

    print(f"Computing tet geometry for {len(jobs)} plates "
          f"({args.workers} workers)...", flush=True)
    t_all = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(compute_one, pid, mesh_path, args.mmap_dir,
                      args.output_dir, args.seed, args.fps_k_max): pid
            for pid, mesh_path in jobs
        }
        for fut in as_completed(futures):
            pid, n_full, n_curated, n_good, elapsed = fut.result()
            print(f"  {pid}: full={n_full} -> curated={n_curated} "
                  f"({n_good} good), {elapsed:.1f}s", flush=True)

    print(f"\nTotal wall: {time.time() - t_all:.1f}s")


if __name__ == '__main__':
    main()
