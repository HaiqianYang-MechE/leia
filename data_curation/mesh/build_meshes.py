#!/usr/bin/env python
"""Build canonical tet meshes from `data/rve_bank/mg_*/graph.json`.

Pipeline (matches the recipe used to produce the paper's meshes):

    physics.meshing.mesh_graph(tiles=CANONICAL_TILES, res_per_cell=30,
                               tetgen_kill_on_timeout=True)

(`CANONICAL_TILES` is sourced from the microplate_lattice manifest.)

      → modular SDF wrap → marching cubes (numba.cuda)
      → TetGen subprocess+kill with CANONICAL_TETGEN switch chain
      → plate-scale points in [-4.9, 4.9]² (x/y) × [-0.98, 0.98] (z)

res_per_cell=30 reproduces the node/tet counts of the meshes used for the
paper. TetGen is deterministic for a given input, but the
GPU SDF / marching-cubes stage is float32 and its rounding depends on the CUDA
compiler, driver and GPU, so on another setup a few plates may get slightly
different connectivity. Downstream steps derive from the built meshes, so this
does not affect reproduction.

Usage:
    python data_curation/mesh/build_meshes.py --all
    python data_curation/mesh/build_meshes.py --plates mg_046 mg_111
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Make `physics.meshing` importable when invoked directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from physics.meshing import mesh_graph
from leia.regime import MICROPLATE_LATTICE


# Canonical defaults — match the meshes used for the paper and consumed
# by every downstream training/FEM pipeline. Tile count comes from the
# microplate_lattice manifest; res_per_cell is a meshing-tool param fixed
# at 30 (verified to produce 309,806 nodes on mg_046).
CANONICAL_TILES = MICROPLATE_LATTICE.tiles
CANONICAL_RES_PER_CELL = 30


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument('--all', action='store_true',
                     help='Build meshes for every plate under --graph-dir.')
    sel.add_argument('--plates', nargs='+',
                     help='One or more plate IDs (e.g. mg_046 mg_111).')
    p.add_argument('--graph-dir', default='data/rve_bank',
                   help='Directory holding {plate_id}/graph.json (default: %(default)s).')
    p.add_argument('--output-dir', default='data/meshes/microplate_lattice',
                   help='Output directory (default: %(default)s).')
    p.add_argument('--tiles', default=','.join(str(x) for x in CANONICAL_TILES),
                   help='Tile counts per axis (default: %(default)s — from manifest).')
    p.add_argument('--res-per-cell', type=int, default=CANONICAL_RES_PER_CELL,
                   help='SDF grid resolution per unit cell (default: %(default)d, '
                        'matches canonical artifact node count exactly).')
    p.add_argument('--skip-existing', action='store_true', default=True,
                   help='Skip plates that already have mesh.npz.')
    p.add_argument('--no-skip-existing', dest='skip_existing', action='store_false')
    p.add_argument('--tetgen-timeout', type=int, default=300,
                   help='Per-plate TetGen timeout in seconds (default: %(default)d).')
    p.add_argument('--gpu-id', type=int, default=None,
                   help='GPU for the SDF subprocess (default: inherit CUDA_VISIBLE_DEVICES).')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Mesh validation (plate-scale)
# ---------------------------------------------------------------------------

def validate_plate_mesh(points, cells, tiles, face_tol=0.05):
    """Check single connected component + all 6 faces present at plate scale.

    Plate-scale boundary planes are at ±tiles[i]*0.98. Face presence is
    checked with a generous tolerance to accommodate TetGen non-determinism
    on coord values.
    """
    n_pts = len(points)

    # Connectivity: build a sparse adjacency from tet edges.
    edges = set()
    for tet in cells:
        for a in range(4):
            for b in range(a + 1, 4):
                u, v = int(tet[a]), int(tet[b])
                if u > v:
                    u, v = v, u
                edges.add((u, v))
    if not edges:
        return False, 'no edges'

    rows, cols = zip(*edges)
    rows, cols = list(rows), list(cols)
    data = np.ones(len(rows) * 2, dtype=np.int8)
    row_idx = np.array(rows + cols)
    col_idx = np.array(cols + rows)
    adj = csr_matrix((data, (row_idx, col_idx)), shape=(n_pts, n_pts))
    n_components, _ = connected_components(adj, directed=False)
    if n_components > 1:
        return False, f'{n_components} disconnected components'

    # Face presence: each face plane is at ±tiles[i] * 0.98.
    face_names = ['-X', '+X', '-Y', '+Y', '-Z', '+Z']
    missing = []
    for axis in range(3):
        face_extent = tiles[axis] * 0.98
        if not np.any(points[:, axis] < -face_extent + face_tol):
            missing.append(face_names[axis * 2])
        if not np.any(points[:, axis] > face_extent - face_tol):
            missing.append(face_names[axis * 2 + 1])
    if missing:
        return False, f"missing faces: {','.join(missing)}"
    return True, 'ok'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_plates(graph_dir):
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(graph_dir, 'mg_*'))
                  if os.path.isfile(os.path.join(p, 'graph.json')))


def main():
    args = parse_args()
    tiles = tuple(int(x) for x in args.tiles.split(','))
    if len(tiles) != 3:
        sys.exit(f'--tiles must be 3 ints; got {args.tiles}')

    plate_ids = discover_plates(args.graph_dir) if args.all else args.plates
    if not plate_ids:
        sys.exit(f'No plates found under {args.graph_dir}/mg_*/graph.json')

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Building meshes for {len(plate_ids)} plates → {args.output_dir}')
    print(f'  tiles={tiles}, res_per_cell={args.res_per_cell}, '
          f'tetgen_timeout={args.tetgen_timeout}s\n')

    n_ok = n_fail = n_skip = 0
    t_start = time.time()

    for i, plate_id in enumerate(plate_ids, 1):
        graph_path = os.path.join(args.graph_dir, plate_id, 'graph.json')
        if not os.path.isfile(graph_path):
            print(f'  [{i}/{len(plate_ids)}] {plate_id}: SKIP (no graph.json)')
            n_skip += 1
            continue

        out_dir = os.path.join(args.output_dir, plate_id)
        out_path = os.path.join(out_dir, 'mesh.npz')
        if args.skip_existing and os.path.exists(out_path):
            print(f'  [{i}/{len(plate_ids)}] {plate_id}: skip (exists)')
            n_skip += 1
            continue

        with open(graph_path) as f:
            graph = json.load(f)

        try:
            t0 = time.time()
            result = mesh_graph(
                graph,
                tiles=tiles,
                res_per_cell=args.res_per_cell,
                tetgen_kill_on_timeout=True,
                timeout=args.tetgen_timeout,
                extract_surface=False,
                gpu_id=args.gpu_id,
            )
            pts, cells = result.points, result.cells

            ok, reason = validate_plate_mesh(pts, cells, tiles)
            if not ok:
                print(f'  [{i}/{len(plate_ids)}] {plate_id}: FAIL ({reason})')
                n_fail += 1
                continue

            os.makedirs(out_dir, exist_ok=True)
            # Atomic: --skip-existing trusts out_path. The temp name does not
            # end in .npz, so it is written through a file object (np.savez
            # would append '.npz').
            tmp_path = out_path + '.partial'
            with open(tmp_path, 'wb') as fh:
                np.savez_compressed(fh,
                                    points=pts.astype(np.float64),
                                    cells=cells.astype(np.int64))
            os.replace(tmp_path, out_path)
            n_ok += 1
            print(f'  [{i}/{len(plate_ids)}] {plate_id}: ok '
                  f'({len(pts)} pts, {len(cells)} tets, '
                  f'{time.time() - t0:.1f}s)')
        except Exception as e:
            print(f'  [{i}/{len(plate_ids)}] {plate_id}: FAIL ({type(e).__name__}: {e})')
            n_fail += 1

    elapsed = time.time() - t_start
    print(f'\nDone in {elapsed:.0f}s — ok={n_ok}, fail={n_fail}, skip={n_skip}')
    if n_fail:
        sys.exit(f'build_meshes.py: {n_fail} plate(s) failed (FAIL lines above); '
                 f'not exiting 0, because the FEM runner would otherwise build data '
                 f'for the remaining plates only.')


if __name__ == '__main__':
    main()
