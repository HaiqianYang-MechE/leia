"""Generate a small lattice mesh and save to NPZ.

Runs in the leia env. Companion to preview_fem_solve.sh — separated from
the stage-1 heredoc because mesh_graph spawns an SDF subprocess via
multiprocessing.spawn, which re-imports the main module from __file__
(stdin breaks this).
"""
import argparse
import json

import numpy as np

from physics.meshing import mesh_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-json", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--res-per-cell", type=int, default=10)
    args = ap.parse_args()

    with open(args.graph_json) as f:
        graph = json.load(f)
    result = mesh_graph(graph, tiles=(5, 5, 1), res_per_cell=args.res_per_cell)
    np.savez(args.out_npz, points=result.points, cells=result.cells)
    print(f"  N={result.points.shape[0]:,} M={result.cells.shape[0]:,}")
    print("PASS preview_fem_meshgen")


if __name__ == "__main__":
    main()
