"""Preview the graph -> tet mesh pipeline on N production plates.

Called by tests/preview_meshing.sh. Lives as a real .py file because
mesh_graph spawns an SDF subprocess via multiprocessing.spawn, which
re-imports the main module from __file__ — heredoc stdin breaks this.

Outputs (per plate, in tests/_artifacts/meshing/):
  <plate>.vtu    ParaView-loadable tet mesh
  <plate>.png    isometric snapshot
  <plate>.json   stats (n_points, n_cells, runtime)
"""

import argparse
import json
import os
import time

import meshio
import numpy as np
import pyvista as pv

from physics.meshing import mesh_graph

pv.OFF_SCREEN = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates", nargs="+", required=True)
    ap.add_argument("--rve-bank", default="data/rve_bank")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--res-per-cell", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for pid in args.plates:
        graph_path = os.path.join(args.rve_bank, pid, "graph.json")
        if not os.path.exists(graph_path):
            print(f"  SKIP {pid}: missing {graph_path}")
            continue
        with open(graph_path) as f:
            graph = json.load(f)

        t0 = time.time()
        result = mesh_graph(graph, tiles=(5, 5, 1),
                            res_per_cell=args.res_per_cell)
        dt = time.time() - t0

        meshio.write(
            os.path.join(args.out_dir, f"{pid}.vtu"),
            meshio.Mesh(points=result.points,
                        cells=[("tetra", result.cells)]),
        )

        grid = pv.UnstructuredGrid(
            {pv.CellType.TETRA: result.cells.astype(np.int64)},
            result.points.astype(np.float64),
        )
        surf = grid.extract_surface()
        p = pv.Plotter(off_screen=True, window_size=(720, 720))
        p.set_background("white")
        p.add_mesh(surf, color="#3a7bd5", smooth_shading=True,
                   show_edges=True, edge_color="#1f3a5f", line_width=0.4)
        p.camera_position = "iso"
        p.camera.zoom(0.9)
        p.show(screenshot=os.path.join(args.out_dir, f"{pid}.png"))
        p.close()

        stats = {
            "plate_id": pid,
            "n_points": int(result.points.shape[0]),
            "n_cells": int(result.cells.shape[0]),
            "tiles": [5, 5, 1],
            "res_per_cell": args.res_per_cell,
            "runtime_sec": round(dt, 2),
        }
        with open(os.path.join(args.out_dir, f"{pid}.json"), "w") as f:
            json.dump(stats, f, indent=2)

        print(f"  OK {pid}: n_points={stats['n_points']:,} "
              f"n_cells={stats['n_cells']:,} ({dt:.1f}s)")

    print("PASS preview_meshing")


if __name__ == "__main__":
    main()
