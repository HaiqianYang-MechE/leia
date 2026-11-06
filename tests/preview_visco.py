"""Render a stored Arruda-Boyce visco trajectory as MP4 + .vtu series.

Reads from data/microplate_visco/microplate_visco.h5 (precomputed by
data_curation/fem/run_visco_load_reverse.py) so the preview runs in the
leia env without needing fenicsx.

Outputs (in tests/_artifacts/visco/):
  frame_NNN.vtu    deformed mesh + displacement
  frame_NNN.png    deformed surface colored by displacement magnitude
  visco_<traj>.mp4 stitched animation
  stats.json       traj index, frame counts
"""

import argparse
import json
import os
import subprocess

import h5py
import meshio
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True,
                    help="microplate_visco.h5 path")
    ap.add_argument("--group", default="solid_plate_visco")
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--n-frames", type=int, default=10)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with h5py.File(args.h5, "r") as f:
        grp = f[args.group]
        coords = grp["coords"][...].astype(np.float64)              # (N, 3)
        topology = grp["topology"][...].astype(np.int64)            # (M, 4)
        disp = grp["displacements"][args.traj].astype(np.float32)   # (T+1, N, 3)
        inputs = grp["inputs"][args.traj]                           # (T, 4)

    T = disp.shape[0]
    N = coords.shape[0]
    print(f"  group={args.group} traj={args.traj} N={N:,} M={topology.shape[0]:,} T={T}")

    frame_idx = np.linspace(0, T - 1, args.n_frames).astype(int)
    print(f"  rendering frames: {frame_idx.tolist()}")

    # Clamp away from zero — see preview_fem.py for rationale.
    vmax = max(
        float(max(np.linalg.norm(disp[t], axis=1).max() for t in frame_idx)),
        1e-6,
    )
    print(f"  vmax (disp magnitude) = {vmax:.4f}")

    vtu_sample = {0, len(frame_idx) // 2, len(frame_idx) - 1}

    for k, t in enumerate(frame_idx):
        u = disp[t]
        deformed = coords + u
        mag = np.linalg.norm(u, axis=1)

        if k in vtu_sample:
            meshio.write(
                os.path.join(args.out_dir, f"frame_{k:03d}.vtu"),
                meshio.Mesh(
                    points=coords,
                    cells=[("tetra", topology)],
                    point_data={"displacement": u, "disp_mag": mag},
                ),
                binary=True, compression="zlib",
            )

        grid = pv.UnstructuredGrid({pv.CellType.TETRA: topology}, deformed)
        grid.point_data["disp_mag"] = mag
        surf = grid.extract_surface()
        p = pv.Plotter(off_screen=True, window_size=(720, 720))
        p.set_background("white")
        p.add_mesh(surf, scalars="disp_mag", clim=[0, vmax], cmap="plasma",
                   smooth_shading=True, show_edges=False)
        p.add_text(
            f"visco traj {args.traj}  frame {t}/{T-1}  "
            f"action={inputs[min(t, T-2)].tolist()}",
            font_size=10, color="black",
        )
        p.camera_position = "iso"
        p.camera.zoom(0.85)
        p.show(screenshot=os.path.join(args.out_dir, f"frame_{k:03d}.png"))
        p.close()
        print(f"    frame {k+1}/{len(frame_idx)} (t={t})")

    mp4_path = os.path.join(args.out_dir, f"visco_traj{args.traj:03d}.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(args.fps),
        "-i", os.path.join(args.out_dir, "frame_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        mp4_path,
    ], check=True, capture_output=True)
    print(f"  wrote {mp4_path}")

    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump({
            "group": args.group,
            "traj": int(args.traj),
            "n_frames_rendered": len(frame_idx),
            "n_frames_total": int(T),
            "n_points": int(N),
            "n_cells": int(topology.shape[0]),
            "vmax_disp": vmax,
        }, f, indent=2)

    print("PASS preview_visco")


if __name__ == "__main__":
    main()
