"""Render a stored FEM trajectory as MP4 + .vtu frame series.

Reads ground-truth FEM data from data/microplate_lattice/npz/<plate>/traj_*.npz
(produced offline by data_curation/fem/run_plate_fem.py) and produces a
visual preview without needing the fem conda env.

Outputs (in tests/_artifacts/fem/):
  frame_NNN.vtu    per-frame deformed mesh + displacement vectors
  frame_NNN.png    deformed mesh colored by displacement magnitude
  <plate>_<traj>.mp4  ffmpeg-compiled animation
"""

import argparse
import json
import os
import subprocess

import meshio
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True,
                    help="Path to traj_*.npz with coords, topology, outputs.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-frames", type=int, default=10,
                    help="Number of frames to render (sampled uniformly).")
    ap.add_argument("--fps", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = np.load(args.npz)
    coords = data["coords"].astype(np.float64)         # (N, 3)
    topology = data["topology"].astype(np.int64)       # (M, 4)
    outputs = data["outputs"].astype(np.float32)       # (T+1, N, 3)
    inputs = data["inputs"]                            # (T, 4)
    plate = str(data["mesh_name"])
    T = outputs.shape[0]
    N = coords.shape[0]
    print(f"  plate={plate} N={N:,} M={topology.shape[0]:,} T={T}")

    # Sample n_frames frames uniformly across the trajectory.
    frame_idx = np.linspace(0, T - 1, args.n_frames).astype(int)
    print(f"  rendering frames: {frame_idx.tolist()}")

    # Compute global vmax across selected frames for stable colormap.
    disp_mags = []
    for t in frame_idx:
        d = outputs[t]
        disp_mags.append(np.linalg.norm(d, axis=1).max())
    # Clamp away from zero — degenerate trajectories (e.g. all-zero frames
    # from a diverged solve) would otherwise give clim=[0, 0] and silently
    # produce a flat-color MP4.
    vmax = max(float(max(disp_mags)), 1e-6)
    print(f"  vmax (disp magnitude) = {vmax:.4f}")

    traj_tag = os.path.splitext(os.path.basename(args.npz))[0]

    # ParaView sample — write only first / middle / last as compressed .vtu
    # (309K-node tet mesh × 10 ASCII .vtu = ~300 MB; binary+zlib trims ~10x).
    vtu_sample_indices = {0, len(frame_idx) // 2, len(frame_idx) - 1}

    for k, t in enumerate(frame_idx):
        u = outputs[t]                                  # (N, 3)
        deformed = coords + u
        mag = np.linalg.norm(u, axis=1)

        if k in vtu_sample_indices:
            meshio.write(
                os.path.join(args.out_dir, f"frame_{k:03d}.vtu"),
                meshio.Mesh(
                    points=coords,
                    cells=[("tetra", topology)],
                    point_data={"displacement": u, "disp_mag": mag},
                ),
                binary=True, compression="zlib",
            )

        # PNG — deformed surface colored by disp magnitude
        grid = pv.UnstructuredGrid(
            {pv.CellType.TETRA: topology}, deformed
        )
        grid.point_data["disp_mag"] = mag
        surf = grid.extract_surface()
        p = pv.Plotter(off_screen=True, window_size=(720, 720))
        p.set_background("white")
        p.add_mesh(surf, scalars="disp_mag", clim=[0, vmax], cmap="viridis",
                   smooth_shading=True, show_edges=False)
        action_idx = min(t, inputs.shape[0] - 1)
        p.add_text(
            f"{plate}  frame {t}/{T-1}  action={inputs[action_idx].tolist()}",
            font_size=10, color="black",
        )
        p.camera_position = "iso"
        p.camera.zoom(0.85)
        p.show(screenshot=os.path.join(args.out_dir, f"frame_{k:03d}.png"))
        p.close()
        print(f"    frame {k+1}/{len(frame_idx)} (t={t})")

    # ffmpeg -> MP4
    mp4_path = os.path.join(args.out_dir, f"{plate}_{traj_tag}.mp4")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(args.fps),
        "-i", os.path.join(args.out_dir, "frame_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        mp4_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"  wrote {mp4_path}")

    # stats sidecar
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump({
            "plate": plate,
            "traj": traj_tag,
            "n_frames_rendered": len(frame_idx),
            "n_frames_total": int(T),
            "n_points": int(N),
            "n_cells": int(topology.shape[0]),
            "vmax_disp": vmax,
        }, f, indent=2)

    print("PASS preview_fem")


if __name__ == "__main__":
    main()
