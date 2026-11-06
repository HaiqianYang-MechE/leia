"""Run the live Neo-Hookean solver on a small mesh and save the trajectory.

Runs in the `leia` env (dolfin / FEniCS 2019.1 is installed there). Inputs are pre-built (mesh
generated via mesh_graph in the leia env), outputs are an NPZ that the
leia-env render path consumes.

Outputs (to --out-npz):
  coords     (N, 3) float32 — reference coords
  topology   (M, 4) int64
  outputs    (T+1, N, 3) float32 — per-frame displacement (frame 0 = zeros)
  inputs     (T, 4) int8 — action sequence (replays for the renderer overlay)
  mesh_name  ()
"""

import argparse
import sys

import numpy as np

from physics.fem import solve_neo_hookean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-npz", required=True,
                    help="Pre-built mesh NPZ with `points`, `cells`.")
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--n-steps", type=int, default=5)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--lmbda", type=float, default=10.0)
    args = ap.parse_args()

    mesh = np.load(args.mesh_npz)
    coords = mesh["points"].astype(np.float64)
    topo = mesh["cells"].astype(np.intp)
    print(f"  mesh: N={coords.shape[0]:,}  M={topo.shape[0]:,}", flush=True)

    # Pure stretch action sequence: 5 steps of [+1, 0, 0, 0].
    actions = np.zeros((args.n_steps, 4), dtype=np.int8)
    actions[:, 0] = 1
    print(f"  actions: {args.n_steps} steps of stretch (+1,0,0,0)", flush=True)

    print(f"  running solve_neo_hookean(mu={args.mu}, lmbda={args.lmbda})...",
          flush=True)
    result = solve_neo_hookean(coords, topo, actions,
                               mu=args.mu, lmbda=args.lmbda)
    print(f"  status: {result['status']}  time: {result['time_seconds']:.1f}s",
          flush=True)
    if result["status"] != "ok":
        print(f"FAIL preview_fem_solve: solver status={result['status']}",
              flush=True)
        sys.exit(1)

    np.savez(
        args.out_npz,
        coords=coords.astype(np.float32),
        topology=topo.astype(np.int64),
        outputs=result["displacements"].astype(np.float32),
        inputs=actions,
        mesh_name=np.array("live_solve", dtype="<U10"),
    )
    print(f"  wrote {args.out_npz}", flush=True)
    print("PASS preview_fem_solve")


if __name__ == "__main__":
    main()
