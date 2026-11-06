"""Run the live Arruda-Boyce 3-branch visco solver on a small mesh.

Runs in the `fenicsx` env (dolfinx 0.8.0). Saves a trajectory NPZ in the
same shape as preview_fem.py expects (coords, topology, outputs, inputs,
mesh_name) so the leia-env render path can ingest it.

Material parameters are the canonical v42 set from
data_curation/fem/run_visco_load_reverse.py.
"""
import argparse
import sys

import numpy as np

from physics.fem import solve_arruda_boyce


# v42 material — matches data_curation/fem/run_visco_load_reverse.py:MATERIAL.
MATERIAL = {
    "Geq_0":   200.0,
    "lambdaL": 10.0,
    "Kbulk":   400_000.0,
    "rho":     1.3e-5,
    "branches": [(300.0, 0.001), (600.0, 0.2), (150.0, 3.0)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--n-steps", type=int, default=3,
                    help="Action steps; FEM sub-steps controlled by n_substeps.")
    ap.add_argument("--n-substeps", type=int, default=2)
    ap.add_argument("--mesh-res", type=int, default=8,
                    help="Uniform tet resolution per axis.")
    args = ap.parse_args()

    # Pure stretch + small twist for visible deformation.
    actions = np.zeros((args.n_steps, 4), dtype=np.int8)
    actions[:, 0] = 1  # stretch
    print(f"  actions: {args.n_steps} steps stretch (n_substeps={args.n_substeps})",
          flush=True)
    print(f"  mesh resolution: ({args.mesh_res},{args.mesh_res},2)",
          flush=True)
    print(f"  running solve_arruda_boyce...", flush=True)

    result = solve_arruda_boyce(
        actions,
        material=MATERIAL,
        mesh_resolution=(args.mesh_res, args.mesh_res, 2),
        n_substeps=args.n_substeps,
        log_progress=False,
    )
    print(f"  status: {result['status']}  time: {result['time_seconds']:.1f}s",
          flush=True)
    if result["status"] != "ok":
        err = result.get("error", "")
        print(f"FAIL preview_visco_solve: status={result['status']} {err}",
              flush=True)
        sys.exit(1)

    np.savez(
        args.out_npz,
        coords=result["coords"].astype(np.float32),
        topology=result["topology"].astype(np.int64),
        outputs=result["displacements"].astype(np.float32),
        inputs=actions,
        mesh_name=np.array("visco_live", dtype="<U10"),
    )
    print(f"  wrote {args.out_npz}", flush=True)
    print("PASS preview_visco_solve")


if __name__ == "__main__":
    main()
