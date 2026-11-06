"""Quasi-static Neo-Hookean solver — drives an action sequence through Newton.

Source-of-truth alignment: matches data_curation/fem/run_plate_fem.py:_solve_one_trajectory
byte-equivalently. The two pre-extraction call sites (data_curation and
design_search) had identical FEniCS forms, parameters, and loop structure;
this module is the lifted shared kernel.
"""

import time

import numpy as np
import dolfin as df

from ._primitives import (
    build_dolfin_mesh,
    build_dirichlet_bcs,
    build_neohookean_problem,
    get_mesh_geometry,
    vertex_displacements,
    ControlExpression,
)


def solve_neo_hookean(
    coords,
    topo,
    actions,
    *,
    mu,
    lmbda,
    speeds=(0.15, 0.08, 0.15, 0.15),
    compute_stress=False,
):
    """Run the quasi-static Neo-Hookean solver under a 4-DOF action sequence.

    Parameters
    ----------
    coords : (N, 3) float64 — reference node coordinates
    topo : (M, 4) intp — tet connectivity
    actions : (T, 4) int or float — per-step action [stretch, twist, sy, sz]
        In data_curation, values are random {-1, 0, +1}; in design_search,
        one-hot per direction. Both produce identical solver behavior.
    mu, lmbda : float — Neo-Hookean material parameters
    speeds : 4-tuple — speed per DOF; default matches MICROPLATE_LATTICE_SPEEDS
    compute_stress : bool — if True, also compute volume-weighted nodal PK1 +
        per-node von Mises stress per frame

    Returns
    -------
    dict with keys:
      'status'         : 'ok' | 'diverged' | 'failed'
      'displacements'  : (T+1, N, 3) float32 — frame 0 = zeros
      'time_seconds'   : float
      'steps_completed': int (only when status='diverged')
      'error'          : str (only when status != 'ok')
      'stress_P'       : (T+1, N, 3, 3) float32, only if compute_stress and status='ok'
      'stress_vm'      : (T+1, N) float32, only if compute_stress and status='ok'
    """
    df.set_log_level(df.LogLevel.ERROR)

    coords = np.asarray(coords, dtype=np.float64)
    topo = np.asarray(topo, dtype=np.intp)
    actions = np.asarray(actions)
    speeds = tuple(float(s) for s in speeds)
    n_steps = len(actions)
    n_nodes = len(coords)

    t_start = time.time()

    try:
        mesh = build_dolfin_mesh(coords, topo)
        x_min, x_max, cy, cz = get_mesh_geometry(coords)

        V = df.VectorFunctionSpace(mesh, "CG", 1)

        boundary_expr = ControlExpression(cy=cy, cz=cz, degree=1)
        bcs = build_dirichlet_bcs(V, x_min, x_max, boundary_expr)

        u = df.Function(V)
        problem, solver = build_neohookean_problem(V, u, bcs, mu, lmbda)
    except Exception as e:
        return {
            "status": "failed",
            "displacements": np.zeros((1, n_nodes, 3), dtype=np.float32),
            "time_seconds": round(time.time() - t_start, 2),
            "error": f"setup failed: {type(e).__name__}: {e}",
        }

    # Frame 0 — undeformed reference.
    trajectory = [vertex_displacements(u, mesh)]

    stress_P_list = []
    stress_vm_list = []
    if compute_stress:
        stress_P_list.append(np.zeros((n_nodes, 3, 3), dtype=np.float32))
        stress_vm_list.append(np.zeros(n_nodes, dtype=np.float32))

    for step in range(n_steps):
        a = actions[step]
        boundary_expr.stretch += float(a[0]) * speeds[0]
        boundary_expr.twist   += float(a[1]) * speeds[1]
        boundary_expr.sy      += float(a[2]) * speeds[2]
        boundary_expr.sz      += float(a[3]) * speeds[3]

        try:
            solver.solve()
        except RuntimeError:
            return {
                "status": "diverged",
                "displacements": np.array(trajectory),
                "time_seconds": round(time.time() - t_start, 2),
                "steps_completed": step,
                "error": f"Newton diverged at step {step}",
            }

        u_step = vertex_displacements(u, mesh)
        trajectory.append(u_step)

        if compute_stress:
            P_node, vm_node = _project_stress_to_nodes(
                coords, topo, u_step, mu, lmbda)
            stress_P_list.append(P_node)
            stress_vm_list.append(vm_node)

    out = {
        "status": "ok",
        "displacements": np.array(trajectory),
        "time_seconds": round(time.time() - t_start, 2),
    }
    if compute_stress:
        out["stress_P"] = np.array(stress_P_list, dtype=np.float32)
        out["stress_vm"] = np.array(stress_vm_list, dtype=np.float32)
    return out


def _project_stress_to_nodes(coords, topo, u_step, mu, lmbda):
    """Volume-weighted projection of per-tet PK1 to nodes, and per-node VM.

    Lifted byte-equivalently from data_curation/fem/run_plate_fem.py — the
    canonical stress-projection for training-data generation.
    """
    from leia.stress import compute_stress_on_merged_mesh

    _, P_tet, _, _, volumes, vm_node = compute_stress_on_merged_mesh(
        coords, topo, u_step,
        mu=mu, lmbda=lmbda, quality_threshold=1e-4,
    )
    n_nodes = len(coords)
    P_node = np.zeros((n_nodes, 3, 3), dtype=np.float64)
    vol_accum = np.zeros(n_nodes, dtype=np.float64)
    good = volumes > 0
    for j in range(4):
        np.add.at(
            P_node.reshape(n_nodes, 9),
            topo[good, j],
            (P_tet[good] * volumes[good, None, None]).reshape(-1, 9),
        )
        np.add.at(vol_accum, topo[good, j], volumes[good])
    vol_accum = np.maximum(vol_accum, 1e-30)
    P_node /= vol_accum[:, None, None]
    return P_node.astype(np.float32), vm_node.astype(np.float32)
