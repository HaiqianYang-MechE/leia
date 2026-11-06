#!/usr/bin/env python
"""Compute GT force-displacement curves for the lattice plate FEM dataset.

Reads displacement trajectories from per-plate NPZ, computes per-frame
generalized force [Fx, Mx, Fy, Fz] via the numpy stress pipeline, generalized
displacement [dx, theta, dy, dz] from discrete actions, and cumulative external
work via the midpoint rule.

Uses boundary-only optimization: computes stress only for tets that touch the
right boundary instead of the whole mesh. The first trajectory per plate is
verified against the full-mesh approach.

Usage:
    python scripts/eval.py gt-fd \\
        --npz-dir  data/microplate_lattice/npz \\
        --mesh-dir data/meshes/microplate_lattice \\
        --output   outputs/leia_lattice/fd_curves_gt.h5
"""
import argparse
import datetime
import glob
import json
import os
import time

import h5py
import numpy as np

from leia.regime import MICROPLATE_LATTICE
from leia.stress import (
    precompute_stress_geometry,
    stress_from_disps,
    assemble_internal_force,
    identify_right_boundary_nodes,
    generalized_force_from_f_int,
    precompute_boundary_geometry,
    boundary_reaction_force,
    compute_gen_disp,
)
SPEED_S, SPEED_T, SPEED_SH, _ = MICROPLATE_LATTICE.speeds


def compute_work(gen_force, gen_disp):
    """Compute cumulative external work using the midpoint rule.

    Parameters
    ----------
    gen_force : (T+1, 4) — generalized force per frame
    gen_disp : (T+1, 4) — generalized displacement per frame

    Returns
    -------
    work : (T+1,) — cumulative external work, frame 0 = 0
    """
    delta_disp = np.diff(gen_disp, axis=0)  # (T, 4)
    mid_force = 0.5 * (gen_force[1:] + gen_force[:-1])  # (T, 4)
    delta_work = np.sum(mid_force * delta_disp, axis=1)  # (T,)
    work = np.zeros(gen_force.shape[0], dtype=np.float64)
    work[1:] = np.cumsum(delta_work)
    return work


def verify_boundary_optimization(coords, topo, dNdx, volumes, good,
                                 right_node_ids, bdy_geom, cy, cz,
                                 outputs, mu, lmbda):
    """Verify boundary-only approach matches full-mesh on representative frames.

    `outputs` is (n_frames, N, 3) — the displacement trajectory for one
    trajectory of this plate. Returns max relative error across components.
    """
    n_nodes = len(coords)
    n_frames = outputs.shape[0]
    max_rel_err = 0.0

    print("    Verifying boundary optimization against full-mesh...")
    t0 = time.time()
    test_frames = [f for f in (5, 10, 15, 20, 25) if f < n_frames]
    for frame in test_frames:
        u_disp = outputs[frame].astype(np.float64)
        _, P, _, _ = stress_from_disps(
            topo, u_disp, dNdx, volumes, good, mu=mu, lmbda=lmbda)
        f_int_full = assemble_internal_force(P, dNdx, volumes, topo, n_nodes)
        gf_full = generalized_force_from_f_int(
            f_int_full, right_node_ids, coords, cy, cz)
        gf_bdy = boundary_reaction_force(
            u_disp, coords, n_nodes, right_node_ids, cy, cz,
            bdy_geom, mu=mu, lmbda=lmbda)
        scale = np.maximum(np.abs(gf_full).max(), 1e-12)
        rel_err = np.abs(gf_full - gf_bdy).max() / scale
        max_rel_err = max(max_rel_err, rel_err)

    print(f"    Verification: max relative error = {max_rel_err:.2e} "
          f"({time.time() - t0:.1f}s)")
    if max_rel_err > 1e-3:
        print(f"    WARNING: boundary optimization error {max_rel_err:.2e} > 1e-3")
    return max_rel_err


def main():
    parser = argparse.ArgumentParser(
        description='Compute GT force-displacement curves for lattice plates')
    parser.add_argument('--npz-dir', required=True,
                        help='Directory with {plate_id}/traj_*.npz')
    parser.add_argument('--mesh-dir', required=True,
                        help='Directory with {plate_id}/mesh.npz for topology')
    parser.add_argument('--curated-plates', default=None,
                        help='JSON manifest with "plates" array '
                             '(default: every subdir of --npz-dir)')
    parser.add_argument('--max-trajs', type=int, default=100,
                        help='Max trajectories per plate')
    parser.add_argument('--output', required=True,
                        help='Output HDF5 path for the F-D curves')
    parser.add_argument('--mu', type=float, default=MICROPLATE_LATTICE.mu,
                        help='Neo-Hookean mu (default from microplate_lattice manifest)')
    parser.add_argument('--lmbda', type=float, default=MICROPLATE_LATTICE.lmbda,
                        help='Neo-Hookean lambda (default from microplate_lattice manifest)')
    parser.add_argument('--plates', nargs='*', default=None,
                        help='Specific plates to process (default: all)')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip boundary optimization verification')
    args = parser.parse_args()

    print(f"NPZ:      {args.npz_dir}")
    print(f"Meshes:   {args.mesh_dir}")
    print(f"Output:   {args.output}")
    print(f"Material: mu={args.mu}, lambda={args.lmbda}")

    if args.curated_plates:
        with open(args.curated_plates) as f:
            all_plates = sorted(json.load(f)['plates'])
    else:
        all_plates = sorted(d for d in os.listdir(args.npz_dir)
                            if os.path.isdir(os.path.join(args.npz_dir, d)))
    if args.plates:
        unknown = sorted(set(args.plates) - set(all_plates))
        if unknown:
            raise SystemExit(f"gt-fd: unknown plate(s) {unknown}; "
                             f"available: {all_plates}")
        plates = [p for p in args.plates if p in all_plates]
    else:
        plates = all_plates
    print(f"Plates:   {len(plates)} plates")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    t_total = time.time()
    total_traj = 0

    with h5py.File(args.output, 'w') as out_h5:
        out_h5.attrs['mu'] = args.mu
        out_h5.attrs['lmbda'] = args.lmbda
        out_h5.attrs['speed_s'] = SPEED_S
        out_h5.attrs['speed_t'] = SPEED_T
        out_h5.attrs['speed_sh'] = SPEED_SH
        out_h5.attrs['bc_eps'] = 1e-2
        out_h5.attrs['timestamp'] = datetime.datetime.now().isoformat()
        out_h5.attrs['gen_force_columns'] = 'Fx,Mx,Fy,Fz'
        out_h5.attrs['gen_disp_columns'] = 'dx,theta,dy,dz'

        for plate_id in plates:
            traj_dir = os.path.join(args.npz_dir, plate_id)
            traj_files = sorted(glob.glob(os.path.join(traj_dir, 'traj_*.npz')))
            if args.max_trajs > 0:
                traj_files = traj_files[:args.max_trajs]
            n_traj = len(traj_files)
            if n_traj == 0:
                continue

            mesh_path = os.path.join(args.mesh_dir, plate_id, 'mesh.npz')
            with np.load(mesh_path) as m:
                topo = m['cells'].astype(np.intp)
            first = np.load(traj_files[0])
            coords = first['coords'].astype(np.float64)
            n_nodes = coords.shape[0]
            n_frames = first['outputs'].shape[0]

            print(f"\n  {plate_id}: {n_nodes:,} nodes, {len(topo):,} tets, "
                  f"{n_traj} traj, {n_frames} frames", flush=True)

            t0 = time.time()
            dNdx, volumes, good = precompute_stress_geometry(
                coords, topo, quality_threshold=1e-4)
            right_node_ids, x_max = identify_right_boundary_nodes(coords, eps=1e-2)
            cy = (coords[:, 1].min() + coords[:, 1].max()) / 2.0
            cz = (coords[:, 2].min() + coords[:, 2].max()) / 2.0
            bdy_geom = precompute_boundary_geometry(
                topo, dNdx, volumes, right_node_ids)
            print(f"    Geometry: {time.time() - t0:.1f}s, "
                  f"{len(right_node_ids)} right nodes", flush=True)

            if not args.no_verify:
                verify_boundary_optimization(
                    coords, topo, dNdx, volumes, good, right_node_ids,
                    bdy_geom, cy, cz,
                    first['outputs'].astype(np.float64), args.mu, args.lmbda)

            all_gen_force = np.zeros((n_traj, n_frames, 4), dtype=np.float64)
            all_gen_disp = np.zeros((n_traj, n_frames, 4), dtype=np.float64)
            all_work = np.zeros((n_traj, n_frames), dtype=np.float64)

            for ti, tf in enumerate(traj_files):
                data = np.load(tf)
                inputs = data['inputs']  # (30, 4)
                outputs = data['outputs'].astype(np.float64)  # (31, N, 3)

                all_gen_disp[ti] = compute_gen_disp(inputs, MICROPLATE_LATTICE.speeds)
                for frame in range(1, n_frames):
                    all_gen_force[ti, frame] = boundary_reaction_force(
                        outputs[frame], coords, n_nodes, right_node_ids,
                        cy, cz, bdy_geom, mu=args.mu, lmbda=args.lmbda)
                all_work[ti] = compute_work(
                    all_gen_force[ti], all_gen_disp[ti])

                if (ti + 1) % 10 == 0 or ti == 0:
                    print(f"    Traj {ti + 1}/{n_traj}", flush=True)

            grp = out_h5.create_group(plate_id)
            grp.create_dataset('gen_force', data=all_gen_force)
            grp.create_dataset('gen_disp', data=all_gen_disp)
            grp.create_dataset('work', data=all_work)
            grp.create_dataset('right_node_ids', data=right_node_ids)
            grp.attrs['cy'] = cy
            grp.attrs['cz'] = cz
            grp.attrs['x_max'] = x_max
            grp.attrs['n_right_nodes'] = len(right_node_ids)
            grp.attrs['n_boundary_tets'] = len(bdy_geom['tet_indices'])
            grp.attrs['n_total_tets'] = len(topo)
            total_traj += n_traj
            print(f"    Done: {n_traj} traj", flush=True)

    elapsed = time.time() - t_total
    print(f"\nTotal: {total_traj} trajectories in {elapsed:.1f}s "
          f"({elapsed / 60:.1f}min)")
    print(f"Output: {args.output}")


if __name__ == '__main__':
    main()
