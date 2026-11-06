#!/usr/bin/env python
"""
Parallel FEM on the 5×5×1 lattice plates of MicroPlate-lattice.

Compressible Neo-Hookean (mu=1, lmbda=10), left face clamped and right face
displacement-driven (Dirichlet), MUMPS solver, run via memory-gated ProcessPoolExecutor. Writes one
NPZ per trajectory plus a metadata sidecar; `--consolidate` additionally
packs everything into a single HDF5 (not consumed by the training/eval
pipeline).

Input meshes: `{plate_dir}/{plate_id}/mesh.npz` (keys `points`, `cells`),
as written by `data_curation/mesh/build_meshes.py`.
Output:       `{output_dir}/npz/{plate_id}/traj_{idx:03d}.npz`.

Usage:
    PYTHONHASHSEED=0 python data_curation/fem/run_plate_fem.py \
        --n-meshes 0 --trajs-per-mesh 500 --target-converged 110 \
        --mu 1.0 --lmbda 10.0 --workers 100 --output-dir data/microplate_lattice

Per-trajectory seeds derive from `hash(plate_name)`, which Python randomises
per process unless PYTHONHASHSEED is set; the runner refuses to start without
it. The paper's data was built with PYTHONHASHSEED=0.
"""

# BLAS/LAPACK thread control per worker (must be before numpy import).
# Workers in spawn context re-import this module; they inherit the parent's
# env, so _PLATE_FEM_OMP (set in main) controls per-worker thread count.
import os
_omp = os.environ.get("_PLATE_FEM_OMP", "1")
os.environ["OMP_NUM_THREADS"] = _omp
os.environ["MKL_NUM_THREADS"] = _omp
os.environ["OPENBLAS_NUM_THREADS"] = _omp

import sys
import time
import argparse
import json
import glob
import re
import multiprocessing
import threading
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import psutil

# Repo root on sys.path so `leia` and `physics` import when invoked directly
# (physics/ is not part of the pip package). Spawned workers re-import this
# module, so they get the same path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from leia.regime import MICROPLATE_LATTICE

# Default plate input directory: the output of data_curation/mesh/build_meshes.py.
# Override with --plate-dir if your meshes live elsewhere.
DEFAULT_PLATE_DIR = "data/meshes/microplate_lattice"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel FEM on lattice plates")
    parser.add_argument('--plate-dir', type=str, default=DEFAULT_PLATE_DIR,
                        help='Plate mesh directory (contains {plate_id}/mesh.npz)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: data/plate_fem/{timestamp})')
    parser.add_argument('--n-meshes', type=int, default=50,
                        help='Number of meshes (0=all)')
    parser.add_argument('--trajs-per-mesh', type=int, default=10,
                        help='Trajectories per mesh')
    parser.add_argument('--workers', type=int, default=200,
                        help='Number of parallel workers')
    parser.add_argument('--mem-limit', type=float, default=85.0,
                        help='Memory usage %% threshold to pause dispatch')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--consolidate', action='store_true',
                        help='After the run, also pack all NPZ into a single '
                             '{output_dir}/plate_dataset.h5. Off by default: '
                             'nothing in the training/eval pipeline reads it, '
                             'and at full scale it doubles the footprint')
    parser.add_argument('--consolidate-only', action='store_true',
                        help='Skip FEM, just consolidate existing NPZ to HDF5')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip trajectories that already have NPZ output')
    parser.add_argument('--no-skip-existing', dest='skip_existing',
                        action='store_false')
    parser.add_argument('--mu', type=float, default=MICROPLATE_LATTICE.mu,
                        help='Neo-Hookean mu (default from microplate_lattice manifest)')
    parser.add_argument('--lmbda', type=float, default=MICROPLATE_LATTICE.lmbda,
                        help='Neo-Hookean lambda (default from microplate_lattice manifest)')
    parser.add_argument('--compute-stress', action='store_true',
                        help='Compute and save PK1/VM stress per frame')
    parser.add_argument('--omp-threads', type=int, default=1,
                        help='OMP_NUM_THREADS per worker (default: 1)')
    parser.add_argument('--target-converged', type=int, default=0,
                        help='Stop dispatching tasks for a mesh after N converged (0=disabled)')
    parser.add_argument('--n-steps', type=int, default=NUM_STEPS,
                        help='Simulation steps per trajectory (default: %(default)d, '
                             'from the microplate_lattice manifest)')
    parser.add_argument('--plate-list', type=str, default=None,
                        help='JSON file with "plates" list to filter meshes')
    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join('data', 'plate_fem', timestamp)

    return args


# ---------------------------------------------------------------------------
# Mesh discovery
# ---------------------------------------------------------------------------

def get_plate_meshes(plate_dir, n_meshes, plate_list=None):
    """Find `{plate_id}/mesh.npz` files, return up to n_meshes (0 = all)."""
    pattern = os.path.join(plate_dir, "*", "mesh.npz")
    files = sorted(glob.glob(pattern))
    if plate_list:
        files = [f for f in files
                 if os.path.basename(os.path.dirname(f)) in plate_list]
    if not files:
        raise RuntimeError(
            f"No mesh.npz files found in {plate_dir}/*/ — build them with "
            f"`python data_curation/mesh/build_meshes.py --all`")
    if n_meshes > 0:
        files = files[:n_meshes]
    return files


def load_mesh_npz(mesh_path):
    """Load a canonical plate mesh written by build_meshes.py.

    Returns (coords float64 (N, 3), topo intp (M, 4)) — the dtypes
    `physics.fem.solve_neo_hookean` expects.
    """
    with np.load(mesh_path) as m:
        coords = m['points'].astype(np.float64)
        topo = m['cells'].astype(np.intp)
    return coords, topo


# ---------------------------------------------------------------------------
# Task list
# ---------------------------------------------------------------------------

def build_task_list(mesh_files, args):
    """Build flat list of (mesh_path, mesh_name, traj_idx, seed) tasks."""
    tasks = []
    npz_dir = os.path.join(args.output_dir, 'npz')

    for mesh_path in mesh_files:
        mesh_name = os.path.basename(os.path.dirname(mesh_path))
        for traj_idx in range(args.trajs_per_mesh):
            # Check if already exists
            if args.skip_existing:
                npz_path = os.path.join(npz_dir, mesh_name,
                                        f"traj_{traj_idx:03d}.npz")
                if os.path.exists(npz_path):
                    continue

            # Deterministic per-trajectory seed
            task_seed = args.seed * 100000 + hash(mesh_name) % 50000 + traj_idx
            task_seed = task_seed & 0x7FFFFFFF  # keep positive

            tasks.append({
                'mesh_path': mesh_path,
                'mesh_name': mesh_name,
                'traj_idx': traj_idx,
                'seed': task_seed,
                'output_dir': args.output_dir,
                'mu': args.mu,
                'lmbda': args.lmbda,
                'compute_stress': args.compute_stress,
                'n_steps': args.n_steps,
            })

    return tasks


# ---------------------------------------------------------------------------
# FEM driver — random 4-channel action sequence + Neo-Hookean kernel
# ---------------------------------------------------------------------------

NUM_STEPS = MICROPLATE_LATTICE.num_steps
PLATE_FEM_SPEEDS = MICROPLATE_LATTICE.speeds  # (stretch, twist, shear_y, shear_z)
SPEED_S, SPEED_T, SPEED_SH_Y, SPEED_SH_Z = PLATE_FEM_SPEEDS
# Two shear DOFs share the same speed in the canonical regime; checked here so a
# manifest edit that breaks symmetry surfaces immediately instead of silently
# falling out via SPEED_SH = SPEED_SH_Y.
assert SPEED_SH_Y == SPEED_SH_Z, "shear_y / shear_z speeds differ — pass them explicitly"
SPEED_SH = SPEED_SH_Y


def _solve_one_trajectory(task):
    """Run one FEM trajectory on a lattice plate mesh.

    Driver layer: reads mesh, generates the random multi-channel action
    sequence, calls `physics.fem.solve_neo_hookean` (which holds the
    shared FEniCS kernel), and writes the trajectory NPZ.
    """
    import random
    from physics.fem import solve_neo_hookean

    np.random.seed(task['seed'])
    random.seed(task['seed'])
    num_steps = task.get('n_steps', NUM_STEPS)
    time.sleep(random.uniform(0, 2.0))

    mesh_name = task['mesh_name']
    traj_idx = task['traj_idx']
    t_start = time.time()

    try:
        # 1. Read mesh (canonical mesh.npz from build_meshes.py)
        coords, topo = load_mesh_npz(task['mesh_path'])

        # 2. Generate 4 discrete action channels (random {-1, 0, +1} runs)
        def generate_actions():
            actions = np.zeros(num_steps, dtype=int)
            t = 0
            while t < num_steps:
                duration = np.random.randint(5, 15)
                end = min(t + duration, num_steps)
                state = np.random.choice([0, 1, -1])
                actions[t:end] = state
                t = end
            return actions

        s_acts = generate_actions()
        t_acts = generate_actions()
        sy_acts = generate_actions()
        sz_acts = generate_actions()
        input_sequence = np.stack(
            [s_acts, t_acts, sy_acts, sz_acts], axis=1)  # (T, 4) int

        # 3. Run the shared Neo-Hookean kernel
        result = solve_neo_hookean(
            coords, topo, input_sequence,
            mu=task['mu'], lmbda=task['lmbda'],
            speeds=PLATE_FEM_SPEEDS,
            compute_stress=task.get('compute_stress', False),
        )

        if result['status'] == 'diverged':
            return {
                'mesh_name': mesh_name,
                'traj_idx': traj_idx,
                'status': 'diverged',
                'steps_completed': result.get('steps_completed', 0),
                'time_seconds': time.time() - t_start,
                'error': result.get('error', 'Newton diverged'),
            }
        if result['status'] == 'failed':
            return {
                'mesh_name': mesh_name,
                'traj_idx': traj_idx,
                'status': 'error',
                'steps_completed': 0,
                'time_seconds': time.time() - t_start,
                'error': result.get('error', 'solver setup failed'),
            }

        trajectory_data = result['displacements']  # (T+1, N, 3) float32

        # 4. Save NPZ — same schema as before
        npz_dir = os.path.join(task['output_dir'], 'npz', mesh_name)
        os.makedirs(npz_dir, exist_ok=True)
        npz_path = os.path.join(npz_dir, f"traj_{traj_idx:03d}.npz")
        save_dict = dict(
            inputs=input_sequence,                  # (T, 4)
            outputs=trajectory_data,                # (T+1, N, 3)
            coords=coords.astype(np.float32),       # (N, 3)
            topology=topo,                          # (M, 4)
            mesh_name=mesh_name,
        )
        if task.get('compute_stress'):
            save_dict['P_nodes'] = result['stress_P']    # (T+1, N, 3, 3)
            save_dict['vm_nodes'] = result['stress_vm']  # (T+1, N)
        # Write to a temp name and rename, so an interrupted run never leaves a
        # truncated traj_*.npz that a resumed run (or any consumer listing
        # *.npz) would pick up. The temp name does not end in .npz, so it is
        # written through a file object (np.savez would append '.npz').
        tmp_path = npz_path + '.partial'
        with open(tmp_path, 'wb') as fh:
            np.savez(fh, **save_dict)
        os.replace(tmp_path, npz_path)

        return {
            'mesh_name': mesh_name,
            'traj_idx': traj_idx,
            'status': 'ok',
            'steps_completed': num_steps,
            'time_seconds': time.time() - t_start,
            'error': None,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'mesh_name': mesh_name,
            'traj_idx': traj_idx,
            'status': 'error',
            'steps_completed': 0,
            'time_seconds': time.time() - t_start,
            'error': str(e),
        }


# ---------------------------------------------------------------------------
# Consolidation: NPZ → HDF5
# ---------------------------------------------------------------------------

def consolidate_to_hdf5(output_dir):
    """Pack all NPZ files into a single HDF5 dataset."""
    import h5py

    npz_dir = os.path.join(output_dir, 'npz')
    h5_path = os.path.join(output_dir, 'plate_dataset.h5')

    # Glob all NPZ files, group by mesh_name
    npz_files = sorted(glob.glob(os.path.join(npz_dir, '*', 'traj_*.npz')))
    if not npz_files:
        print("  No NPZ files found to consolidate.")
        return None

    # Group by mesh_name
    mesh_groups = {}
    for npz_path in npz_files:
        mesh_name = os.path.basename(os.path.dirname(npz_path))
        mesh_groups.setdefault(mesh_name, []).append(npz_path)

    print(f"  Consolidating {len(npz_files)} NPZ files "
          f"from {len(mesh_groups)} meshes → {h5_path}")

    with h5py.File(h5_path, 'w') as f:
        for mesh_name in sorted(mesh_groups):
            paths = sorted(mesh_groups[mesh_name])
            # Load first to get shapes and detect stress arrays
            first = np.load(paths[0])
            coords = first['coords']       # [N, 3]
            topology = first['topology']    # [M, 4]
            n_nodes = coords.shape[0]
            has_stress = 'P_nodes' in first

            all_inputs = []
            all_outputs = []
            all_P = [] if has_stress else None
            all_vm = [] if has_stress else None
            for p in paths:
                data = np.load(p)
                all_inputs.append(data['inputs'])    # [30, 4]
                all_outputs.append(data['outputs'])  # [31, N, 3]
                if has_stress:
                    all_P.append(data['P_nodes'])    # [31, N, 3, 3]
                    all_vm.append(data['vm_nodes'])  # [31, N]

            inputs = np.stack(all_inputs, axis=0)    # [T, 30, 4]
            outputs = np.stack(all_outputs, axis=0)  # [T, 31, N, 3]

            grp = f.create_group(mesh_name)
            grp.create_dataset('coords', data=coords)
            grp.create_dataset('topology', data=topology)
            grp.create_dataset('inputs', data=inputs)
            grp.create_dataset('outputs', data=outputs)
            if has_stress:
                P_nodes = np.stack(all_P, axis=0)    # [T, 31, N, 3, 3]
                vm_nodes = np.stack(all_vm, axis=0)  # [T, 31, N]
                grp.create_dataset('P_nodes', data=P_nodes)
                grp.create_dataset('vm_nodes', data=vm_nodes)
                print(f"    {mesh_name}: {len(paths)} traj, "
                      f"{n_nodes} nodes (with stress)")

        f.attrs['n_meshes'] = len(mesh_groups)
        f.attrs['n_trajectories'] = len(npz_files)
        f.attrs['created'] = datetime.now().isoformat()

    size_gb = os.path.getsize(h5_path) / (1024 ** 3)
    print(f"  HDF5 written: {h5_path} ({size_gb:.1f} GB, "
          f"{len(mesh_groups)} meshes, {len(npz_files)} trajectories)")
    return h5_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if not args.consolidate_only:
        hashseed = os.environ.get('PYTHONHASHSEED')
        if hashseed is None:
            sys.exit("run_plate_fem.py: set PYTHONHASHSEED (the paper build used "
                     "PYTHONHASHSEED=0). Trajectory seeds derive from hash(plate_name), "
                     "which Python randomises per process otherwise, so the build would "
                     "not be reproducible.")
        if hashseed != '0':
            print(f"WARNING: PYTHONHASHSEED={hashseed}; the run is reproducible, but its "
                  f"trajectory seeds differ from the paper build (PYTHONHASHSEED=0).",
                  flush=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Plate FEM Pipeline — Lattice Plates")
    print("=" * 60)

    if args.consolidate_only:
        print(f"  Consolidate-only mode: {args.output_dir}")
        consolidate_to_hdf5(args.output_dir)
        return

    # Discover meshes
    plate_list = None
    if args.plate_list:
        with open(args.plate_list) as f:
            plate_list = set(json.load(f)['plates'])
        print(f"  Plate list: {args.plate_list} ({len(plate_list)} plates)")
    mesh_files = get_plate_meshes(args.plate_dir, args.n_meshes, plate_list)
    n_meshes = len(mesh_files)

    print(f"  Plate dir: {args.plate_dir}")
    print(f"  Meshes: {n_meshes}")
    print(f"  Trajectories per mesh: {args.trajs_per_mesh}")
    print(f"  Total sims: {n_meshes * args.trajs_per_mesh}")
    print(f"  Workers: {args.workers}")
    print(f"  Memory limit: {args.mem_limit}%")
    print(f"  Material: mu={args.mu}, lambda={args.lmbda}")
    print(f"  Compute stress: {args.compute_stress}")
    print(f"  Consolidate to HDF5: {args.consolidate}")
    print(f"  OMP threads/worker: {args.omp_threads}")
    if args.target_converged > 0:
        print(f"  Target converged/mesh: {args.target_converged}")
    print(f"  Output: {args.output_dir}")

    # Build task list
    tasks = build_task_list(mesh_files, args)
    n_skipped = n_meshes * args.trajs_per_mesh - len(tasks)
    print(f"  Tasks to run: {len(tasks)} ({n_skipped} skipped)")

    if not tasks:
        print("\n  No tasks to run.")
        if args.consolidate:
            consolidate_to_hdf5(args.output_dir)
        return

    # Set OMP threads for spawn workers (inherited via env)
    os.environ["_PLATE_FEM_OMP"] = str(args.omp_threads)
    os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.omp_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.omp_threads)

    # Run FEM
    effective_workers = min(args.workers, len(tasks))
    print(f"\n  Launching {effective_workers} workers for {len(tasks)} tasks...")

    t_start = time.time()
    n_done = 0
    n_ok = 0
    n_diverged = 0
    n_errors = 0
    converged_per_mesh = {}  # mesh_name -> count of converged trajectories

    # Pre-populate from existing NPZ files (for resume / top-up runs)
    npz_base = os.path.join(args.output_dir, 'npz')
    if os.path.isdir(npz_base):
        for mesh_name in os.listdir(npz_base):
            mesh_npz_dir = os.path.join(npz_base, mesh_name)
            if os.path.isdir(mesh_npz_dir):
                n_existing = len([f for f in os.listdir(mesh_npz_dir)
                                  if re.fullmatch(r'traj_\d+\.npz', f)])
                if n_existing > 0:
                    converged_per_mesh[mesh_name] = n_existing
    if converged_per_mesh:
        print(f"  Pre-existing converged: {sum(converged_per_mesh.values())} "
              f"across {len(converged_per_mesh)} meshes")

    # Background memory monitor
    mem_stop = threading.Event()
    def _memory_monitor():
        while not mem_stop.wait(10):
            mem = psutil.virtual_memory()
            elapsed = time.time() - t_start
            prefix = "  [WARNING]" if mem.percent > 80 else "  [MEM]"
            print(f"{prefix} {mem.used / 1024**3:.0f}/"
                  f"{mem.total / 1024**3:.0f}GB ({mem.percent:.0f}%) "
                  f"at {elapsed:.0f}s, {n_done}/{len(tasks)} done",
                  flush=True)

    mem_thread = threading.Thread(target=_memory_monitor, daemon=True)
    mem_thread.start()

    mem_limit = args.mem_limit
    mp_ctx = multiprocessing.get_context('spawn')

    with ProcessPoolExecutor(max_workers=effective_workers,
                             mp_context=mp_ctx) as executor:
        futures = {}
        task_idx = 0
        pool_broken = False

        def _submit_next():
            nonlocal task_idx, pool_broken
            target = args.target_converged
            if pool_broken:
                return False
            while task_idx < len(tasks):
                t = tasks[task_idx]
                # Skip meshes that already reached the converged target
                if target > 0 and converged_per_mesh.get(t['mesh_name'], 0) >= target:
                    task_idx += 1
                    continue
                try:
                    fut = executor.submit(_solve_one_trajectory, t)
                except BrokenProcessPool:
                    # A worker died hard (OOM kill, solver segfault). Stop
                    # submitting; finished results are kept and re-running
                    # resumes from the written trajectories.
                    pool_broken = True
                    print("  [POOL] a worker process died; no more tasks will be "
                          "submitted. Re-run to resume.", flush=True)
                    return False
                futures[fut] = t
                task_idx += 1
                return True
            return False

        # Fill initial pool
        for _ in range(effective_workers):
            if task_idx >= len(tasks):
                break
            if psutil.virtual_memory().percent >= mem_limit:
                break
            _submit_next()

        while futures:
            done_set = set()
            for fut in as_completed(futures, timeout=None):
                done_set.add(fut)
                break

            for fut in done_set:
                task_info = futures.pop(fut)
                try:
                    result = fut.result()
                except Exception as e:
                    result = {
                        'mesh_name': task_info['mesh_name'],
                        'traj_idx': task_info['traj_idx'],
                        'status': 'error',
                        'steps_completed': 0,
                        'time_seconds': 0,
                        'error': str(e),
                    }

                n_done += 1
                mesh_name = result['mesh_name']
                if result['status'] == 'ok':
                    n_ok += 1
                    converged_per_mesh[mesh_name] = converged_per_mesh.get(mesh_name, 0) + 1
                    if (args.target_converged > 0 and
                            converged_per_mesh[mesh_name] == args.target_converged):
                        print(f"  [TARGET] {mesh_name} reached "
                              f"{args.target_converged} converged — "
                              f"skipping remaining tasks for this mesh",
                              flush=True)
                elif result['status'] == 'diverged':
                    n_diverged += 1
                else:
                    n_errors += 1

                if n_done % 10 == 0 or n_done == len(tasks):
                    mem = psutil.virtual_memory()
                    elapsed = time.time() - t_start
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta = (len(tasks) - n_done) / rate if rate > 0 else 0
                    print(
                        f"  [{n_done}/{len(tasks)}] "
                        f"ok={n_ok} div={n_diverged} err={n_errors} "
                        f"mem={mem.percent:.0f}% "
                        f"rate={rate:.1f}/s ETA={eta:.0f}s "
                        f"| {result['mesh_name']}/traj_{result['traj_idx']:03d} "
                        f"{result['status']} ({result['time_seconds']:.1f}s)",
                        flush=True
                    )

                # Backfill with memory gate
                if task_idx < len(tasks):
                    while psutil.virtual_memory().percent >= mem_limit:
                        if not futures:
                            break
                        print(f"  [GATE] Memory {psutil.virtual_memory().percent:.0f}% "
                              f">= {mem_limit:.0f}% — waiting...", flush=True)
                        time.sleep(5)
                    _submit_next()

    mem_stop.set()
    t_elapsed = time.time() - t_start

    # Save meta
    convergence_rate = n_ok / max(1, n_done)
    # Record the solver constants the workers actually used (same env, so the
    # import succeeds whenever the workers did; fall back to the known values
    # only if dolfin itself cannot import here).
    try:
        from physics.fem.neohookean._primitives import (
            NEWTON_MAX_ITER, NEWTON_RELATIVE_TOLERANCE, BOUNDARY_EPS)
    except ImportError:
        NEWTON_MAX_ITER, NEWTON_RELATIVE_TOLERANCE, BOUNDARY_EPS = 15, 1e-3, 1e-2
    meta = {
        'plate_dir': args.plate_dir,
        'n_meshes': n_meshes,
        'trajs_per_mesh': args.trajs_per_mesh,
        'total_tasks': len(tasks),
        'n_ok': n_ok,
        'n_diverged': n_diverged,
        'n_errors': n_errors,
        'convergence_rate': convergence_rate,
        'target_converged': args.target_converged,
        'converged_per_mesh': converged_per_mesh,
        'workers': args.workers,
        'seed': args.seed,
        'time_seconds': t_elapsed,
        'created': datetime.now().isoformat(),
        'simulation_params': {
            'mu': args.mu,
            'lmbda': args.lmbda,
            'compute_stress': args.compute_stress,
            'omp_threads': args.omp_threads,
            'num_steps': args.n_steps,
            'speed_stretch': SPEED_S,
            'speed_twist': SPEED_T,
            'speed_shear': SPEED_SH,
            'newton_max_iterations': NEWTON_MAX_ITER,
            'newton_relative_tolerance': NEWTON_RELATIVE_TOLERANCE,
            'bc_epsilon': BOUNDARY_EPS,
        },
    }
    meta_path = os.path.join(args.output_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Pipeline Complete")
    print(f"{'=' * 60}")
    print(f"  Total: {n_done} done ({n_skipped} skipped)")
    print(f"  Converged: {n_ok}/{n_done} ({100*convergence_rate:.0f}%)")
    if converged_per_mesh:
        for mn in sorted(converged_per_mesh):
            print(f"    {mn}: {converged_per_mesh[mn]} converged")
    print(f"  Diverged: {n_diverged}, Errors: {n_errors}")
    print(f"  Time: {t_elapsed:.0f}s ({t_elapsed/60:.1f} min)")
    print(f"  Output: {args.output_dir}")

    # Optional NPZ → HDF5 consolidation (not consumed by the pipeline)
    if args.consolidate:
        print(f"\nConsolidating NPZ → HDF5...")
        consolidate_to_hdf5(args.output_dir)

    if pool_broken:
        sys.exit(2)
    if n_done > 0 and n_ok == 0:
        sys.exit(f"run_plate_fem.py: none of the {n_done} tasks converged "
                 f"({n_diverged} diverged, {n_errors} errors); see the per-task "
                 f"messages above (a worker that cannot import dolfin shows up "
                 f"here as errors).")


if __name__ == '__main__':
    main()
