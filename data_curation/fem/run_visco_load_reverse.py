#!/usr/bin/env python
"""
Load-Reverse viscoelastic plate FEM — generation + HDF5 packaging.

50 random loading steps + 50 exact reverse steps. The reverse protocol
creates matched cumBC points: forward step k has identical cumBC to
reverse step (100-k), but different displacement and stress due to
viscoelastic memory. This breaks the cumBC-as-sufficient-statistic pattern.

Two subcommands:
    fem            Run FEM per seed (per-seed NPZ output). Requires FEniCSx.
    build-hdf5     Concatenate per-seed NPZ → single HDF5 + train/val/test
                   split JSON. Pure numpy + h5py; no FEniCSx required.

Usage:
    # Per-seed FEM (FEniCSx env). The dataset build (README § 1):
    conda run -n fenicsx python data_curation/fem/run_visco_load_reverse.py fem --batch 1100 --workers 100
    # A single seed, for inspection. Plots/video need matplotlib, pyvista and
    # imageio, which the fenicsx env does not ship; skip them or install them:
    conda run -n fenicsx python data_curation/fem/run_visco_load_reverse.py fem --seed 0 --skip-plots --skip-video

    # Package per-seed NPZ → HDF5 (main env)
    python data_curation/fem/run_visco_load_reverse.py build-hdf5 \\
        --source-dir data/visco_load_reverse_production \\
        --output-h5 data/microplate_visco/microplate_visco.h5 \\
        --split-json data/manifests/microplate_visco_splits.json
"""

# Thread and MPI control (must be before numpy and mpi4py)
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMPI_MCA_btl", "self,tcp")   # Avoid InfiniBand exhaustion with 100+ workers
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("UCX_TLS", "self,sm,tcp")      # Same for MPICH/UCX builds: 100 single-rank workers on
os.environ.setdefault("FI_PROVIDER", "tcp")          # a Mellanox NIC exhaust its UAR pool and every worker dies

import sys
import time
import json
import argparse
import numpy as np

# Repo root on sys.path so `physics.fem` imports when invoked directly
# (physics/ is not part of the pip package). Spawned batch workers re-import
# this module, so they get the same path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# ============================================================================
# Constants
# ============================================================================

# Plate geometry (matches existing visco plate)
PLATE_XMIN, PLATE_XMAX = -4.9, 4.9
PLATE_YMIN, PLATE_YMAX = -4.9, 4.9
PLATE_ZMIN, PLATE_ZMAX = -0.98, 0.98
MESH_NX, MESH_NY, MESH_NZ = 10, 10, 2

# Time stepping (single phase, no hold or release)
DT_12 = 0.002     # 2 ms per FEM substep
DT_3  = 0.0002    # unused but kept for compatibility
N_FORWARD_STEPS = 50   # 50 random forward actions
PHASE1_STEPS = 500     # 100 actions × 5 substeps
PHASE2_STEPS = 0       # no hold
PHASE3_STEPS = 0       # no release
TOTAL_STEPS = PHASE1_STEPS  # 500 steps

# Action generation: 50 forward + 50 reverse = 100 steps
N_ACTION_STEPS = 100
N_SUBSTEPS = PHASE1_STEPS // N_ACTION_STEPS  # 5 FEM steps per action step
SPEED_S  = 0.10  # stretch (x-direction)
SPEED_T  = 0.06  # twist (rotation around x)
SPEED_SH = 0.10  # shear (y and z)

# Material: Arruda-Boyce equilibrium + 3-branch neo-Hookean non-equilibrium (v42 params)
MATERIAL = {
    'Geq_0':   200.0,       # equilibrium shear modulus [kPa]
    'lambdaL': 10.0,        # locking stretch
    'Kbulk':   400_000.0,   # bulk modulus [kPa]
    'rho':     1.3e-5,      # mass density [kg/mm^3] (10x physical for slower vibration)
    'branches': [
        (300.0, 0.001),     # Branch 0: fast damping (stabilizes release)
        (600.0, 0.2),       # Branch A: main relaxation (visible in hold, τ matches hold window)
        (150.0, 3.0),       # Branch B: slow background (partial relaxation, carries memory)
    ],
}

# Newmark-beta (trapezoidal rule)
ALPHA_NM = 0.0
GAMMA_NM = 0.5
BETA_NM  = 0.25


# ============================================================================
# Action generation (same pattern as existing visco plate dataset)
# ============================================================================

def generate_actions(n_steps, rng):
    """Generate one channel of discrete actions {-1, 0, +1} with random durations."""
    actions = np.zeros(n_steps, dtype=np.int8)
    t = 0
    while t < n_steps:
        duration = rng.integers(5, 15)
        end = min(t + duration, n_steps)
        state = rng.choice([-1, 0, 1])
        actions[t:end] = state
        t = end
    return actions


def generate_all_actions(n_steps, seed):
    """Generate the load-reverse 4-DOF action sequence: n_steps // 2 random
    forward actions followed by their exact reverse (100 = 50 + 50 by default)."""
    rng = np.random.default_rng(seed)
    n_fwd = n_steps // 2
    forward = np.stack([
        generate_actions(n_fwd, rng),
        generate_actions(n_fwd, rng),
        generate_actions(n_fwd, rng),
        generate_actions(n_fwd, rng),
    ], axis=1)  # (50, 4)
    reverse = (-forward[::-1]).copy()  # (50, 4)
    return np.vstack([forward, reverse])  # (100, 4)


# ============================================================================
# FEM driver — thin wrapper around physics.fem.solve_arruda_boyce
# ============================================================================

def run_simulation(args):
    """Run 3-phase viscoelastic FEM. Returns path to saved NPZ.

    Driver layer: parses args, generates actions, calls the shared kernel
    (`physics.fem.solve_arruda_boyce`), writes NPZ + metadata. The kernel
    holds the FEniCSx form, time integration, and Newton solve.
    """
    from physics.fem import solve_arruda_boyce
    import copy

    mat = copy.deepcopy(MATERIAL)
    if args.fast_branch_G is not None:
        mat['branches'][0] = (args.fast_branch_G, mat['branches'][0][1])
    speed_s = args.speed if args.speed is not None else SPEED_S
    speed_t = (args.speed * 0.6) if args.speed is not None else SPEED_T
    speed_sh = args.speed if args.speed is not None else SPEED_SH
    seed = args.seed

    p1_steps, p2_steps = PHASE1_STEPS, PHASE2_STEPS
    p3_steps = args.phase3_steps if args.phase3_steps else PHASE3_STEPS
    n_action_steps = args.n_action_steps if args.n_action_steps else N_ACTION_STEPS
    release_ramp = args.release_ramp
    n_substeps = p1_steps // n_action_steps
    # Keep the phase-1 length consistent with the action table (actions are
    # repeated n_substeps times): 100 actions x 5 = 500 by default; a custom
    # --n-action-steps shortens it to the largest multiple that fits.
    p1_steps = n_action_steps * n_substeps
    dt_12 = DT_12
    dt_3 = DT_3

    actions = generate_all_actions(n_action_steps, seed)

    result = solve_arruda_boyce(
        actions,
        material=mat,
        box_min=(PLATE_XMIN, PLATE_YMIN, PLATE_ZMIN),
        box_max=(PLATE_XMAX, PLATE_YMAX, PLATE_ZMAX),
        mesh_resolution=(MESH_NX, MESH_NY, MESH_NZ),
        speeds=(speed_s, speed_t, speed_sh),
        n_substeps=n_substeps,
        dt_load=dt_12,
        dt_release=dt_3,
        p1_steps=p1_steps,
        p2_steps=p2_steps,
        p3_steps=p3_steps,
        release_ramp=release_ramp,
        newmark=(ALPHA_NM, GAMMA_NM, BETA_NM),
        log_progress=True,
    )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, f'visco_lr_seed{seed}.npz')

    # Atomic write: resume and build-hdf5 see complete files only. The temp
    # name does not end in .npz (so no listing picks it up), hence the file
    # object: np.savez would otherwise append '.npz'.
    tmp_path = npz_path + '.partial'
    _fh = open(tmp_path, 'wb')
    np.savez(_fh,
             # Mesh
             coords=result['coords'],
             topology=result['topology'],
             # Per-frame fields
             displacements=result['displacements'],
             velocities=result['velocities'],
             cauchy_stress=result['cauchy_stress'],
             Cv=result['Cv'],
             pressure=result['pressure'],
             # Per-frame metadata
             bc_state=result['bc_state'],
             phase_labels=result['phase_labels'],
             times=result['times'],
             actions_5dof=result['actions_5dof'],
             traction_scale=result['traction_scale'],
             reaction_force=result['reaction_force'],
             # Phase 1 raw + expanded actions (for reference + reproduction)
             actions_raw=actions,
             actions_expanded=result['actions_expanded'],
             # Config — full reproduction metadata
             dt=np.float32(dt_12),
             dt_phase3=np.float32(dt_3),
             phase1_steps=np.int32(p1_steps),
             phase2_steps=np.int32(p2_steps),
             phase3_steps=np.int32(p3_steps),
             seed=np.int32(seed),
             material=json.dumps(mat),
             dataset_config=json.dumps({
                 'material': mat,
                 'time_stepping': {'dt_phase12': dt_12, 'dt_phase3': dt_3,
                                   'phase1_steps': p1_steps, 'phase2_steps': p2_steps,
                                   'phase3_steps': p3_steps},
                 'loading': {'speed_s': speed_s, 'speed_t': speed_t, 'speed_sh': speed_sh,
                             'n_action_steps': n_action_steps, 'n_substeps': n_substeps},
                 'release': {'ramp_steps': release_ramp, 'method': 'traction'},
                 'mesh': {'plate_xmin': PLATE_XMIN, 'plate_xmax': PLATE_XMAX,
                          'plate_ymin': PLATE_YMIN, 'plate_ymax': PLATE_YMAX,
                          'plate_zmin': PLATE_ZMIN, 'plate_zmax': PLATE_ZMAX,
                          'mesh_nx': MESH_NX, 'mesh_ny': MESH_NY, 'mesh_nz': MESH_NZ},
                 'newmark': {'alpha': ALPHA_NM, 'gamma': GAMMA_NM, 'beta': BETA_NM},
             }))
    _fh.close()
    os.replace(tmp_path, npz_path)

    elapsed = result['time_seconds']
    size_mb = os.path.getsize(npz_path) / (1024**2)
    print(f"\n  Saved: {npz_path} ({size_mb:.1f} MB)")
    print(f"  Shapes: disp={result['displacements'].shape}, "
          f"vel={result['velocities'].shape}, "
          f"cauchy={result['cauchy_stress'].shape}, "
          f"Cv={result['Cv'].shape}")

    meta = {
        'seed': seed,
        'dt_phase12': dt_12,
        'dt_phase3': dt_3,
        'phase1_steps': p1_steps,
        'phase2_steps': p2_steps,
        'phase3_steps': p3_steps,
        'total_steps': p1_steps + p2_steps + p3_steps,
        'material': mat,
        'loading': {'speed_s': speed_s, 'speed_t': speed_t, 'speed_sh': speed_sh,
                    'n_action_steps': n_action_steps, 'release_ramp': release_ramp},
        'n_vertices': int(result['coords'].shape[0]),
        'n_cells': int(result['topology'].shape[0]),
        'wall_clock_s': elapsed,
        'final_bc_state': {k: float(v) for k, v in result['final_bc_state'].items()},
        'status': result['status'],
    }
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return npz_path


# ============================================================================
# Diagnostic plots
# ============================================================================

_PLOT_DEPS_MSG = ("plots/video need matplotlib, pyvista and imageio, which the fenicsx "
                  "env does not ship: install them or pass --skip-plots --skip-video")


def make_diagnostic_plots(npz_path, output_dir):
    """Generate stress, displacement, and velocity time-trace plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(_PLOT_DEPS_MSG) from e

    d = np.load(npz_path)
    coords = d['coords']
    topo = d['topology']
    displacements = d['displacements']
    velocities = d['velocities']
    p1 = int(d['phase1_steps'])
    p2 = int(d['phase2_steps'])
    p3 = int(d['phase3_steps'])
    times = d['times']

    n_frames = displacements.shape[0]
    t_phase1_end = times[p1] if p1 < len(times) else times[-1]
    t_phase2_end = times[p1 + p2] if (p1 + p2) < len(times) else times[-1]

    cauchy_stress = d['cauchy_stress']

    # Probe points
    right_mask = np.isclose(coords[:, 0], PLATE_XMAX)
    center_y = (PLATE_YMIN + PLATE_YMAX) / 2
    center_z = (PLATE_ZMIN + PLATE_ZMAX) / 2
    dists = np.sqrt((coords[:, 1] - center_y)**2 + (coords[:, 2] - center_z)**2)
    dists[~right_mask] = 1e10
    tip_idx = np.argmin(dists)

    cx = (PLATE_XMIN + PLATE_XMAX) / 2
    dists2 = np.sqrt((coords[:, 0] - cx)**2 + (coords[:, 1] - center_y)**2
                     + (coords[:, 2] - center_z)**2)
    mid_idx = np.argmin(dists2)

    probe_names = ['Right-face tip', 'Plate center']
    probe_indices = [tip_idx, mid_idx]

    def add_phase_bands(ax):
        ax.axvspan(0, t_phase1_end, alpha=0.08, color='red')
        ax.axvspan(t_phase1_end, t_phase2_end, alpha=0.08, color='blue')
        ax.axvspan(t_phase2_end, times[-1], alpha=0.08, color='green')
        ax.axvline(t_phase1_end, color='red', ls='--', lw=0.8, alpha=0.5)
        ax.axvline(t_phase2_end, color='blue', ls='--', lw=0.8, alpha=0.5)

    # --- Stress time ---
    def von_mises_from_cauchy(sigma):
        s = sigma - np.trace(sigma, axis1=-2, axis2=-1)[..., None, None] / 3.0 * np.eye(3)
        return np.sqrt(1.5 * np.maximum(np.sum(s * s, axis=(-2, -1)), 0.0))

    vm_probes = {idx: von_mises_from_cauchy(cauchy_stress[:, idx]) for idx in probe_indices}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for col, (idx, name) in enumerate(zip(probe_indices, probe_names)):
        vm = vm_probes[idx]
        ax = axes[col]
        add_phase_bands(ax)
        ax.plot(times, vm, 'k-', lw=0.8)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('von Mises stress [kPa]')
        ax.set_title(f'{name} (node {idx})')
        ax.grid(True, alpha=0.3)
        # Phase labels (use axes coordinates for y to avoid ylim issues)
        for label, t_start_l, t_end_l, color in [
            ('LOAD', 0, t_phase1_end, 'red'),
            ('HOLD', t_phase1_end, t_phase2_end, 'blue'),
            ('RELEASE', t_phase2_end, times[-1], 'green')]:
            ax.text((t_start_l + t_end_l) / 2, 0.95, label, ha='center', fontsize=9,
                    color=color, alpha=0.7, transform=ax.get_xaxis_transform())
    plt.suptitle('Von Mises Stress at Probe Points', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, 'stress_time.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # --- Displacement time ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for col, (idx, name) in enumerate(zip(probe_indices, probe_names)):
        u_probe = displacements[:, idx]
        for row, (comp, label, color) in enumerate(
                [(2, 'u_z', 'tab:blue'), (None, '|u|', 'tab:orange')]):
            ax = axes[row, col]
            add_phase_bands(ax)
            if comp is not None:
                ax.plot(times, u_probe[:, comp], color=color, lw=0.8)
                ax.set_ylabel(f'{label} [mm]')
            else:
                ax.plot(times, np.linalg.norm(u_probe, axis=-1), color=color, lw=0.8)
                ax.set_ylabel(f'{label} [mm]')
            ax.set_title(f'{name}' if row == 0 else '')
            ax.grid(True, alpha=0.3)
            if row == 1:
                ax.set_xlabel('Time [s]')
    plt.suptitle('Displacement at Probe Points', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, 'displacement_time.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # --- Velocity time ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for col, (idx, name) in enumerate(zip(probe_indices, probe_names)):
        v_probe = velocities[:, idx]
        for row, (comp, label, color) in enumerate(
                [(2, 'v_z', 'tab:red'), (None, '|v|', 'tab:purple')]):
            ax = axes[row, col]
            add_phase_bands(ax)
            if comp is not None:
                ax.plot(times, v_probe[:, comp], color=color, lw=0.5)
                ax.set_ylabel(f'{label} [mm/s]')
            else:
                ax.plot(times, np.linalg.norm(v_probe, axis=-1), color=color, lw=0.5)
                ax.set_ylabel(f'{label} [mm/s]')
            ax.set_title(f'{name}' if row == 0 else '')
            ax.grid(True, alpha=0.3)
            if row == 1:
                ax.set_xlabel('Time [s]')
    plt.suptitle('Velocity at Probe Points', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, 'velocity_time.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# Video rendering
# ============================================================================

def render_video(npz_path, output_dir):
    """Render multi-panel video: displacement, von Mises stress, velocity."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pyvista as pv
        import imageio
        from PIL import Image
    except ImportError as e:
        raise ImportError(_PLOT_DEPS_MSG) from e

    pv.OFF_SCREEN = True

    d = np.load(npz_path)
    coords = d['coords'].astype(np.float64)
    topo = d['topology'].astype(np.int64)
    displacements = d['displacements'].astype(np.float64)
    velocities = d['velocities'].astype(np.float64)
    cauchy_stress = d['cauchy_stress'].astype(np.float64)
    times_all = d['times']
    p1 = int(d['phase1_steps'])
    p2 = int(d['phase2_steps'])

    n_frames_total = displacements.shape[0]
    t_phase1_end = times_all[p1] if p1 < len(times_all) else times_all[-1]
    t_phase2_end = times_all[p1 + p2] if (p1 + p2) < len(times_all) else times_all[-1]

    # Subsample
    stride = max(1, n_frames_total // 500)
    frame_indices = list(range(0, n_frames_total, stride))
    if frame_indices[-1] != n_frames_total - 1:
        frame_indices.append(n_frames_total - 1)
    n_frames = len(frame_indices)
    print(f"  Video: {n_frames_total} -> {n_frames} frames (stride={stride})")

    # Pre-compute scalars for subsampled frames
    disp_sub = displacements[frame_indices]
    vel_sub = velocities[frame_indices]
    disp_mag = np.linalg.norm(disp_sub, axis=-1)
    vel_mag = np.linalg.norm(vel_sub, axis=-1)

    # Von Mises from stored Cauchy stress
    def von_mises_nodal(sigma):
        s = sigma - np.trace(sigma, axis1=-2, axis2=-1)[..., None, None] / 3.0 * np.eye(3)
        return np.sqrt(1.5 * np.maximum(np.sum(s * s, axis=(-2, -1)), 0.0))

    cauchy_sub = cauchy_stress[frame_indices]
    print(f"  Computing von Mises for {n_frames} video frames...")
    vm_all = np.array([von_mises_nodal(cauchy_sub[i]) for i in range(n_frames)])

    # Color scales (99.5th percentile)
    disp_vmax = max(float(np.percentile(disp_mag, 99.5)), 1e-6)
    vel_vmax = max(float(np.percentile(vel_mag, 99.5)), 1e-6)
    vm_vmax = max(float(np.percentile(vm_all, 99.5)), 1e-6)
    print(f"  Scales: |u|_max={disp_vmax:.3f}, VM_max={vm_vmax:.1f}, |v|_max={vel_vmax:.1f}")

    # Camera setup
    grid0 = pv.UnstructuredGrid({pv.CellType.TETRA: topo}, coords)
    pl = pv.Plotter(off_screen=True, window_size=(500, 500))
    pl.add_mesh(grid0)
    pl.view_isometric()
    pl.camera.zoom(0.85)
    cam = pl.camera_position
    pl.close()

    panel_size = 500

    def render_mesh(disp_t, scalars, vmax, cmap, title):
        deformed = coords + disp_t
        grid = pv.UnstructuredGrid({pv.CellType.TETRA: topo}, deformed)
        grid.point_data['S'] = scalars
        p = pv.Plotter(off_screen=True, window_size=(panel_size, panel_size))
        p.add_mesh(grid, scalars='S', clim=[0, vmax], cmap=cmap,
                   show_edges=False, show_scalar_bar=True,
                   scalar_bar_args={'title': title, 'n_labels': 3,
                                    'label_font_size': 14, 'title_font_size': 16,
                                    'color': 'white', 'width': 0.3})
        p.camera_position = cam
        p.set_background('black')
        img = p.screenshot(return_img=True)
        p.close()
        return img

    # Probe node for time trace
    cx = (coords[:, 0].min() + coords[:, 0].max()) / 2
    cy_c = (coords[:, 1].min() + coords[:, 1].max()) / 2
    cz_c = (coords[:, 2].min() + coords[:, 2].max()) / 2
    dists = np.sqrt((coords[:, 0]-cx)**2 + (coords[:, 1]-cy_c)**2 + (coords[:, 2]-cz_c)**2)
    center_idx = np.argmin(dists)

    # Full-res time traces for bottom panel (subsample for plotting)
    trace_stride = max(1, n_frames_total // 2000)
    times_trace = times_all[::trace_stride]

    # Von Mises at center node (subsampled for trace)
    vm_center_trace = np.array([von_mises_nodal(cauchy_stress[i, center_idx:center_idx+1])[0]
                                for i in range(0, n_frames_total, trace_stride)])
    disp_z_center = displacements[::trace_stride, center_idx, 2]

    def render_time_trace(t_current, width, height):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width/100, height/100), dpi=100)

        # Von Mises stress at center (left panel)
        ax1.plot(times_trace, vm_center_trace, 'gold', lw=1)
        ax1.axvline(t_current, color='white', lw=2, ls='--', alpha=0.8)
        ax1.axvspan(0, t_phase1_end, alpha=0.15, color='red')
        ax1.axvspan(t_phase1_end, t_phase2_end, alpha=0.15, color='blue')
        ax1.axvspan(t_phase2_end, times_all[-1], alpha=0.15, color='green')
        ax1.set_xlabel('Time [s]', color='white', fontsize=10)
        ax1.set_ylabel('VM stress [kPa]', color='white', fontsize=10)
        ax1.set_title('Von Mises (center)', color='white', fontsize=11)
        ax1.set_facecolor('#1a1a1a')
        ax1.tick_params(colors='white', labelsize=8)
        ax1.grid(True, alpha=0.2)
        for spine in ax1.spines.values():
            spine.set_color('white')

        # Displacement z at center (right panel)
        ax2.plot(times_trace[:len(disp_z_center)], disp_z_center, 'dodgerblue', lw=1)
        ax2.axvline(t_current, color='white', lw=2, ls='--', alpha=0.8)
        ax2.axvspan(0, t_phase1_end, alpha=0.15, color='red')
        ax2.axvspan(t_phase1_end, t_phase2_end, alpha=0.15, color='blue')
        ax2.axvspan(t_phase2_end, times_all[-1], alpha=0.15, color='green')
        ax2.set_xlabel('Time [s]', color='white', fontsize=10)
        ax2.set_ylabel('u_z [mm]', color='white', fontsize=10)
        ax2.set_title('Displacement z (center)', color='white', fontsize=11)
        ax2.set_facecolor('#1a1a1a')
        ax2.tick_params(colors='white', labelsize=8)
        ax2.grid(True, alpha=0.2)
        for spine in ax2.spines.values():
            spine.set_color('white')

        fig.set_facecolor('black')
        plt.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return buf

    # Video setup
    frame_w = panel_size * 3
    trace_h = 250
    frame_h = panel_size + trace_h
    frame_w = ((frame_w + 15) // 16) * 16
    frame_h = ((frame_h + 15) // 16) * 16

    out_path = os.path.join(output_dir, 'visco_3phase.mp4')
    writer = imageio.get_writer(out_path, fps=15, codec='libx264',
                                quality=8, macro_block_size=16)
    print(f"  Rendering {n_frames} frames -> {out_path}")
    t0 = time.time()

    for fi in range(n_frames):
        orig_idx = frame_indices[fi]
        t_sim = times_all[orig_idx]

        if orig_idx <= p1:
            phase_label = "LOAD"
        elif orig_idx <= p1 + p2:
            phase_label = "HOLD"
        else:
            phase_label = "RELEASE"

        img_disp = render_mesh(disp_sub[fi], disp_mag[fi], disp_vmax, 'turbo', '|u| [mm]')
        img_vm = render_mesh(disp_sub[fi], vm_all[fi], vm_vmax, 'inferno', 'VM [kPa]')
        img_vel = render_mesh(disp_sub[fi], vel_mag[fi], vel_vmax, 'plasma', '|v| [mm/s]')

        top = np.concatenate([img_disp, img_vm, img_vel], axis=1)

        # Add text overlay: phase label + VM stats + time
        from PIL import Image as PILImage, ImageDraw, ImageFont
        top_pil = PILImage.fromarray(top)
        draw = ImageDraw.Draw(top_pil)
        vm_max_frame = float(vm_all[fi].max())
        vm_mean_frame = float(vm_all[fi].mean())
        overlay_text = f"{phase_label}  t={t_sim:.3f}s  VM_max={vm_max_frame:.0f}  VM_mean={vm_mean_frame:.0f} kPa"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        draw.text((10, 10), overlay_text, fill=(255, 255, 255), font=font)
        top = np.array(top_pil)

        trace = render_time_trace(t_sim, frame_w, trace_h)
        trace_pil = Image.fromarray(trace).resize((frame_w, trace_h), Image.LANCZOS)
        trace = np.array(trace_pil)

        if top.shape[1] != frame_w:
            top_pil = Image.fromarray(top).resize((frame_w, panel_size), Image.LANCZOS)
            top = np.array(top_pil)

        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        frame[:top.shape[0], :top.shape[1]] = top[:frame_h, :frame_w]
        y_off = top.shape[0]
        h_avail = frame_h - y_off
        frame[y_off:y_off+min(trace.shape[0], h_avail), :min(trace.shape[1], frame_w)] = \
            trace[:min(trace.shape[0], h_avail), :min(trace.shape[1], frame_w)]

        writer.append_data(frame)

        if fi % 50 == 0 or fi == n_frames - 1:
            elapsed = time.time() - t0
            eta = (n_frames - fi - 1) * (elapsed / (fi + 1)) if fi > 0 else 0
            print(f"    Frame {fi+1}/{n_frames} [{phase_label}] t={t_sim:.3f}s "
                  f"[{elapsed:.0f}s, ETA {eta:.0f}s]")

    writer.close()
    size_mb = os.path.getsize(out_path) / (1024**2)
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    # --- subcommand: fem ---
    fem_p = sub.add_parser('fem', help='Run FEM per seed (FEniCSx env required).')
    fem_p.add_argument('--seed', type=int, default=42)
    fem_p.add_argument('--output-dir', default='data/visco_load_reverse_production')
    fem_p.add_argument('--skip-video', action='store_true',
                       help='Skip video rendering')
    fem_p.add_argument('--skip-plots', action='store_true',
                       help='Skip diagnostic plots')
    fem_p.add_argument('--fast-branch-G', type=float, default=None,
                       help='Override G for fast damping branch (kPa)')
    fem_p.add_argument('--speed', type=float, default=None,
                       help='Override loading speed for stretch/shear (twist = speed * 0.6)')
    fem_p.add_argument('--n-action-steps', type=int, default=None,
                       help='Override number of action steps in Phase 1')
    fem_p.add_argument('--release-ramp', type=int, default=0,
                       help='Ramp traction to zero over N steps (0 = ramp over entire Phase 3)')
    fem_p.add_argument('--phase3-steps', type=int, default=None,
                       help='Override number of Phase 3 steps')
    fem_p.add_argument('--batch', type=int, default=0,
                       help='Run N trajectories (seeds 0..N-1) in parallel. 0 = single run.')
    fem_p.add_argument('--workers', type=int, default=100,
                       help='Number of parallel workers for batch mode')

    # --- subcommand: build-hdf5 ---
    bh = sub.add_parser('build-hdf5',
                        help='Concatenate per-seed NPZ → single HDF5 (no FEniCSx).')
    bh.add_argument('--source-dir', required=True,
                    help='Per-seed NPZ root (output of `fem` subcommand).')
    bh.add_argument('--output-h5', required=True,
                    help='Output HDF5 path (e.g. data/microplate_visco/microplate_visco.h5).')
    bh.add_argument('--split-json', required=True,
                    help='Output split JSON path (e.g. data/manifests/microplate_visco_splits.json).')
    bh.add_argument('--allow-short-pool', action='store_true',
                    help='Build even if fewer than 1000 converged trajectories are available '
                         '(the split then differs from the committed manifest).')

    return parser.parse_args()


# ============================================================================
# build-hdf5 subcommand: concatenate per-seed NPZ → single HDF5 + split JSON.
# Lightweight (numpy + h5py); kept here so the workflow is one file.
# ============================================================================

_BUILD_HDF5_STRIDE = 5
_BUILD_HDF5_RAW_FRAMES = 501       # init + 500 FEM steps
_BUILD_HDF5_STRIDED_FRAMES = 101   # init + 100 action steps
_BUILD_HDF5_N_ACTIONS = 100        # 50 forward + 50 reverse
_BUILD_HDF5_POOL_SIZE = 1000
_BUILD_HDF5_SPLIT_SEED = 42
_BUILD_HDF5_GROUP = "solid_plate_visco"


def build_hdf5(args):
    """Concatenate per-seed NPZ files into a single stride-subsampled HDF5
    and write the train/val/test split JSON. No FEniCSx dependency.
    """
    import h5py

    src = args.source_dir
    out_h5 = args.output_h5
    out_split = args.split_json
    out_dir = os.path.dirname(out_h5) or '.'
    os.makedirs(out_dir, exist_ok=True)

    print(f"Scanning {src}...")
    seeds = sorted([d for d in os.listdir(src) if d.startswith('seed_')],
                   key=lambda x: int(x.split('_')[1]))

    converged = []
    for s in seeds:
        sid = int(s.split('_')[1])
        npz_path = os.path.join(src, s, f'visco_lr_seed{sid}.npz')   # exact name only
        if os.path.exists(npz_path):
            d = np.load(npz_path, mmap_mode='r')
            if d['displacements'].shape[0] == _BUILD_HDF5_RAW_FRAMES:
                converged.append((sid, npz_path))

    print(f"Found {len(converged)} converged trajectories")
    pool = converged[:_BUILD_HDF5_POOL_SIZE]
    n_traj = len(pool)
    if n_traj == 0:
        sys.exit(f"build-hdf5: no converged trajectory under {src}")
    if n_traj < _BUILD_HDF5_POOL_SIZE and not args.allow_short_pool:
        sys.exit(f"build-hdf5: only {n_traj} converged trajectories; the paper pool is the "
                 f"first {_BUILD_HDF5_POOL_SIZE} converged seeds (run `fem --batch 1100`), and "
                 f"a shorter pool gives a different split than the committed manifest. "
                 f"Pass --allow-short-pool to build it anyway.")
    print(f"Using pool of {n_traj}")

    first = np.load(pool[0][1])
    coords = first['coords']
    topology = first['topology']
    n_nodes = coords.shape[0]
    stride_idx = np.arange(0, _BUILD_HDF5_RAW_FRAMES, _BUILD_HDF5_STRIDE)
    assert len(stride_idx) == _BUILD_HDF5_STRIDED_FRAMES

    print(f"Writing {out_h5}...")
    tmp_h5 = out_h5 + '.partial'   # atomic: train/eval never see a half-written file
    with h5py.File(tmp_h5, 'w') as f:
        grp = f.create_group(_BUILD_HDF5_GROUP)
        grp.create_dataset('coords', data=coords)
        grp.create_dataset('topology', data=topology.astype(np.int32))

        sf = _BUILD_HDF5_STRIDED_FRAMES
        ds_inp = grp.create_dataset('inputs', shape=(n_traj, _BUILD_HDF5_N_ACTIONS, 4), dtype=np.int8)
        ds_disp = grp.create_dataset('displacements', shape=(n_traj, sf, n_nodes, 3),
                                     dtype=np.float32, chunks=(1, sf, n_nodes, 3))
        ds_vel = grp.create_dataset('velocities', shape=(n_traj, sf, n_nodes, 3),
                                    dtype=np.float32, chunks=(1, sf, n_nodes, 3))
        ds_stress = grp.create_dataset('cauchy_stress', shape=(n_traj, sf, n_nodes, 3, 3),
                                       dtype=np.float32, chunks=(1, sf, n_nodes, 3, 3))
        ds_pres = grp.create_dataset('pressure', shape=(n_traj, sf, n_nodes),
                                     dtype=np.float32, chunks=(1, sf, n_nodes))
        ds_cv = grp.create_dataset('Cv', shape=(n_traj, sf, 3, n_nodes, 6),
                                   dtype=np.float32, chunks=(1, sf, 3, n_nodes, 6))

        seed_ids = np.array([s[0] for s in pool], dtype=np.int32)
        grp.create_dataset('seed_ids', data=seed_ids)

        # Metadata: pull the canonical material constants from this file's
        # MATERIAL dict so a single edit propagates to both subcommands.
        grp.attrs['Geq_0'] = MATERIAL['Geq_0']
        grp.attrs['lambdaL'] = MATERIAL['lambdaL']
        grp.attrs['Kbulk'] = MATERIAL['Kbulk']
        grp.attrs['rho'] = MATERIAL['rho']
        grp.attrs['branch_G'] = [b[0] for b in MATERIAL['branches']]
        grp.attrs['branch_tau'] = [b[1] for b in MATERIAL['branches']]
        grp.attrs['stride'] = _BUILD_HDF5_STRIDE
        grp.attrs['protocol'] = 'load_reverse'
        grp.attrs['n_forward_steps'] = 50

        for i, (sid, npz_path) in enumerate(pool):
            if i % 100 == 0:
                print(f"  {i}/{n_traj} (seed_{sid})...")
            d = np.load(npz_path)
            ds_inp[i] = d['actions_raw'][:_BUILD_HDF5_N_ACTIONS]
            ds_disp[i] = d['displacements'][stride_idx]
            ds_vel[i] = d['velocities'][stride_idx]
            ds_stress[i] = d['cauchy_stress'][stride_idx]
            ds_pres[i] = d['pressure'][stride_idx]
            ds_cv[i] = d['Cv'][stride_idx]

    os.replace(tmp_h5, out_h5)
    size_gb = os.path.getsize(out_h5) / (1024 ** 3)
    print(f"HDF5: {out_h5} ({size_gb:.1f} GB)")

    # Split (deterministic via fixed seed).
    rng = np.random.RandomState(_BUILD_HDF5_SPLIT_SEED)
    indices = rng.permutation(n_traj)
    n_train = int(n_traj * 0.8)
    n_val = int(n_traj * 0.1)
    split = {
        'pool_size': n_traj,
        'seed': _BUILD_HDF5_SPLIT_SEED,
        'train': sorted(indices[:n_train].tolist()),
        'val': sorted(indices[n_train:n_train + n_val].tolist()),
        'test': sorted(indices[n_train + n_val:].tolist()),
        'protocol': 'load_reverse',
        'n_forward_steps': 50,
    }
    os.makedirs(os.path.dirname(out_split) or '.', exist_ok=True)
    tmp_split = out_split + '.partial'
    with open(tmp_split, 'w') as f:
        json.dump(split, f, indent=2)
    os.replace(tmp_split, out_split)
    print(f"Split: {len(split['train'])} train / "
          f"{len(split['val'])} val / {len(split['test'])} test")


def _run_one_seed(task):
    """Worker function for batch mode. Runs one seed, returns status dict."""
    import argparse
    seed = task['seed']
    output_dir = task['output_dir']
    release_ramp = task['release_ramp']

    args = argparse.Namespace(
        seed=seed, output_dir=output_dir,
        skip_video=True, skip_plots=True,
        fast_branch_G=None, speed=None,
        n_action_steps=None, release_ramp=release_ramp,
        phase3_steps=None, batch=0, workers=1)

    t0 = time.time()
    try:
        npz_path = run_simulation(args)
        # Check if complete
        d = np.load(npz_path)
        n = d['displacements'].shape[0]
        expected = int(d['phase1_steps']) + int(d['phase2_steps']) + int(d['phase3_steps']) + 1
        status = 'complete' if n == expected else 'diverged'
        return {'seed': seed, 'status': status, 'wall_s': time.time() - t0,
                'n_frames': n, 'expected': expected}
    except Exception as e:
        return {'seed': seed, 'status': 'error', 'wall_s': time.time() - t0,
                'error': str(e)}


def _jit_warmup(fast_branch_G):
    """One-step solve to populate the FEniCSx JIT cache (runs in a spawned worker).

    ffcx compiles the forms into a per-user cache (~/.cache/fenics) on first
    use. Workers that start together on a cold cache race on the same files
    and all die (FileExistsError, then "JIT compilation timed out"). The
    kernels do not depend on the numeric parameters — material and time-step
    values are dolfinx Constants — so one solve of a single step with the same
    mesh and material structure compiles exactly what the workers need.
    """
    import copy
    from physics.fem import solve_arruda_boyce

    mat = copy.deepcopy(MATERIAL)
    if fast_branch_G is not None:
        mat['branches'][0] = (fast_branch_G, mat['branches'][0][1])
    solve_arruda_boyce(
        np.zeros((1, 4), dtype=int),
        material=mat,
        box_min=(PLATE_XMIN, PLATE_YMIN, PLATE_ZMIN),
        box_max=(PLATE_XMAX, PLATE_YMAX, PLATE_ZMAX),
        mesh_resolution=(MESH_NX, MESH_NY, MESH_NZ),
        n_substeps=1, dt_load=DT_12, dt_release=DT_3,
        p1_steps=1, p2_steps=0, p3_steps=0,
        newmark=(ALPHA_NM, GAMMA_NM, BETA_NM),
        log_progress=False,
    )
    return True


def run_batch(args):
    """Run multiple seeds in parallel using ProcessPoolExecutor."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    import threading

    n_seeds = args.batch
    n_workers = min(args.workers, n_seeds)
    base_dir = args.output_dir

    os.makedirs(base_dir, exist_ok=True)

    # Build task list
    tasks = []
    for seed in range(n_seeds):
        out_dir = os.path.join(base_dir, f'seed_{seed}')
        npz_path = os.path.join(out_dir, f'visco_lr_seed{seed}.npz')
        if os.path.exists(npz_path):
            # Check if already complete
            try:
                d = np.load(npz_path)
                n = d['displacements'].shape[0]
                expected = int(d['phase1_steps']) + int(d['phase2_steps']) + int(d['phase3_steps']) + 1
                if n == expected:
                    print(f"  Skipping seed={seed} (already complete)")
                    continue
            except Exception:
                pass
        tasks.append({'seed': seed, 'output_dir': out_dir, 'release_ramp': args.release_ramp})

    print("=" * 60)
    print(f"Batch Mode: {n_seeds} seeds, {len(tasks)} to run, {n_workers} workers")
    print("=" * 60)

    if not tasks:
        print("All seeds already complete.")
        return

    mp_ctx = multiprocessing.get_context('spawn')

    # Warm the FEniCSx JIT cache once before launching the workers (see _jit_warmup).
    print("JIT warm-up: compiling the FEniCSx forms once before launching workers...",
          flush=True)
    t_warm = time.time()
    with ProcessPoolExecutor(max_workers=1, mp_context=mp_ctx) as warm:
        warm.submit(_jit_warmup, args.fast_branch_G).result()
    print(f"JIT warm-up done ({time.time() - t_warm:.0f}s)", flush=True)

    t_start = time.time()
    n_done = 0
    n_ok = 0
    n_div = 0
    n_err = 0

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as executor:
        futures = {executor.submit(_run_one_seed, t): t for t in tasks}

        for fut in as_completed(futures):
            try:
                result = fut.result()
            except BrokenProcessPool:
                print("\n[POOL] a worker process died (OOM kill or solver crash); "
                      "the remaining seeds were not run. Finished seeds are on disk; "
                      "re-run the same command to resume.", flush=True)
                sys.exit(2)
            except Exception as e:
                task = futures[fut]
                result = {'seed': task.get('seed', '?'), 'status': 'error',
                          'wall_s': 0.0, 'error': str(e)}
                print(f"  seed={result['seed']} error: {str(e)[:200]}", flush=True)
            n_done += 1
            if result['status'] == 'complete':
                n_ok += 1
            elif result['status'] == 'diverged':
                n_div += 1
            else:
                n_err += 1

            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(tasks) - n_done) / rate if rate > 0 else 0

            if n_done % 10 == 0 or n_done == len(tasks):
                print(f"  [{n_done}/{len(tasks)}] ok={n_ok} div={n_div} err={n_err} "
                      f"rate={rate:.2f}/s ETA={eta:.0f}s | seed={result['seed']} "
                      f"{result['status']} ({result['wall_s']:.0f}s)", flush=True)

    elapsed = time.time() - t_start
    print(f"\nBatch complete: {n_ok}/{n_seeds} succeeded, "
          f"{n_div} diverged, {n_err} errors, {elapsed/60:.0f} min total")
    print(f"Success rate: {n_ok/n_seeds:.0%}")
    if tasks and n_ok == 0:
        sys.exit(f"run_visco_load_reverse.py: none of the {len(tasks)} dispatched seeds "
                 f"completed ({n_div} diverged, {n_err} errors); see the per-seed messages above.")


def main():
    args = parse_args()

    if args.cmd == 'build-hdf5':
        build_hdf5(args)
        return

    # cmd == 'fem'
    if args.batch > 0:
        run_batch(args)
        return

    # Single run
    npz_path = run_simulation(args)

    if not args.skip_plots:
        print("\nGenerating diagnostic plots...")
        make_diagnostic_plots(npz_path, args.output_dir)

    if not args.skip_video:
        print("\nRendering video...")
        render_video(npz_path, args.output_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
