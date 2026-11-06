"""Rendering / dataset helpers shared by `scripts/render_*.py`.

Generic primitives (numpy / imageio / pyvista) reusable across rendering
scripts. Script-specific things (panel composition, action labels, plate
rollout flow) stay in the calling script.
"""
import glob
import json
import os

import imageio
import numpy as np
import scipy.sparse as sp


def build_node_adjacency(topo, n_nodes):
    """Row-normalized averaging operator from tet connectivity.

    For each tet (4 nodes) emits 12 ordered (i, j) pairs (i ≠ j); duplicates
    are collapsed to a boolean adjacency, then row-normalized so the operator
    averages each node's value across its mesh neighbors.

    Returns a sparse (n_nodes, n_nodes) matrix usable as `avg_op @ u`.
    """
    pairs = np.concatenate([
        np.stack([topo[:, i], topo[:, j]], axis=1)
        for i in range(4) for j in range(4) if i != j
    ], axis=0)
    adj = sp.csr_matrix(
        (np.ones(len(pairs), dtype=np.float32), (pairs[:, 0], pairs[:, 1])),
        shape=(n_nodes, n_nodes))
    adj = (adj > 0).astype(np.float32)
    deg = np.maximum(np.asarray(adj.sum(axis=1)).ravel(), 1.0)
    return sp.diags(1.0 / deg) @ adj


def laplacian_smooth(u, avg_op, n_iters, alpha):
    """Iterative Laplacian smoothing of a per-node field.

    `u_new = (1 - alpha) * u + alpha * (avg_op @ u)`, repeated n_iters times.
    Removes isolated-node spikes in decoded displacement while preserving
    bulk deformation.
    """
    if n_iters <= 0 or alpha == 0.0:
        return u
    out = u.copy()
    for _ in range(n_iters):
        out = (1.0 - alpha) * out + alpha * (avg_op @ out)
    return out


def load_npz_plate_data(npz_dir, fps_dir, mesh_dir, plate_id, max_trajs):
    """Load per-plate trajectory data from the NPZ + mesh + FPS layout.

    Returns a dict with keys: coords, topology, outputs, fps_order, inputs.
    Coords are float64 (numerical stability for stress geometry); outputs
    and inputs are float32; topology is int64.
    """
    traj_files = sorted(glob.glob(os.path.join(npz_dir, plate_id, 'traj_*.npz')))
    if not traj_files:
        raise FileNotFoundError(
            f"No traj_*.npz files found in {npz_dir}/{plate_id}/")
    traj_files = traj_files[:max_trajs]

    first = np.load(traj_files[0])
    coords = first['coords'].astype(np.float64)

    fps_path = os.path.join(fps_dir, f'{plate_id}_fps_order.npy')
    if not os.path.exists(fps_path):
        raise FileNotFoundError(
            f"{fps_path} not found: rendering subsamples with the same FPS order "
            f"the tokenizer trained on. Run scripts/precompute_fps.py for "
            f"{plate_id} (README § 2 builds it for every plate).")
    fps_order = np.load(fps_path)

    mesh = np.load(os.path.join(mesh_dir, plate_id, 'mesh.npz'))
    topology = mesh['cells'].astype(np.int64)

    outputs_list, inputs_list = [], []
    for tf in traj_files:
        d = np.load(tf)
        outputs_list.append(d['outputs'].astype(np.float32))
        inputs_list.append(d['inputs'].astype(np.float32))

    return {
        'coords': coords,
        'topology': topology,
        'outputs': np.stack(outputs_list, axis=0),
        'fps_order': fps_order,
        'inputs': np.stack(inputs_list, axis=0),
    }


def discover_npz_plates(npz_dir, curated_plates_json=None):
    """Return a sorted list of plate IDs.

    With `curated_plates_json`, reads the `plates` key from that JSON.
    Otherwise scans `npz_dir` for subdirectories that contain `traj_*.npz`.
    """
    if curated_plates_json is not None:
        with open(curated_plates_json) as f:
            return sorted(json.load(f)['plates'])

    plate_ids = []
    for name in sorted(os.listdir(npz_dir)):
        subdir = os.path.join(npz_dir, name)
        if os.path.isdir(subdir) and glob.glob(os.path.join(subdir, 'traj_*.npz')):
            plate_ids.append(name)
    return plate_ids


def composite_overlay(panel_img, overlay_rgba, x, y):
    """Alpha-blend an RGBA overlay onto an RGB panel at (x, y).

    Trims the overlay to fit panel bounds; safe with overlays that extend
    past the panel edge. Returns a new image (does not mutate input).
    """
    h, w = overlay_rgba.shape[:2]
    panel_h, panel_w = panel_img.shape[:2]
    h = min(h, panel_h - y)
    w = min(w, panel_w - x)
    if h <= 0 or w <= 0:
        return panel_img
    alpha = overlay_rgba[:h, :w, 3:4].astype(np.float32) / 255.0
    rgb = overlay_rgba[:h, :w, :3].astype(np.float32)
    out = panel_img.copy()
    region = out[y:y+h, x:x+w].astype(np.float32)
    out[y:y+h, x:x+w] = (region * (1.0 - alpha) + rgb * alpha).astype(np.uint8)
    return out


def compose_grid(panels, n_rows, n_cols):
    """Concatenate `len(panels) == n_rows * n_cols` equal-size panels into a
    grid frame, padded so width and height are multiples of 16 (h264
    macroblock alignment).
    """
    if len(panels) != n_rows * n_cols:
        raise ValueError(
            f"compose_grid: need {n_rows * n_cols} panels, got {len(panels)}")
    rows = [np.concatenate(panels[r*n_cols:(r+1)*n_cols], axis=1)
            for r in range(n_rows)]
    frame = np.concatenate(rows, axis=0)
    h, w = frame.shape[:2]
    pad_h = ((h + 15) // 16) * 16
    pad_w = ((w + 15) // 16) * 16
    if pad_h > h or pad_w > w:
        padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
        padded[:h, :w] = frame
        frame = padded
    return frame


def open_writer(path, fps, codec='libx264', quality=8, macro_block_size=16):
    """Standardized imageio writer for h264 videos."""
    return imageio.get_writer(path, fps=fps, codec=codec, quality=quality,
                              macro_block_size=macro_block_size)


def fit_camera_to_envelope(env_min, env_max, panel_size, zoom=0.85):
    """Fit a pyvista isometric camera to a bbox; return camera_position.

    Pyvista is imported lazily — callers that don't use this function don't
    pay the import cost.
    """
    import pyvista as pv
    box = pv.Box(bounds=[env_min[0], env_max[0],
                         env_min[1], env_max[1],
                         env_min[2], env_max[2]])
    p = pv.Plotter(off_screen=True, window_size=(panel_size, panel_size))
    p.add_mesh(box, opacity=0.0)
    p.view_isometric()
    p.camera.zoom(zoom)
    cam = p.camera_position
    p.close()
    return cam
