"""SDF + marching-cubes via leia.microgen.graph_to_plate_mesh, in subprocess.

CUDA isolation is required: numba.cuda is sticky once imported in a process,
and we want the parent (which may hold the LEIA inference model on GPU 0) to
be free of meshing context. Subprocess is started with CUDA_VISIBLE_DEVICES
pinned so the SDF kernel runs on a dedicated GPU.
"""

import json
import os
import subprocess
import sys
import tempfile


def run_sdf_subprocess(graph_data, tiles, res_per_cell, stl_path, gpu_id=None,
                       timeout=600):
    """Run leia.microgen.graph_to_plate_mesh in a subprocess; write STL.

    Parameters
    ----------
    graph_data : dict — lattice graph spec (nodes, beams).
    tiles : 3-tuple — repeat counts per axis (e.g. (5, 5, 1) for canonical).
    res_per_cell : int — SDF grid resolution per unit cell.
    stl_path : str — output STL path; subprocess writes binary STL here.
    gpu_id : int | None — CUDA_VISIBLE_DEVICES value for the subprocess;
        None inherits the caller's environment.
    timeout : float — hard limit; returns None on expiry.

    Returns
    -------
    dict | None
        {"n_points", "n_cells", "bounds"} on success; None on failure.
    """
    # Repo root = three levels up from physics/meshing/_sdf.py.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    script = f"""
import sys
sys.path.insert(0, {repr(repo_root)})
import json

graph_data = json.loads({repr(json.dumps(graph_data))})
tiles = {tuple(tiles)}
res_per_cell = {int(res_per_cell)}
stl_path = {repr(stl_path)}

from leia.microgen import graph_to_plate_mesh

mesh = graph_to_plate_mesh(graph_data, tiles=tiles, res_per_cell=res_per_cell)
mesh.save(stl_path, binary=True)

stats = {{"n_points": mesh.n_points, "n_cells": mesh.n_cells,
          "bounds": list(mesh.bounds)}}
print("MESH_STATS:" + json.dumps(stats))
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="_plate_mesh_"
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True,
            timeout=timeout, env=env,
        )
        if result.returncode != 0:
            print(f"SDF subprocess stderr: {result.stderr[:500]}")
            return None
        for line in result.stdout.split("\n"):
            if line.startswith("MESH_STATS:"):
                return json.loads(line[len("MESH_STATS:"):])
        return {}
    except subprocess.TimeoutExpired:
        return None
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)
