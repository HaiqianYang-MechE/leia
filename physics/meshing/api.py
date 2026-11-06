"""Unified mesh-engine API: lattice graph → plate-scale tet mesh.

Single function `mesh_graph(...)` that both `data_curation/` (offline batch
canonical-mesh generation) and `design_search/core/meshing.py` (online
per-candidate mesh generation) collapse onto. Both use `CANONICAL_TETGEN`
switches (matching on-disk artifacts); they differ only on
`tetgen_kill_on_timeout` (subprocess+kill for offline robustness vs inline
for online latency) and `res_per_cell` (30 offline, 10 online).

Output coordinate system is plate scale — `tiles[i]*0.98` per axis. For the
canonical training/FEM use case, pass `tiles=(5, 5, 1)` → x/y in [-4.9, 4.9],
z in [-0.98, 0.98]. The trained tokenizer's positional encoding assumes this
scale; using a different scale would silently corrupt every downstream
computation (FEM speeds, gradient operators, force assembly).
"""

import os
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ._sdf import run_sdf_subprocess
from ._tetgen import run_tetgen, CANONICAL_TETGEN
from ._surface import extract_surface_faces


@dataclass
class MeshResult:
    points: np.ndarray                          # (N, 3) float64 — plate-scale coords
    cells: np.ndarray                           # (M, 4) int64 — tet connectivity
    surface_faces: Optional[np.ndarray] = None  # (F, 3) int64 — only if extract_surface


class MeshingError(Exception):
    pass


def mesh_graph(
    graph_data: dict,
    *,
    tiles: Tuple[int, int, int],
    res_per_cell: int = 10,
    tetgen_kill_on_timeout: bool = True,
    timeout: float = 300.0,
    sdf_timeout: float = 600.0,
    extract_surface: bool = True,
    gpu_id: Optional[int] = None,
) -> MeshResult:
    """Lattice graph → plate-scale tet mesh.

    Parameters
    ----------
    graph_data : dict — {"nodes": [[x, y, z], ...], "beams": [{"idx": [i, j], ...}, ...]}
    tiles : 3-tuple — REQUIRED. Tile counts per axis. (5, 5, 1) for canonical
        training/FEM (plate width 9.8 in x/y); (1, 1, 1) for single-cell.
    res_per_cell : int — SDF grid resolution per unit cell. Canonical training
        meshes use 30; design_search interactive uses 10.
    tetgen_kill_on_timeout : bool — robustness knob. True = subprocess+kill
        (data_curation offline); False = inline (design_search online).
    timeout : float — per-switch TetGen wall-clock limit. Enforced both
        modes: SIGKILL when kill_on_timeout=True, thread-watchdog otherwise.
    sdf_timeout : float — wall-clock limit for the SDF + marching-cubes
        subprocess. Default 600s comfortably covers canonical (5,5,1) at
        res_per_cell=30 (~10–30s observed); tighten for online use.
    extract_surface : bool — if True, populate `MeshResult.surface_faces`
        with outward-oriented boundary triangles (rendering / boundary BCs).
    gpu_id : int | None — GPU index for the SDF subprocess. None = inherit the
        caller's CUDA_VISIBLE_DEVICES (do not override it).

    Returns
    -------
    MeshResult with plate-scale coordinates.

    Raises
    ------
    MeshingError if the SDF or any TetGen-switch attempt fails.
    """
    import pyvista as pv

    # Step 1: SDF + marching-cubes via subprocess (CUDA isolation).
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        stl_path = tmp.name
    try:
        stats = run_sdf_subprocess(
            graph_data,
            tiles=tiles,
            res_per_cell=res_per_cell,
            stl_path=stl_path,
            gpu_id=gpu_id,
            timeout=sdf_timeout,
        )
        if stats is None:
            raise MeshingError("SDF subprocess failed or timed out")
        surface = pv.read(stl_path)
    finally:
        if os.path.exists(stl_path):
            os.unlink(stl_path)

    # Step 2: TetGen with progressive CANONICAL_TETGEN fallback chain.
    result = run_tetgen(surface, CANONICAL_TETGEN, timeout=timeout,
                        kill_on_timeout=tetgen_kill_on_timeout)
    if result is None:
        raise MeshingError(
            f"TetGen failed for all switches: {CANONICAL_TETGEN}")
    points, cells = result

    # Step 3 (optional): extract surface faces.
    surface_faces = extract_surface_faces(cells, points) if extract_surface else None

    return MeshResult(points=points, cells=cells, surface_faces=surface_faces)
