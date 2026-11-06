"""Single mesh-engine API for lattice-graph → plate-scale tet meshes.

Public API:
    mesh_graph(graph, *, tiles, res_per_cell, …) → MeshResult
    MeshResult        — dataclass with points, cells, optional surface_faces
    MeshingError      — raised on SDF / TetGen failure
    CANONICAL_TETGEN  — switch chain matching on-disk canonical artifacts

This module is the single source of truth for mesh generation. Both
`data_curation/mesh/build_meshes.py` (offline batch canonical) and
`design_search/core/meshing.py` (online per-candidate) collapse onto it
via thin driver wrappers — they use the same switches and differ only
in subprocess+kill vs inline TetGen execution.
"""

from .api import mesh_graph, MeshResult, MeshingError
from ._tetgen import CANONICAL_TETGEN

__all__ = [
    "mesh_graph",
    "MeshResult",
    "MeshingError",
    "CANONICAL_TETGEN",
]
