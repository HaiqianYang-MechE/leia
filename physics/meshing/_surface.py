"""Surface-face extraction from a tet mesh, with outward-normal orientation.

Each tet has 4 triangular faces. A boundary face appears in exactly one tet
(interior faces are shared between two). After identifying boundaries, each
triangle is oriented so its normal points away from the tet interior — this
makes the result usable as a render surface or as a basis for boundary BCs.

Vectorized via numpy: the per-tet Python loop is replaced with a single
`np.unique(axis=0)` over all (4M, 3) faces. ~50–100× speedup on canonical
1.37M-tet meshes.
"""

import numpy as np


# Tet local-vertex layout for the 4 faces, paired with the local index of
# the opposite vertex (used for outward-normal orientation).
_FACE_LOCAL = np.array([
    [0, 1, 2],  # face 0; opposite vertex 3
    [0, 1, 3],  # face 1; opposite vertex 2
    [0, 2, 3],  # face 2; opposite vertex 1
    [1, 2, 3],  # face 3; opposite vertex 0
], dtype=np.int64)
_OPP_LOCAL = np.array([3, 2, 1, 0], dtype=np.int64)


def extract_surface_faces(cells, points=None):
    """Extract boundary triangular faces from a tet mesh.

    Parameters
    ----------
    cells : (M, 4) int — tet connectivity.
    points : (N, 3) float | None — node coordinates. Required for outward-
        normal orientation; if None, returned faces use natural tet ordering
        with no consistency guarantee.

    Returns
    -------
    (F, 3) int64 surface triangle faces. When `points` is provided, each
    face is oriented so its normal points away from the tet's opposite
    vertex (i.e., outward).
    """
    cells = np.asarray(cells)
    if cells.size == 0:
        return np.empty((0, 3), dtype=np.int64)

    # All 4 faces × M tets, in original vertex ordering. Shape (4M, 3).
    all_faces = cells[:, _FACE_LOCAL].reshape(-1, 3)
    # Opposite vertex aligned with each face row. Shape (4M,).
    all_opp = cells[:, _OPP_LOCAL].reshape(-1)

    # Canonical face key = sorted vertex triple. Boundary faces appear once;
    # interior faces appear twice (shared by two tets).
    faces_sorted = np.sort(all_faces, axis=1)
    _, inverse, counts = np.unique(
        faces_sorted, axis=0, return_inverse=True, return_counts=True
    )
    is_boundary = counts[inverse] == 1

    boundary_faces = all_faces[is_boundary]   # (F, 3) — original ordering
    boundary_opp = all_opp[is_boundary]       # (F,)

    if points is None:
        return boundary_faces.astype(np.int64)

    # Vectorized outward-normal orientation: flip winding when the face
    # normal points toward the opposite vertex (= toward the tet interior).
    p0 = points[boundary_faces[:, 0]]
    p1 = points[boundary_faces[:, 1]]
    p2 = points[boundary_faces[:, 2]]
    p_opp = points[boundary_opp]

    normal = np.cross(p1 - p0, p2 - p0)
    center = (p0 + p1 + p2) / 3.0
    to_opp = p_opp - center
    flip = np.einsum('ij,ij->i', normal, to_opp) > 0

    surface = boundary_faces.copy()
    surface[flip] = boundary_faces[flip][:, [0, 2, 1]]   # swap v1, v2

    return surface.astype(np.int64)
