"""Online per-candidate mesh generation for design-search runtime.

Thin driver around `physics.meshing.mesh_graph`. The actual SDF +
TetGen + surface-extraction kernel lives in `physics/meshing/`; this
module provides the design-search-flavored entry point with:

  - GPU round-robin (avoid GPU 0, which holds the inference model)
  - Inline TetGen (no subprocess startup overhead per candidate)

This file and `data_curation/mesh/build_meshes.py` share the same
mesh engine with the `CANONICAL_TETGEN` switch chain. They differ only
in `kill_on_timeout` (design_search inline for low online latency;
data_curation subprocess+kill for batch robustness) and `res_per_cell`
(design_search 10 from `ServingConfig`; data_curation 30 to match
canonical training artifacts).
"""

import os
import threading

from physics.meshing import mesh_graph, MeshResult, MeshingError

__all__ = ["MeshResult", "MeshingError", "generate_plate_mesh"]


# Round-robin GPU counter for spreading SDF across free GPUs.
# GPU 0 holds the inference tokenizer/dynamics; SDF subprocesses use a
# separate CUDA context, so they don't conflict — but routing them off
# GPU 0 still minimizes VRAM contention with the live inference state.
#
# Lock guards the read+increment: callers run under ThreadPoolExecutor
# (characterize.mesh_candidates_parallel, landscape/{score,novelty,pca}),
# and a naked counter would let two threads pin the same GPU.
_gpu_counter = 0
_gpu_counter_lock = threading.Lock()
_mesh_gpus = None  # resolved lazily from the visible devices


def _resolve_mesh_gpus():
    """Physical GPU ids for SDF subprocesses: every visible GPU except the
    first (which serves inference); on a single-GPU box, that GPU."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        ids = [v.strip() for v in visible.split(",") if v.strip()]
    else:
        try:
            import torch
            ids = [str(i) for i in range(torch.cuda.device_count())]
        except Exception:
            ids = []
    if len(ids) <= 1:
        return ids or ["0"]
    return ids[1:]


def _next_mesh_gpu():
    """Round-robin across available meshing GPUs (thread-safe)."""
    global _gpu_counter, _mesh_gpus
    with _gpu_counter_lock:
        if _mesh_gpus is None:
            _mesh_gpus = _resolve_mesh_gpus()
        gpu = _mesh_gpus[_gpu_counter % len(_mesh_gpus)]
        _gpu_counter += 1
    return gpu


def generate_plate_mesh(graph_data, config, *, extract_surface=False):
    """Generate a tiled plate tet mesh from an RVE graph (online use).

    Parameters
    ----------
    graph_data : dict — {"nodes": [[x,y,z],...], "beams": [{"idx":[i,j], ...}, ...]}
    config : ServingConfig — supplies tiles, mesh_res. Defaults: (5,5,1) and 10.
    extract_surface : bool — populate `MeshResult.surface_faces` for
        rendering. Defaults to False; current consumers (engine.create_session,
        FEM, scoring) only read points + cells. Pass True from rendering
        paths (filmstrips, STL export) only.

    Returns
    -------
    MeshResult with plate-scale coordinates. `surface_faces` is None
    unless `extract_surface=True`.

    Raises
    ------
    MeshingError on SDF or TetGen failure.
    """
    return mesh_graph(
        graph_data,
        tiles=tuple(config.tiles),
        res_per_cell=config.mesh_res,
        tetgen_kill_on_timeout=False,    # inline; matches today's behavior
        timeout=300,
        extract_surface=extract_surface,
        gpu_id=_next_mesh_gpu(),
    )
