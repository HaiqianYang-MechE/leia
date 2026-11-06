"""Stable kernel for the LEIA design-search package.

Surrogate-eval primitive layer used by both Figure 4 (`design_search.beam`)
and Figure 5 (`design_search.landscape`) pipelines. `meshing` and `fem`
are thin drivers around `physics.meshing` and `physics.fem.neohookean`.
The names re-exported below are the public API.
"""

from design_search.core.config import ServingConfig
from design_search.core.engine import ModelEngine, Session, StepResult
from design_search.core.meshing import (
    MeshResult,
    MeshingError,
    generate_plate_mesh,
)
from design_search.core.fem import FEMWorkerPool, solve_characterization
from design_search.core.graph import validate_graph
from design_search.core.objectives import (
    anisotropy_score,
    compute_work,
    volume_fraction,
)
from design_search.core.characterize import (
    DIRECTIONS_2,
    DIRECTIONS_4,
    sync_characterize,
    mesh_candidates_parallel,
    load_seed_graph,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    "ServingConfig",
    "ModelEngine",
    "Session",
    "StepResult",
    "MeshResult",
    "MeshingError",
    "generate_plate_mesh",
    "FEMWorkerPool",
    "solve_characterization",
    "validate_graph",
    "anisotropy_score",
    "compute_work",
    "volume_fraction",
    "DIRECTIONS_2",
    "DIRECTIONS_4",
    "sync_characterize",
    "mesh_candidates_parallel",
    "load_seed_graph",
    "save_checkpoint",
    "load_checkpoint",
]
