"""Subprocess pool for batch FEM ground-truth characterization.

Used by Figure 4 (`beam/fem_validate.py`, `beam/bench.py`) and Figure 5
(`landscape/fem.py`) to validate or label candidate plates against FEniCS
Neo-Hookean ground truth. Each worker runs FEniCS in an isolated spawn-
context process — MUMPS deadlocks in forked processes, and dolfin requires
clean process state per solve.

The FEM kernel itself lives in `physics/fem/neohookean/` (shared with
`data_curation/fem/run_plate_fem.py`). This module is the thin driver
layer: subprocess pool, GPU-friendly env vars, future tracking.
"""

# BLAS/LAPACK thread control — must be before numpy import.
# Workers in spawn context re-import this module.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# MPI transport — avoid InfiniBand exhaustion with many workers.
os.environ["OMPI_MCA_btl"] = "self,tcp"
os.environ["OMPI_MCA_pml"] = "ob1"

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from leia.regime import MICROPLATE_LATTICE


# The two loadings the anisotropy score consumes (stretch work / shear work);
# the same pair as `design_search.core.characterize.DIRECTIONS_2` on the LEIA
# side. The kernel default is all four directions (full characterization).
SCORED_DIRECTIONS = [(0, "stretch"), (2, "shear_y")]


def solve_characterization(task: dict) -> dict:
    """Run FEM characterization on a plate mesh: one loading ramp per direction
    (`task['directions']`, a list of (channel, label); default all four).

    Subprocess entry point — delegates to `physics.fem.neohookean.characterize_directions`,
    which lazy-imports dolfin (so FEniCS only loads in worker processes).

    Args:
        task: {
            'session_id': str,
            'points': np.ndarray [N, 3] float64,
            'cells': np.ndarray [M, 4] int64,
            'mu': float,
            'lmbda': float,
            'n_steps': int (per direction),
            'speeds': tuple (stretch, twist, shear_y, shear_z),
        }

    Returns:
        {
            'session_id': str,
            'status': 'ok' | 'partial' | 'failed',
            'directions': {label: {'outputs': [T+1,N,3], 'inputs': [T,4]}},
            'coords': np.ndarray [N, 3] float32,
            'topology': np.ndarray [M, 4] int64,
            'time_seconds': float,
            'errors': {direction: str},
        }
    """
    from physics.fem import characterize_directions

    coords = task['points'].astype(np.float64)
    topo = task['cells'].astype(np.intp)

    result = characterize_directions(
        coords, topo,
        mu=task['mu'],
        lmbda=task['lmbda'],
        n_steps=task['n_steps'],
        speeds=task['speeds'],
        directions=task.get('directions'),
    )
    result['session_id'] = task['session_id']
    return result


class FEMWorkerPool:
    """Manages a ProcessPoolExecutor for FEM ground-truth generation.

    Workers use spawn context (required for FEniCS) and run
    solve_characterization() in separate processes.
    """

    def __init__(self, max_workers: int = 50,
                 mu: float = MICROPLATE_LATTICE.mu,
                 lmbda: float = MICROPLATE_LATTICE.lmbda,
                 speeds: tuple = MICROPLATE_LATTICE.speeds):
        self.max_workers = max_workers
        self.mu = mu
        self.lmbda = lmbda
        self.speeds = speeds
        ctx = multiprocessing.get_context('spawn')
        self._pool = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
        )
        self._futures = {}  # session_id -> Future

    def submit(self, session_id: str, points: np.ndarray, cells: np.ndarray,
               n_steps: int = 15, directions=None) -> "concurrent.futures.Future":
        """Submit an FEM characterization job (all four directions unless
        `directions`, a list of (channel, label), narrows it).

        Args:
            session_id: Plate session ID.
            points: [N, 3] mesh node coordinates.
            cells: [M, 4] tet connectivity.
            n_steps: Steps per direction.

        Returns:
            Future that resolves to the result dict from solve_characterization().
        """
        task = {
            'session_id': session_id,
            'points': points,
            'cells': cells,
            'mu': self.mu,
            'lmbda': self.lmbda,
            'n_steps': n_steps,
            'speeds': self.speeds,
            'directions': directions,
        }
        future = self._pool.submit(solve_characterization, task)
        self._futures[session_id] = future
        return future

    def get_pending(self) -> list[str]:
        """Return session IDs with pending or running FEM jobs."""
        return [sid for sid, f in self._futures.items() if not f.done()]

    def shutdown(self, wait: bool = True):
        """Gracefully shut down the worker pool."""
        self._pool.shutdown(wait=wait)
