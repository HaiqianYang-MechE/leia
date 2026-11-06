"""4-DOF direction-sweep characterization.

Used by `design_search/core/fem.py` to label candidate plates with a
ground-truth response in the scored directions (stretch and shear_y by
default, `DIRECTIONS_2`; all four on request). Wraps
`solve_neo_hookean` once per direction with one-hot actions.

The result dict shape matches the pre-extraction
`design_search/core/fem.py:solve_characterization` exactly so callers
of FEMWorkerPool see byte-equivalent return values.
"""

import time

import numpy as np

from .solver import solve_neo_hookean


# Direction definitions: (action_channel, label).
# Channel mapping matches data_curation/fem/run_plate_fem.py and the
# canonical `MICROPLATE_LATTICE_SPEEDS` in `leia.stress`.
DIRECTIONS = [
    (0, "stretch"),
    (1, "twist"),
    (2, "shear_y"),
    (3, "shear_z"),
]


def characterize_directions(
    coords,
    topo,
    *,
    mu,
    lmbda,
    n_steps=30,
    speeds=(0.15, 0.08, 0.15, 0.15),
    directions=None,
):
    """Run a 4-direction Neo-Hookean characterization sweep.

    Parameters
    ----------
    coords : (N, 3) float64
    topo : (M, 4) int
    mu, lmbda : float
    n_steps : int — load steps per direction
    speeds : 4-tuple — speed per DOF; default matches MICROPLATE_LATTICE_SPEEDS
    directions : list of (channel, label) tuples; default = all 4

    Returns
    -------
    dict shaped to match the legacy `solve_characterization` output:
      {
        'status': 'ok' | 'partial' | 'failed',
        'directions': {label: {'outputs': (T+1, N, 3) float32, 'inputs': (T, 4) int32}},
        'coords': (N, 3) float32,
        'topology': (M, 4) int64,
        'time_seconds': float,
        'errors': {label: str},
      }
    """
    coords = np.asarray(coords, dtype=np.float64)
    topo = np.asarray(topo, dtype=np.intp)

    if directions is None:
        directions = DIRECTIONS

    t_start = time.time()
    result = {
        "status": "ok",
        "directions": {},
        "coords": coords.astype(np.float32),
        "topology": topo.astype(np.int64),
        "time_seconds": 0.0,
        "errors": {},
    }

    for dir_idx, dir_label in directions:
        actions = np.zeros((n_steps, 4), dtype=np.int32)
        actions[:, dir_idx] = 1

        try:
            r = solve_neo_hookean(
                coords, topo, actions,
                mu=mu, lmbda=lmbda, speeds=speeds,
            )
        except Exception as e:
            result["errors"][dir_label] = str(e)
            result["status"] = "partial"
            continue

        if r["status"] != "ok":
            result["errors"][dir_label] = r.get("error", r["status"])
            result["status"] = "partial"
            continue

        result["directions"][dir_label] = {
            "outputs": r["displacements"],
            "inputs": actions,
        }

    if not result["directions"]:
        result["status"] = "failed"

    result["time_seconds"] = round(time.time() - t_start, 2)
    return result
