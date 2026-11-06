"""Objective functions for lattice topology beam search.

volume_fraction: solid volume fraction from graph geometry
compute_work: area under F-D curve via trapezoidal integration
anisotropy_score: composite score = stretch_work / (shear_work * vol_frac)
"""

import math

import numpy as np


def volume_fraction(graph: dict) -> float:
    """Compute solid volume fraction of an RVE graph.

    Sum of beam cylinder volumes (pi * r_avg^2 * length) divided by unit cube
    volume (1.0). Uses average radius for tapered beams.

    Typical range: 0.001–0.07 for production lattice graphs.
    """
    nodes = graph["nodes"]
    beams = graph["beams"]
    solid_vol = 0.0

    for beam in beams:
        i, j = beam["idx"]
        pi_node = nodes[i]
        pj_node = nodes[j]
        length = math.sqrt(sum((pj_node[k] - pi_node[k]) ** 2 for k in range(3)))

        r_start = beam.get("r_start", 0.05)
        r_end = beam.get("r_end", 0.05)
        r_avg = (r_start + r_end) / 2.0

        solid_vol += math.pi * r_avg ** 2 * length

    # Unit cube volume = 1.0
    return solid_vol


def compute_work(disp: list[float], force: list[float]) -> float:
    """Compute work (energy) as area under the F-D curve.

    Uses trapezoidal integration of |force| with respect to |disp|.
    Absolute values because gen_force is negative for tensile stretch
    (reaction force convention in `design_search/core/engine.py`).
    """
    d = np.abs(np.array(disp, dtype=np.float64))
    f = np.abs(np.array(force, dtype=np.float64))
    if len(d) < 2:
        return 0.0
    return float(np.trapezoid(f, d))


def anisotropy_score(
    graph: dict,
    char_result: dict,
    eps: float = 1e-8,
    formula: str = "product",
    lambda_vf: float = 100.0,
) -> float:
    """Composite anisotropy score: hard to stretch, easy to bend, lightweight.

    Two formulas:
      formula="product"  (default, original):
          s = stretch_work / (shear_work * vol_frac + eps)
          1/vol_frac singularity rewards arbitrarily thin lattices.

      formula="additive" (new):
          s = stretch_work / shear_work - lambda_vf * vol_frac
          Bounded; lambda_vf controls the trade-off between anisotropy
          (stretch/shear ratio) and lightness (vol_frac penalty).

    char_result must have keys "stretch" and "shear_y", each with
    "disp" and "force" lists (from sync_characterize).
    """
    stretch = char_result.get("stretch", {})
    shear = char_result.get("shear_y", {})

    stretch_disp = stretch.get("disp", [])
    shear_disp = shear.get("disp", [])

    # Both directions must have produced an F-D curve. At high mesh resolution
    # one Newton solve can diverge while the other succeeds; treating that as
    # a real score (with shear_work ≈ 0 in the denominator) blows up to 1e8.
    # Return 0 instead — caller filters non-positive scores.
    if len(stretch_disp) < 2 or len(shear_disp) < 2:
        return 0.0

    stretch_work = compute_work(stretch_disp, stretch.get("force", []))
    shear_work = compute_work(shear_disp, shear.get("force", []))

    if stretch_work < eps or shear_work < eps:
        return 0.0

    vol_frac = volume_fraction(graph)

    if formula == "product":
        return stretch_work / (shear_work * vol_frac + eps)
    elif formula == "additive":
        return stretch_work / shear_work - lambda_vf * vol_frac
    else:
        raise ValueError(f"Unknown score formula: {formula!r} "
                         f"(expected 'product' or 'additive')")


# ========== SELF-TEST ==========

if __name__ == "__main__":
    import json
    import os

    # --- Test volume_fraction ---
    graph_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "rve_bank", "mg_046", "graph.json",
    )
    with open(graph_path) as f:
        g = json.load(f)

    vf = volume_fraction(g)
    print(f"mg_046: {len(g['nodes'])} nodes, {len(g['beams'])} beams, vol_frac = {vf:.6f}")

    # Manual check: all beams have r=0.01 or similar, lengths ~0.3-1.0
    # Expected: small number (sparse lattice)
    assert 0.0001 < vf < 0.5, f"vol_frac {vf} out of expected range"

    # --- Test compute_work ---
    # Linear F-D: force = disp, from 0 to 2 → area = 0.5 * 2 * 2 = 2.0
    disp = [0.0, 1.0, 2.0]
    force = [0.0, 1.0, 2.0]
    w = compute_work(disp, force)
    print(f"\nLinear F-D [0,2]: work = {w:.4f} (expected 2.0)")
    assert abs(w - 2.0) < 1e-10

    # Negative forces (reaction force convention)
    force_neg = [0.0, -1.0, -2.0]
    w_neg = compute_work(disp, force_neg)
    print(f"Negative forces: work = {w_neg:.4f} (expected 2.0)")
    assert abs(w_neg - 2.0) < 1e-10

    # Constant force = 3.0 over disp [0, 1, 2] → area = 3.0 * 2 = 6.0
    w_const = compute_work([0, 1, 2], [3, 3, 3])
    print(f"Constant force=3: work = {w_const:.4f} (expected 6.0)")
    assert abs(w_const - 6.0) < 1e-10

    # Empty / single point
    assert compute_work([], []) == 0.0
    assert compute_work([0], [0]) == 0.0

    # --- Test anisotropy_score ---
    fake_char = {
        "stretch": {"disp": [0, 1, 2], "force": [0, 5, 10]},  # work = 10.0
        "shear_y": {"disp": [0, 1, 2], "force": [0, 1, 2]},   # work = 2.0
    }
    score = anisotropy_score(g, fake_char)
    expected = 10.0 / (2.0 * vf)
    print(f"\nAnisotropy score: {score:.2f} (expected {expected:.2f})")
    assert abs(score - expected) / expected < 1e-6

    # Zero stretch work → score = 0
    zero_char = {
        "stretch": {"disp": [0, 0, 0], "force": [0, 0, 0]},
        "shear_y": {"disp": [0, 1, 2], "force": [0, 1, 2]},
    }
    assert anisotropy_score(g, zero_char) == 0.0

    print("\nAll tests passed.")
