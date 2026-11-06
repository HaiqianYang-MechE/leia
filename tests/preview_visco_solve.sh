#!/usr/bin/env bash
# Live Arruda-Boyce 3-branch visco solve + render.
# Two-stage pipeline:
#
#   1. fenicsx env — physics.fem.solve_arruda_boyce runs 3 stretch action
#                    steps × 2 sub-steps on an 8×8×2 uniform tet mesh.
#                    Saves trajectory NPZ.
#   2. leia env    — preview_fem.py renders the trajectory NPZ to MP4 + .vtu.
#
# Exercises the live dolfinx-side visco kernel — the only test that does so.
# preview_visco.sh (stored data) cannot catch a fenicsx-side regression.
#
# ~2–4 min total. Skips with a clear message if `fenicsx` env is missing.

set -euo pipefail

cd "$(dirname "$0")/.."

if ! conda env list 2>/dev/null | grep -qE "^fenicsx\s"; then
    echo "SKIP preview_visco_solve: missing 'fenicsx' conda env"
    exit 77   # SKIP (release_qualify.sh counts these separately)
fi

ART_DIR="tests/_artifacts/visco_solve"
TRAJ_NPZ="$ART_DIR/traj.npz"

rm -rf "$ART_DIR"
mkdir -p "$ART_DIR"

# ---- stage 1: live visco solve in fenicsx env ----
echo ">>> preview_visco_solve [1/2] solve_arruda_boyce in fenicsx env"
source tests/_conda_activate.sh fenicsx
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
python tests/preview_visco_solve.py \
    --out-npz "$TRAJ_NPZ" \
    --n-steps 3 --n-substeps 2 --mesh-res 8

# ---- stage 2: render in leia env ----
echo ""
echo ">>> preview_visco_solve [2/2] render trajectory in leia env"
source tests/_conda_activate.sh leia
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
xvfb-run -a python tests/preview_fem.py \
    --npz "$TRAJ_NPZ" \
    --out-dir "$ART_DIR" \
    --n-frames 6 --fps 3

echo ""
echo "PASS preview_visco_solve"
echo "Inspect:"
echo "  ls $ART_DIR"
echo "  mpv $ART_DIR/visco_live_traj.mp4"
