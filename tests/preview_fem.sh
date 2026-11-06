#!/usr/bin/env bash
# Preview a stored FEM trajectory as MP4 + ParaView .vtu series.
#
# Outputs (in tests/_artifacts/fem/):
#   frame_NNN.vtu        per-frame mesh + displacement (ParaView)
#   frame_NNN.png        deformed mesh colored by displacement magnitude
#   <plate>_<traj>.mp4   stitched animation
#   stats.json           plate / traj / counts
#
# Reads a precomputed FEM trajectory (no fem env / dolfin needed).
# ~1–2 min on the 309K-node mg_046 plate.

set -euo pipefail

cd "$(dirname "$0")/.."

source tests/_conda_activate.sh leia

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ART_DIR="tests/_artifacts/fem"
PLATE="mg_046"
NPZ_DIR="data/microplate_lattice/npz/$PLATE"

# Pick the first available trajectory.
NPZ=$(ls "$NPZ_DIR"/traj_*.npz 2>/dev/null | head -1 || true)
if [[ -z "$NPZ" ]]; then
    echo "SKIP preview_fem: no trajectories under $NPZ_DIR"
    echo "  Build the MicroPlate lattice dataset first (see README.md)."
    exit 77   # SKIP (release_qualify.sh counts these separately)
fi

rm -rf "$ART_DIR"
mkdir -p "$ART_DIR"

echo ">>> preview_fem — $NPZ -> $ART_DIR"

xvfb-run -a python tests/preview_fem.py \
    --npz "$NPZ" \
    --out-dir "$ART_DIR" \
    --n-frames 10 \
    --fps 4

echo ""
echo "PASS preview_fem"
echo "Inspect:"
echo "  ls $ART_DIR"
echo "  mpv $ART_DIR/${PLATE}_*.mp4"
echo "  paraview $ART_DIR/frame_005.vtu"
