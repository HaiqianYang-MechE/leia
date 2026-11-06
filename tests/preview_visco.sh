#!/usr/bin/env bash
# Preview a stored Arruda-Boyce visco trajectory as MP4 + ParaView .vtu series.
#
# Reads from data/microplate_visco/microplate_visco.h5 (built offline by
# data_curation/fem/run_visco_load_reverse.py). For a *live* solver run
# see preview_visco_solve.sh.
#
# ~30s on the 363-node visco mesh.

set -euo pipefail

cd "$(dirname "$0")/.."

source tests/_conda_activate.sh leia

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ART_DIR="tests/_artifacts/visco"
H5="data/microplate_visco/microplate_visco.h5"

if [[ ! -f "$H5" ]]; then
    echo "SKIP preview_visco: missing $H5"
    echo "  Build the MicroPlate visco dataset first (see README.md)."
    exit 77   # SKIP (release_qualify.sh counts these separately)
fi

rm -rf "$ART_DIR"
mkdir -p "$ART_DIR"

echo ">>> preview_visco — $H5 traj=0 -> $ART_DIR"

xvfb-run -a python tests/preview_visco.py \
    --h5 "$H5" \
    --traj 0 \
    --n-frames 10 \
    --fps 4 \
    --out-dir "$ART_DIR"

echo ""
echo "PASS preview_visco"
echo "Inspect:"
echo "  ls $ART_DIR"
echo "  mpv $ART_DIR/visco_*.mp4"
echo "  paraview $ART_DIR/frame_005.vtu"
