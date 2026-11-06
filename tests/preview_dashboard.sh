#!/usr/bin/env bash
# Preview the GT-vs-LEIA dashboard render on one plate.
#
# Outputs (in tests/_artifacts/dashboard/):
#   <plate>_traj*.mp4    — side-by-side GT vs predicted, von Mises colormap
#   <plate>_fd.png       — force-displacement curve (if --save-fd-h5)
#
# ~1–2 min on 1 GPU. Skips with a clear message if production checkpoints
# are missing — train them first (see README.md).

set -euo pipefail

cd "$(dirname "$0")/.."

source tests/_conda_activate.sh leia

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ART_DIR="tests/_artifacts/dashboard"
TOK_CKPT="checkpoints/tokenizer/production_v2/best.pt"
DYN_CKPT="checkpoints/dynamics/production_v2/best_ar.pt"
PLATE="mg_046"

for f in "$TOK_CKPT" "$DYN_CKPT"; do
    if [[ ! -f "$f" ]]; then
        echo "SKIP preview_dashboard: missing $f"
        echo "  Train the production checkpoints first (see README.md)."
        exit 77   # SKIP (release_qualify.sh counts these separately)
    fi
done

rm -rf "$ART_DIR"
mkdir -p "$ART_DIR"

echo ">>> preview_dashboard — plate $PLATE -> $ART_DIR"

# Headless render needs xvfb-run for matplotlib + pyvista off-screen.
xvfb-run -a python scripts/render_e2e_dashboard.py \
    --tokenizer "$TOK_CKPT" \
    --dynamics  "$DYN_CKPT" \
    --npz-dir   data/microplate_lattice/npz \
    --fps-dir   data/microplate_lattice/fps \
    --mesh-dir  data/meshes/microplate_lattice \
    --plate-id  "$PLATE" \
    --n-trajs   1 \
    --fps       6 \
    --output-dir "$ART_DIR"

echo ""
echo "PASS preview_dashboard"
echo "Inspect:"
echo "  ls $ART_DIR"
echo "  mpv $ART_DIR/*.mp4"
