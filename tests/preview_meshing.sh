#!/usr/bin/env bash
# Preview the graph -> tet mesh pipeline on 3 production plates.
#
# Outputs (per plate, in tests/_artifacts/meshing/):
#   <plate>.vtu          — ParaView-loadable tet mesh
#   <plate>.png          — isometric snapshot
#   <plate>.json         — mesh stats (n_points, n_cells, runtime)
#
# ~30s/plate at the canonical (5,5,1) tile config and res_per_cell=10.
# SDF kernel needs GPU 0 (mesh_graph default).

set -euo pipefail

cd "$(dirname "$0")/.."

source tests/_conda_activate.sh leia

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ART_DIR="tests/_artifacts/meshing"
rm -rf "$ART_DIR"
mkdir -p "$ART_DIR"

echo ">>> preview_meshing — 3 plates -> .vtu + .png in $ART_DIR"

# Headless PNG render needs an X server; use xvfb-run.
xvfb-run -a python tests/preview_meshing.py \
    --plates mg_007 mg_046 mg_111 \
    --rve-bank data/rve_bank \
    --out-dir "$ART_DIR"

echo ""
echo "PASS preview_meshing"
echo "Inspect:"
echo "  ls $ART_DIR"
echo "  paraview $ART_DIR/mg_046.vtu"
