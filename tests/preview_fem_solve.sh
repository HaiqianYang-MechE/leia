#!/usr/bin/env bash
# Live Neo-Hookean FEM solve on a small lattice mesh + render the result.
#
# Single-env pipeline (leia env has both torch and dolfin / FEniCS 2019.1):
#   1. physics.meshing.mesh_graph generates a small (8K-node) tet mesh
#      from the mg_111 plate graph, saves to NPZ.
#   2. physics.fem.solve_neo_hookean runs 5 stretch steps in dolfin /
#      FEniCS 2019.1, saves trajectory NPZ.
#   3. preview_fem.py renders the trajectory NPZ to MP4 + .vtu.
#
# This is the only test that exercises the live FEniCS solver, so it
# catches kernel-side regressions (dolfin upgrades, solver bugs) that
# preview_fem.sh (stored data) cannot.
#
# ~3–5 min total.

set -euo pipefail

cd "$(dirname "$0")/.."

source tests/_conda_activate.sh leia

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ART_DIR="tests/_artifacts/fem_solve"
PLATE="mg_111"   # 8K-node mesh — small enough for a quick solve

rm -rf "$ART_DIR"
mkdir -p "$ART_DIR"

MESH_NPZ="$ART_DIR/mesh.npz"
TRAJ_NPZ="$ART_DIR/traj.npz"

# ---- stage 1: mesh ----
echo ">>> preview_fem_solve [1/3] mesh_graph"
python tests/preview_fem_meshgen.py \
    --graph-json "data/rve_bank/$PLATE/graph.json" \
    --out-npz "$MESH_NPZ"

# ---- stage 2: live FEM solve ----
echo ""
echo ">>> preview_fem_solve [2/3] solve_neo_hookean"
python tests/preview_fem_solve.py \
    --mesh-npz "$MESH_NPZ" \
    --out-npz "$TRAJ_NPZ" \
    --n-steps 5 \
    --mu 1.0 --lmbda 10.0

# ---- stage 3: render ----
echo ""
echo ">>> preview_fem_solve [3/3] render trajectory"
xvfb-run -a python tests/preview_fem.py \
    --npz "$TRAJ_NPZ" \
    --out-dir "$ART_DIR" \
    --n-frames 6 --fps 3

echo ""
echo "PASS preview_fem_solve"
echo "Inspect:"
echo "  ls $ART_DIR"
echo "  mpv $ART_DIR/live_solve_traj.mp4"
