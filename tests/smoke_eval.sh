#!/usr/bin/env bash
# Smoke each eval.py subcommand on 1 plate × 1 in-dist val trajectory
# (5 e2e frames, 1 tokenizer frame, 1 gt-fd trajectory).
# Catches regressions in scripts/eval.py and leia/_eval_pipelines/*.py
# without spending the ~minutes a multi-plate README eval run takes.
#
# Asserts each subcommand both runs AND produces sane numerics — production
# v2 should hit disp_rel_l2 < 0.5, head_vm_corr > 0.7, vm_corr > 0.5 on a
# representative val frame. Loose floors so this catches regressions
# without flaking on minor numeric drift.
#
# Outputs in tests/_artifacts/_scratch/eval/. ~1–2 min on 1 GPU. Skips
# with a clear message if checkpoints are missing.

set -euo pipefail

cd "$(dirname "$0")/.."

source tests/_conda_activate.sh leia

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ART_DIR="tests/_artifacts/_scratch/eval"
TOK_CKPT="checkpoints/tokenizer/production_v2/best.pt"
DYN_CKPT="checkpoints/dynamics/production_v2/best_ar.pt"
PLATE="mg_046"

for f in "$TOK_CKPT" "$DYN_CKPT"; do
    if [[ ! -f "$f" ]]; then
        echo "SKIP smoke_eval: missing $f"
        echo "  Train the production checkpoints first (see README.md)."
        exit 77   # SKIP (release_qualify.sh counts these separately)
    fi
done

rm -rf "$ART_DIR"
mkdir -p "$ART_DIR"

# Single-plate manifest used by all 3 subcommands.
PLATES_JSON="$ART_DIR/plates.json"
python -c "import json; json.dump({'plates': ['$PLATE']}, open('$PLATES_JSON', 'w'))"

echo ">>> smoke_eval — eval.py {e2e,tokenizer,gt-fd} on $PLATE"

# ---------- eval.py e2e (1 plate, val traj 100, 1 frame) ----------
# Match the canonical val-split flags from the README eval recipe:
# --traj-indices 100 picks the first in-distribution val trajectory; --full-mesh + --mesh-dir
# + --grad-u-dir are needed for head_vm_corr and disp_vm_corr to be valid
# instead of NaN.
echo ""
echo "--- eval.py e2e (1 plate, val traj, 1 frame) ---"
python scripts/eval.py e2e \
    --tokenizer "$TOK_CKPT" \
    --dynamics  "$DYN_CKPT" \
    --npz-dir       data/microplate_lattice/npz \
    --fps-dir       data/microplate_lattice/fps \
    --grad-u-dir    data/microplate_lattice/grad_u \
    --mesh-dir      data/meshes/microplate_lattice \
    --curated-plates "$PLATES_JSON" \
    --max-trajs 101 --traj-indices "100" \
    --stress-mu 1.0 --stress-lmbda 10.0 \
    --full-mesh \
    --eval-frames 5 \
    --output-dir "$ART_DIR/e2e"

# ---------- eval.py tokenizer (1 plate, 1 frame) ----------
echo ""
echo "--- eval.py tokenizer (1 plate, 1 frame) ---"
python scripts/eval.py tokenizer \
    --checkpoint "$TOK_CKPT" \
    --mmap-dir   data/microplate_lattice/plate_dataset_curated_mmap_100k \
    --n-frames 1 \
    --stress-mu 1.0 --stress-lmbda 10.0 \
    --output-dir "$ART_DIR/tokenizer"

# ---------- eval.py gt-fd (GT-only; no ckpts) ----------
echo ""
echo "--- eval.py gt-fd (1 plate, 1 traj, GT only) ---"
python scripts/eval.py gt-fd \
    --npz-dir  data/microplate_lattice/npz \
    --mesh-dir data/meshes/microplate_lattice \
    --curated-plates "$PLATES_JSON" \
    --max-trajs 1 \
    --mu 1.0 --lmbda 10.0 \
    --output "$ART_DIR/fd_curves.h5" \
    --no-verify

# ---------- verify outputs + numeric sanity ----------
python - <<PYEOF
import glob, json, math, os, sys

import h5py

fail = []

# e2e: AR-rollout numerics on val traj should be sane on production v2.
e2e_jsons = glob.glob("$ART_DIR/e2e/*.json")
if not e2e_jsons:
    fail.append("e2e: no JSON output")
else:
    j = json.load(open(e2e_jsons[0]))
    o = j["overall"]
    disp = o["disp_rel_l2_median"]
    vm = o["head_vm_corr_median"]
    print(f"  e2e: disp_rel_l2_median={disp:.4f}  head_vm_corr_median={vm}")
    if not isinstance(disp, (int, float)) or math.isnan(disp) or disp >= 0.5:
        fail.append(f"e2e: disp_rel_l2_median={disp} (expected < 0.5)")
    if not isinstance(vm, (int, float)) or math.isnan(vm) or vm <= 0.7:
        fail.append(f"e2e: head_vm_corr_median={vm} (expected > 0.7)")

# tokenizer: stress correlation should be solid on the production ckpt.
tok_jsons = glob.glob("$ART_DIR/tokenizer/*metrics.json")
if not tok_jsons:
    fail.append("tokenizer: no metrics JSON")
else:
    j = json.load(open(tok_jsons[0]))
    vm = j.get("overall_median_vm_corr", j.get("median_vm_corr"))
    print(f"  tokenizer: overall_median_vm_corr={vm}")
    if not isinstance(vm, (int, float)) or math.isnan(vm) or vm <= 0.5:
        fail.append(f"tokenizer: vm_corr={vm} (expected > 0.5)")

# gt-fd: HDF5 must contain at least one trajectory with non-trivial force.
fd_path = "$ART_DIR/fd_curves.h5"
if not os.path.exists(fd_path):
    fail.append("gt-fd: no output H5")
else:
    with h5py.File(fd_path, "r") as h:
        plates = list(h.keys())
        if not plates:
            fail.append("gt-fd: H5 has no plate groups")
        else:
            f = h[plates[0]]["gen_force"][...]
            mag = float(abs(f).max())
            print(f"  gt-fd: plate={plates[0]} max|force|={mag:.4g}")
            if mag < 1e-3:
                fail.append(f"gt-fd: max|force|={mag} (expected > 1e-3)")

if fail:
    print("")
    for f in fail:
        print(f"  FAIL {f}")
    sys.exit(1)
print("  all sanity checks passed")
PYEOF

echo ""
echo "PASS smoke_eval"
