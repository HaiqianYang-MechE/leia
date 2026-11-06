#!/usr/bin/env bash
# Preview tokenizer + dynamics training on a tiny dataset (3 plates).
# Asserts loss decreases — catches a broken loss / dataloader / optimizer
# wiring before anyone wastes 8 GPUs on a multi-day production run.
#
# Outputs (in tests/_artifacts/train_small/):
#   tokenizer.log        full training stdout
#   dynamics.log         full training stdout
#   summary.json         first/last loss + delta + verdict (pass/fail)
#
# ~6–10 min on 1 GPU. Exits 77 (SKIP) if the tokenizer mmap is missing;
# skips the dynamics half with a clear message if the production tokenizer
# checkpoint or its structure-token mmap is missing.

set -euo pipefail

cd "$(dirname "$0")/.."

source tests/_conda_activate.sh leia

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ART_DIR="tests/_artifacts/train_small"
TOK_MMAP="data/microplate_lattice/plate_dataset_curated_mmap_100k"
DYN_MMAP="data/microplate_lattice/structure_tokens_prod_mmap"
TOK_CKPT="checkpoints/tokenizer/production_v2/best.pt"

if [[ ! -d "$TOK_MMAP" ]]; then
    echo "SKIP preview_train_small: missing $TOK_MMAP (README § 2 builds it)"
    exit 77
fi

rm -rf "$ART_DIR"
mkdir -p "$ART_DIR"
# Containment: training scripts write under checkpoints/; wipe the preview
# runs on every exit (pass or fail), never tests/_artifacts.
trap 'rm -rf checkpoints/tokenizer/preview_tok_* checkpoints/dynamics/preview_dyn_*' EXIT

echo ">>> preview_train_small — tiny tokenizer + dynamics training"

# ---------- 1) tokenizer: 200 steps, 3 plates, val every 20 steps ----------
echo ""
echo "--- tokenizer (200 steps, 3 plates) ---"
bash scripts/launch.sh train-tokenizer --gpus 1 --run-name preview_tok -- \
    --mmap-dir "$TOK_MMAP" \
    --max-rves 3 \
    --num-latents 64 --hidden-dim 128 \
    --num-encoder-layers 2 --num-decoder-layers 0 \
    --n-input-frames 3 --stress-head \
    --batch-size 4 \
    --max-steps 200 --val-interval 20 \
    --lr 3e-4 \
    --bf16 \
    2>&1 | tee "$ART_DIR/tokenizer.log"

# ---------- 2) dynamics: 5 epochs, 3 plates, uses production v2 tokenizer --
if [[ ! -f "$TOK_CKPT" || ! -d "$DYN_MMAP" ]]; then
    echo ""
    echo "SKIP dynamics: missing $TOK_CKPT or $DYN_MMAP"
    echo "  Train and promote the production tokenizer and build the dynamics mmap first (README § 3-4)."
else
    echo ""
    echo "--- dynamics (5 epochs, 3 plates) ---"
    bash scripts/launch.sh train-dynamics --gpus 1 --run-name preview_dyn -- \
        --mmap-dir "$DYN_MMAP" \
        --tokenizer-checkpoint "$TOK_CKPT" \
        --max-groups 3 \
        --epochs 5 --batch-size 4 \
        --method continuous_pf --rollout-steps 3 --velocity-input \
        --lr 1e-4 \
        --bf16 \
        2>&1 | tee "$ART_DIR/dynamics.log"
fi

# ---------- 3) loss-decrease assertion ----------
python - <<PYEOF
import json, re, sys, os

ART = "$ART_DIR"
results = {}
verdict = "pass"

# Tokenizer: 'train recon=0.123456'
tok_log = os.path.join(ART, "tokenizer.log")
if os.path.exists(tok_log):
    losses = [float(m.group(1)) for m in
              re.finditer(r"train recon=([0-9.]+)", open(tok_log).read())]
    if len(losses) >= 4:
        first = sum(losses[:len(losses)//2]) / (len(losses)//2)
        last = sum(losses[len(losses)//2:]) / (len(losses) - len(losses)//2)
        ratio = last / first if first > 0 else float("nan")
        ok = ratio < 0.95
        results["tokenizer"] = {
            "n_points": len(losses),
            "first_half_mean": round(first, 6),
            "last_half_mean": round(last, 6),
            "ratio": round(ratio, 4),
            "ok": ok,
        }
        if not ok:
            verdict = "fail"
        print(f"  tokenizer: first={first:.4f}, last={last:.4f}, "
              f"ratio={ratio:.3f} {'OK' if ok else 'FAIL'}")
    else:
        results["tokenizer"] = {"error": f"only {len(losses)} loss points"}
        verdict = "fail"

# Dynamics: 'train loss=0.1234'
dyn_log = os.path.join(ART, "dynamics.log")
if os.path.exists(dyn_log):
    losses = [float(m.group(1)) for m in
              re.finditer(r"train loss=([0-9.]+)", open(dyn_log).read())]
    if len(losses) >= 3:
        ok = losses[-1] < losses[0] * 0.95
        results["dynamics"] = {
            "n_epochs": len(losses),
            "first": round(losses[0], 4),
            "last": round(losses[-1], 4),
            "ratio": round(losses[-1] / losses[0], 4) if losses[0] > 0 else None,
            "ok": ok,
        }
        if not ok:
            verdict = "fail"
        print(f"  dynamics: first={losses[0]:.4f}, last={losses[-1]:.4f} "
              f"{'OK' if ok else 'FAIL'}")
    else:
        results["dynamics"] = {"error": f"only {len(losses)} epoch points"}
        verdict = "fail"

with open(os.path.join(ART, "summary.json"), "w") as f:
    json.dump({"verdict": verdict, "metrics": results}, f, indent=2)

print(f"\nverdict: {verdict}")
sys.exit(0 if verdict == "pass" else 1)
PYEOF

echo ""
echo "PASS preview_train_small"
echo "Inspect:"
echo "  cat $ART_DIR/summary.json"
echo "  less $ART_DIR/tokenizer.log"
echo "  less $ART_DIR/dynamics.log"
