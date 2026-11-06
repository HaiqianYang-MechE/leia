#!/usr/bin/env bash
# Smoke-test the design_search inference consumers past their first
# inference call, to catch the class of bug where a method missing on a
# dependency only surfaces at execution time (--help cannot see it).
#
# Pass/fail only. Discards outputs to tests/_artifacts/_scratch/.
# ~3–5 min on 1 GPU. Skips gracefully if production checkpoints are missing.

set -euo pipefail

cd "$(dirname "$0")/.."

source tests/_conda_activate.sh leia

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ART_DIR="tests/_artifacts/_scratch"
TOK_CKPT="checkpoints/tokenizer/production_v2/best.pt"
DYN_CKPT="checkpoints/dynamics/production_v2/best_ar.pt"

for f in "$TOK_CKPT" "$DYN_CKPT"; do
    if [[ ! -f "$f" ]]; then
        echo "SKIP smoke_consumers: missing $f"
        echo "  Train the production checkpoints first (see README.md)."
        exit 77   # SKIP (release_qualify.sh counts these separately)
    fi
done

rm -rf "$ART_DIR"
mkdir -p "$ART_DIR/landscape"

echo ">>> smoke_consumers — design_search inference paths"

# ---- synthesize a 1-candidate landscape input from a real plate graph ----
python - <<PYEOF
import json, os
graph = json.load(open("data/rve_bank/mg_046/graph.json"))
out = {
    "candidates": [{"id": "mg_046_smoke", "graph": graph}],
}
os.makedirs("$ART_DIR/landscape", exist_ok=True)
with open("$ART_DIR/landscape/candidates.json", "w") as f:
    json.dump(out, f)
print(f"  wrote $ART_DIR/landscape/candidates.json (1 candidate)")
PYEOF

# ---- score.py --metric=score (1 candidate, minimal directions) ----
echo ""
echo "--- score.py --metric=score on 1 candidate ---"
python -m design_search.landscape.score \
    --metric=score \
    --candidates "$ART_DIR/landscape/candidates.json" \
    --output-dir "$ART_DIR/score_out" \
    --max-candidates 1 \
    --steps-per-direction 2 \
    --num-shards 1 --shard-id 0

# ---- score.py --metric=consistency ----
echo ""
echo "--- score.py --metric=consistency on 1 candidate ---"
python -m design_search.landscape.score \
    --metric=consistency \
    --landscape-dirs "$ART_DIR/landscape" \
    --output-dir "$ART_DIR/consistency_out" \
    --n-steps 3 \
    --n-gpus 1 \
    --num-shards 1 --shard-id 0

# ---- verify the consistency.jsonl actually got non-error rows ----
python - <<PYEOF
import json, glob, sys

ok_score = 0
for p in glob.glob("$ART_DIR/score_out/*.jsonl"):
    for line in open(p):
        r = json.loads(line)
        if r.get("status") == "ok":
            ok_score += 1

ok_cons = 0
for p in glob.glob("$ART_DIR/consistency_out/*.jsonl"):
    for line in open(p):
        r = json.loads(line)
        if r.get("status") == "ok":
            ok_cons += 1

print(f"  score:       {ok_score} ok rows")
print(f"  consistency: {ok_cons} ok rows")

if ok_score == 0 or ok_cons == 0:
    print("FAIL smoke_consumers: at least one consumer produced no ok rows")
    sys.exit(1)
PYEOF

echo ""
echo "PASS smoke_consumers"
