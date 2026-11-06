#!/usr/bin/env bash
# Single-command end-to-end reproduction of fig_ood_landscape.pdf.
#
# Runs the 15-stage OOD-landscape pipeline (random walk + novelty search +
# LEIA scoring + FEM ground truth + encoder-cycle consistency + held-out
# encode + OOD-head training + held-out characterization + 4-panel figure)
# against the v2 production checkpoint. All artifacts land under
# design_search/outputs/repro_ood_<run_name>/. Each stage is sentinel-gated
# so a re-run skips finished work.
#
# Usage:
#   bash design_search/repro/figure5.sh <run_name> [--fast] [--from-stage N]
#
# Env knobs (override before invocation):
#   OOD_RANDOM_SEED               (default 20260503)
#   OOD_WALK_DEPTH                (default 55)
#   OOD_NOVELTY_B / _K / _D       (default 2 / 6 / 8)
#   OOD_FEM_WORKERS               (default 150)
#   OOD_FEM_WORKERS_FALLBACK      (default 100)
#   OOD_FEM_RAM_DANGER_PCT        (default 80; %)
#   OOD_N_GPUS                    (default 8)
#   PYTHONUNBUFFERED is forced to 1 and PYTHONHASHSEED to 0 (the novelty search seeds
#   from string hashes; without it the candidate set changes on every run).

set -u -o pipefail
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

# ---------- arg parse ----------
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <run_name> [--fast] [--from-stage N]" >&2
    exit 1
fi

RUN_NAME="$1"; shift
FAST=0
FROM_STAGE=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast)        FAST=1; shift ;;
        --from-stage)  FROM_STAGE="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

# ---------- config ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="$REPO_ROOT/design_search/outputs/repro_ood_${RUN_NAME}"

OOD_RANDOM_SEED="${OOD_RANDOM_SEED:-20260503}"
OOD_WALK_DEPTH="${OOD_WALK_DEPTH:-55}"
OOD_NOVELTY_B="${OOD_NOVELTY_B:-2}"
OOD_NOVELTY_K="${OOD_NOVELTY_K:-6}"
OOD_NOVELTY_D="${OOD_NOVELTY_D:-8}"
OOD_FEM_WORKERS="${OOD_FEM_WORKERS:-150}"
OOD_FEM_WORKERS_FALLBACK="${OOD_FEM_WORKERS_FALLBACK:-100}"
OOD_FEM_RAM_DANGER_PCT="${OOD_FEM_RAM_DANGER_PCT:-80}"
OOD_CONSISTENCY_NSTEPS=10   # fixed: the OOD feature layout in ood/pipeline.py assumes 10 steps
OOD_N_GPUS="${OOD_N_GPUS:-8}"

# Fast mode: smoke-test config (~30 min total)
if (( FAST )); then
    OOD_WALK_DEPTH=4
    OOD_NOVELTY_B=2
    OOD_NOVELTY_K=2
    OOD_NOVELTY_D=2
    OOD_FEM_WORKERS=32
    OOD_FEM_WORKERS_FALLBACK=16
    echo "FAST mode: walk_depth=$OOD_WALK_DEPTH, novelty BKD=$OOD_NOVELTY_B/$OOD_NOVELTY_K/$OOD_NOVELTY_D, FEM workers=$OOD_FEM_WORKERS"
fi
export OOD_FEM_RAM_DANGER_PCT

# ---------- conda env ----------
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "$CONDA_DEFAULT_ENV" != "leia" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate leia
fi

mkdir -p "$RUN_ROOT/walk" "$RUN_ROOT/novelty" "$RUN_ROOT/consistency" \
         "$RUN_ROOT/ood/ckpts" "$RUN_ROOT/heldout" "$RUN_ROOT/figures" \
         "$RUN_ROOT/logs"

GLOBAL_LOG="$RUN_ROOT/logs/orchestrator.log"
exec > >(tee -a "$GLOBAL_LOG") 2>&1

echo "================================================================"
echo "  figure5.sh  $(date -Iseconds)"
echo "  run_name=$RUN_NAME  run_root=$RUN_ROOT"
echo "  walk_depth=$OOD_WALK_DEPTH  novelty B/K/D=$OOD_NOVELTY_B/$OOD_NOVELTY_K/$OOD_NOVELTY_D"
echo "  FEM workers=$OOD_FEM_WORKERS (fallback $OOD_FEM_WORKERS_FALLBACK; danger ${OOD_FEM_RAM_DANGER_PCT}%)"
echo "  random_seed=$OOD_RANDOM_SEED  consistency_nsteps=$OOD_CONSISTENCY_NSTEPS  gpus=$OOD_N_GPUS"
echo "  from_stage=$FROM_STAGE  fast=$FAST"
echo "================================================================"

# ---------- helpers ----------
stage_log() { echo "$RUN_ROOT/logs/stage$(printf '%02d' "$1")_$2.log"; }
have()      { [[ -f "$1" || -d "$1" ]]; }

run_stage() {
    # $1 stage_num, $2 stage_label, $3 sentinel, $4... cmd
    local n="$1"; local label="$2"; local sentinel="$3"; shift 3
    if (( n < FROM_STAGE )); then
        echo "STAGE $n $label SKIP (--from-stage $FROM_STAGE)"
        return 0
    fi
    if have "$sentinel"; then
        echo "STAGE $n $label SKIP (sentinel exists: $sentinel)"
        return 0
    fi
    local lf
    lf="$(stage_log "$n" "$label")"
    echo "STAGE $n $label START  log=$lf"
    if "$@" >>"$lf" 2>&1; then
        echo "STAGE $n $label DONE"
    else
        local ec=$?
        echo "STAGE $n $label FAIL exit=$ec  (last 20 lines of log:)"
        tail -20 "$lf" 2>/dev/null || true
        exit $ec
    fi
}

run_stage_sharded() {
    # $1 stage_num, $2 label, $3 sentinel, $4 ngpus, $5... per-shard cmd parts
    # Per-shard cmd will be invoked with extra args:  --shard-id N --num-shards $4 --device cuda:N
    local n="$1"; local label="$2"; local sentinel="$3"; local ngpus="$4"; shift 4
    # Returns 0 only when the stage actually ran; 3 when skipped, so callers
    # do not merge shards or write the sentinel for work that did not happen.
    if (( n < FROM_STAGE )); then
        echo "STAGE $n $label SKIP (--from-stage $FROM_STAGE)"
        return 3
    fi
    if have "$sentinel"; then
        echo "STAGE $n $label SKIP (sentinel exists: $sentinel)"
        return 3
    fi
    local lf_base
    lf_base="$RUN_ROOT/logs/stage$(printf '%02d' "$n")_${label}"
    echo "STAGE $n $label START  ngpus=$ngpus  logs=${lf_base}_shard{0..$((ngpus-1))}.log"
    local pids=()
    for ((i = 0; i < ngpus; i++)); do
        ( "$@" --shard-id "$i" --num-shards "$ngpus" --device "cuda:$i" \
              >"${lf_base}_shard${i}.log" 2>&1 ) &
        pids+=($!)
    done
    local fail=0
    for pid in "${pids[@]}"; do
        wait "$pid" || fail=1
    done
    if (( fail )); then
        echo "STAGE $n $label FAIL  (one or more shards failed; check ${lf_base}_shard*.log)"
        for ((i = 0; i < ngpus; i++)); do
            echo "  shard $i tail:"
            tail -10 "${lf_base}_shard${i}.log" 2>/dev/null | sed 's/^/    /'
        done
        exit 1
    fi
    echo "STAGE $n $label DONE"
}

# ---------- stage 1: random-walk candidates ----------
run_stage 1 walk "$RUN_ROOT/walk/candidates.json" \
    python -m design_search.landscape.walk \
        --run-root "$RUN_ROOT" \
        --depth "$OOD_WALK_DEPTH" \
        --rng-base "$OOD_RANDOM_SEED"

# ---------- stage 2: fit PCA on training plates ----------
run_stage 2 fit_pca "$RUN_ROOT/pca.json" \
    python -m design_search.landscape.pca fit-pca \
        --run-root "$RUN_ROOT"

# ---------- stage 3: LEIA score walk (sharded) ----------
WALK_DONE_SENTINEL="$RUN_ROOT/walk/.leia_scored.flag"
if ! have "$WALK_DONE_SENTINEL"; then
    if run_stage_sharded 3 leia_walk "$WALK_DONE_SENTINEL" "$OOD_N_GPUS" \
        python -m design_search.landscape.score --metric=score \
            --run-root "$RUN_ROOT" --stage walk; then
        # With one shard score.py writes leia_scores.jsonl itself; with several,
        # merge the shard files (atomically, so a crash cannot leave it empty).
        if (( OOD_N_GPUS > 1 )); then
            cat "$RUN_ROOT"/walk/leia_scores_shard*.jsonl > "$RUN_ROOT/walk/leia_scores.jsonl.tmp" \
                && mv "$RUN_ROOT/walk/leia_scores.jsonl.tmp" "$RUN_ROOT/walk/leia_scores.jsonl"
        fi
        : > "$WALK_DONE_SENTINEL"
    fi
fi

# ---------- stage 4: novelty search ----------
run_stage 4 novelty "$RUN_ROOT/novelty/candidates.json" \
    python -m design_search.landscape.novelty \
        --run-root "$RUN_ROOT" \
        --B "$OOD_NOVELTY_B" --K "$OOD_NOVELTY_K" --D "$OOD_NOVELTY_D" \
        --rng-base $((OOD_RANDOM_SEED + 100)) \
        --n-gpus "$OOD_N_GPUS"

# ---------- stage 5: LEIA score novelty (sharded) ----------
NOVELTY_DONE_SENTINEL="$RUN_ROOT/novelty/.leia_scored.flag"
if ! have "$NOVELTY_DONE_SENTINEL"; then
    if run_stage_sharded 5 leia_novelty "$NOVELTY_DONE_SENTINEL" "$OOD_N_GPUS" \
        python -m design_search.landscape.score --metric=score \
            --run-root "$RUN_ROOT" --stage novelty; then
        if (( OOD_N_GPUS > 1 )); then
            cat "$RUN_ROOT"/novelty/leia_scores_shard*.jsonl > "$RUN_ROOT/novelty/leia_scores.jsonl.tmp" \
                && mv "$RUN_ROOT/novelty/leia_scores.jsonl.tmp" "$RUN_ROOT/novelty/leia_scores.jsonl"
        fi
        : > "$NOVELTY_DONE_SENTINEL"
    fi
fi

# ---------- stage 6: FEM walk (RAM watchdog) ----------
WALK_FEM_DONE_SENTINEL="$RUN_ROOT/walk/.fem_done.flag"
if ! have "$WALK_FEM_DONE_SENTINEL"; then
    if (( 6 < FROM_STAGE )); then
        echo "STAGE 6 fem_walk SKIP (--from-stage $FROM_STAGE)"
    else
        echo "STAGE 6 fem_walk START  workers=$OOD_FEM_WORKERS  log=$RUN_ROOT/logs/fem_walk.log"
        if bash "$SCRIPT_DIR/_fem_watchdog.sh" \
                --run-root "$RUN_ROOT" \
                --stage walk \
                --workers "$OOD_FEM_WORKERS" \
                --workers-fallback "$OOD_FEM_WORKERS_FALLBACK" \
                --log "$RUN_ROOT/logs/fem_walk.log"; then
            : > "$WALK_FEM_DONE_SENTINEL"
            echo "STAGE 6 fem_walk DONE"
        else
            echo "STAGE 6 fem_walk FAIL (_fem_watchdog.sh exit: 2 = RAM watchdog tripped twice, 3 = FEM error; see logs/)"
            exit 1
        fi
    fi
else
    echo "STAGE 6 fem_walk SKIP (sentinel exists)"
fi

# ---------- stage 7: FEM novelty (RAM watchdog) ----------
NOVELTY_FEM_DONE_SENTINEL="$RUN_ROOT/novelty/.fem_done.flag"
if ! have "$NOVELTY_FEM_DONE_SENTINEL"; then
    if (( 7 < FROM_STAGE )); then
        echo "STAGE 7 fem_novelty SKIP (--from-stage $FROM_STAGE)"
    else
        echo "STAGE 7 fem_novelty START  workers=$OOD_FEM_WORKERS  log=$RUN_ROOT/logs/fem_novelty.log"
        if bash "$SCRIPT_DIR/_fem_watchdog.sh" \
                --run-root "$RUN_ROOT" \
                --stage novelty \
                --workers "$OOD_FEM_WORKERS" \
                --workers-fallback "$OOD_FEM_WORKERS_FALLBACK" \
                --log "$RUN_ROOT/logs/fem_novelty.log"; then
            : > "$NOVELTY_FEM_DONE_SENTINEL"
            echo "STAGE 7 fem_novelty DONE"
        else
            echo "STAGE 7 fem_novelty FAIL (_fem_watchdog.sh exit: 2 = RAM watchdog tripped twice, 3 = FEM error; see logs/)"
            exit 1
        fi
    fi
else
    echo "STAGE 7 fem_novelty SKIP (sentinel exists)"
fi

# ---------- stage 8: encoder-cycle consistency (sharded) ----------
CONSISTENCY_SENTINEL="$RUN_ROOT/consistency/consistency.jsonl"
if ! have "$CONSISTENCY_SENTINEL"; then
    if run_stage_sharded 8 consistency "$CONSISTENCY_SENTINEL" "$OOD_N_GPUS" \
        python -m design_search.landscape.score --metric=consistency \
            --run-root "$RUN_ROOT" \
            --n-steps 10; then   # fixed: the OOD feature layout assumes 10 consistency steps
        if (( OOD_N_GPUS > 1 )); then
            cat "$RUN_ROOT"/consistency/consistency_shard*.jsonl > "$CONSISTENCY_SENTINEL.tmp" \
                && mv "$CONSISTENCY_SENTINEL.tmp" "$CONSISTENCY_SENTINEL"
        fi
    fi
fi

# ---------- stage 9: encode held-out plates ----------
run_stage 9 heldout_encode "$RUN_ROOT/heldout_latents.json" \
    python -m design_search.landscape.pca encode-heldout \
        --run-root "$RUN_ROOT"

# ---------- stage 10: build OOD dataset ----------
run_stage 10 build_ood "$RUN_ROOT/ood/dataset.npz" \
    python -m design_search.landscape.ood.pipeline build \
        --run-root "$RUN_ROOT" \
        --rng-seed "$OOD_RANDOM_SEED"

# ---------- stage 11: train OOD heads ----------
run_stage 11 train_ood "$RUN_ROOT/ood/ckpts/full/best.pt" \
    python -m design_search.landscape.ood.pipeline train \
        --run-root "$RUN_ROOT" --rng-seed "$OOD_RANDOM_SEED"

# ---------- stage 12: eval OOD heads ----------
run_stage 12 eval_ood "$RUN_ROOT/ood/eval_summary.md" \
    python -m design_search.landscape.ood.pipeline eval \
        --run-root "$RUN_ROOT"

# ---------- stage 13: predict on full set ----------
run_stage 13 predict_full "$RUN_ROOT/ood/predictions_full.jsonl" \
    python -m design_search.landscape.ood.pipeline predict \
        --run-root "$RUN_ROOT"

# ---------- stage 14: held-out characterization ----------
run_stage 14 heldout_char "$RUN_ROOT/heldout/heldout_eval.md" \
    env OMPI_MCA_btl=self,tcp \
    python -m design_search.landscape.heldout \
        --run-root "$RUN_ROOT"

# ---------- stage 15: render 4-panel figure ----------
FIG_SENTINEL="$RUN_ROOT/figures/fig_ood_landscape.pdf"
if (( 15 < FROM_STAGE )); then
    echo "STAGE 15 figure SKIP (--from-stage $FROM_STAGE)"
elif have "$FIG_SENTINEL"; then
    echo "STAGE 15 figure SKIP (sentinel exists)"
else
    echo "STAGE 15 figure START"
    if python -m design_search.landscape.figure pca \
            --run-root "$RUN_ROOT" \
            --metric-b vm_corr_mean \
            --consistency-metric mean \
            --panel-order truth-first \
            --minimal \
            >"$RUN_ROOT/logs/stage15_figure.log" 2>&1; then
        # Visualize writes a long-named file under figures/. Symlink the
        # canonical name so consumers can rely on a stable path.
        cd "$RUN_ROOT/figures"
        for ext in pdf png; do
            src="$(ls landscape_panels_combined_vm_corr_mean_heldout_consistency-mean_panelD_truth-first_minimal.$ext 2>/dev/null || true)"
            if [[ -n "$src" ]]; then
                ln -sf "$src" "fig_ood_landscape.$ext"
            fi
        done
        cd "$REPO_ROOT"
        echo "STAGE 15 figure DONE  -> $FIG_SENTINEL"
    else
        echo "STAGE 15 figure FAIL"
        tail -20 "$RUN_ROOT/logs/stage15_figure.log"
        exit 1
    fi
fi

# ---------- manifest ----------
python -c "
import json, os, time
m = {
    'run_name': '$RUN_NAME',
    'run_root': '$RUN_ROOT',
    'completed_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'tokenizer_ckpt': 'checkpoints/tokenizer/production_v2/best.pt',
    'dynamics_ckpt': 'checkpoints/dynamics/production_v2/best_ar.pt',
    'env': {
        'random_seed': $OOD_RANDOM_SEED,
        'walk_depth': $OOD_WALK_DEPTH,
        'novelty_B': $OOD_NOVELTY_B,
        'novelty_K': $OOD_NOVELTY_K,
        'novelty_D': $OOD_NOVELTY_D,
        'fem_workers': $OOD_FEM_WORKERS,
        'fem_workers_fallback': $OOD_FEM_WORKERS_FALLBACK,
        'fem_ram_danger_pct': $OOD_FEM_RAM_DANGER_PCT,
        'consistency_nsteps': $OOD_CONSISTENCY_NSTEPS,
        'fast_mode': bool($FAST),
    },
}
def cnt(j):
    try:
        with open(j) as f: return sum(1 for _ in f)
    except FileNotFoundError: return None
m['counts'] = {
    'walk_leia_jsonl_lines':    cnt('$RUN_ROOT/walk/leia_scores.jsonl'),
    'walk_fem_jsonl_lines':     cnt('$RUN_ROOT/walk/fem_scores.jsonl'),
    'novelty_leia_jsonl_lines': cnt('$RUN_ROOT/novelty/leia_scores.jsonl'),
    'novelty_fem_jsonl_lines':  cnt('$RUN_ROOT/novelty/fem_scores.jsonl'),
    'consistency_jsonl_lines':  cnt('$RUN_ROOT/consistency/consistency.jsonl'),
    'predictions_full_lines':   cnt('$RUN_ROOT/ood/predictions_full.jsonl'),
}
with open('$RUN_ROOT/manifest.json', 'w') as f:
    json.dump(m, f, indent=2)
print('Wrote', '$RUN_ROOT/manifest.json')
"

echo
echo "================================================================"
echo "  COMPLETE  $(date -Iseconds)"
echo "  Figure: $FIG_SENTINEL"
echo "  Manifest: $RUN_ROOT/manifest.json"
echo "================================================================"
