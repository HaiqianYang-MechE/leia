#!/usr/bin/env bash
# Reproduce Figure 4 (surrogate-guided beam search over lattice topologies)
# end-to-end. All artifacts land in design_search/outputs/<run_name>/.
#
# Usage:
#   bash design_search/repro/figure4.sh [run_name]
#
# Smoke test (small config, ~10 min):
#   DESIGN_SEARCH_B=2 DESIGN_SEARCH_K=3 DESIGN_SEARCH_D=2 \
#   DESIGN_SEARCH_N_RANDOM=3 \
#       bash design_search/repro/figure4.sh smoke_001
#
# Full reproduction (production config, ~1.5 h):
#   bash design_search/repro/figure4.sh repro_001
#
# Mesh resolution, applied to all stages: DESIGN_SEARCH_MESH_RES=10 (default;
# the SDF resolution per unit cell, 30 = the training meshes). The paper's
# panel (d) lines come from one run per resolution.
#
# Stages (each idempotent — re-running skips already-completed stages):
#   1. Beam search (B×K×D × 2-direction characterization, GPU)         ~30 min
#   2. FEM validation on iter -1..D-1 best-per-iter (parallel)         ~16 min
#   3. Per-candidate isolated timing bench (serial LEIA + serial FEM)  ~50 min
#   4. Filmstrip rendering (pyvista offscreen via xvfb-run)             ~1 min
#   5. Figure assembly (matplotlib)                                    ~30 s
#
# Stage 2 partial FEM failures are tolerated (some Newton solves can diverge).

set -u -o pipefail
# No `set -e` — stage 2 may have partial FEM failures we want to continue past.

# ---------- arg parsing ----------
RUN_NAME="${1:-repro_001}"

# Search params (overridable via env for smoke testing).
B="${DESIGN_SEARCH_B:-5}"
K="${DESIGN_SEARCH_K:-8}"
D="${DESIGN_SEARCH_D:-10}"
STEPS="${DESIGN_SEARCH_STEPS:-20}"
N_RANDOM="${DESIGN_SEARCH_N_RANDOM:-20}"
RANDOM_SEED="${DESIGN_SEARCH_RANDOM_SEED:-42}"
SEED_PLATE="${DESIGN_SEARCH_SEED:-mg_046}"
MESH_RES="${DESIGN_SEARCH_MESH_RES:-10}"

# Decode is memory-bound at high mesh resolution. Sub-batch size for
# sync_characterize must shrink as N grows. Defaults below are calibrated for
# 80 GB H100; override via DESIGN_SEARCH_BATCH_SIZE if needed.
if [[ -n "${DESIGN_SEARCH_BATCH_SIZE:-}" ]]; then
    BATCH_SIZE="$DESIGN_SEARCH_BATCH_SIZE"
elif [[ "$MESH_RES" -ge 30 ]]; then
    BATCH_SIZE=2
elif [[ "$MESH_RES" -ge 20 ]]; then
    BATCH_SIZE=4
else
    BATCH_SIZE=8
fi

# Anisotropy score formula. 'product' = original stretch/(shear*vf); 'additive'
# = stretch/shear - lambda_vf*vf (bounded, no 1/vf singularity).
SCORE_FORMULA="${DESIGN_SEARCH_SCORE_FORMULA:-product}"
LAMBDA_VF="${DESIGN_SEARCH_LAMBDA_VF:-100}"
MIN_RADIUS="${DESIGN_SEARCH_MIN_RADIUS:-0}"  # 0 = no per-beam radius floor

# ---------- repo + output paths ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

OUT="design_search/outputs/$RUN_NAME"
mkdir -p "$OUT/fem_validation" "$OUT/bench" "$OUT/filmstrip"

# Tee all output to reproduce.log while still showing on stdout
LOG="$OUT/reproduce.log"
exec > >(tee -a "$LOG") 2>&1

# ---------- environment ----------
export PYTHONUNBUFFERED=1
# MPI env vars for the FEM worker pool (single-node TCP transport).
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_pml=ob1

# Activate the leia conda env unless already active (conda from PATH, not a
# hardcoded install location — same as figure5.sh).
if [[ "${CONDA_DEFAULT_ENV:-}" != "leia" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: conda env 'leia' is not active and 'conda' is not on PATH." >&2
        echo "       Run 'conda activate leia' first (see README.md § Install)." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate leia
fi

# ---------- header ----------
echo "================================================================"
echo "  figure4.sh  —  run_name = $RUN_NAME"
echo "  $(date -Iseconds)"
echo "================================================================"
echo "  search params: seed=$SEED_PLATE B=$B K=$K D=$D steps=$STEPS"
echo "  bench params:  n_random=$N_RANDOM random_seed=$RANDOM_SEED reuse_fem=${DESIGN_SEARCH_BENCH_REUSE_FEM:-0}"
echo "  mesh_res:      $MESH_RES (batch_size=$BATCH_SIZE for sync_characterize)"
echo "  score:         formula=$SCORE_FORMULA, lambda_vf=$LAMBDA_VF"
echo "  min_radius:    $MIN_RADIUS"
echo "  output:        $OUT"
echo "  log:           $LOG"
echo

# Resolve and log production checkpoints (v2 via design_search/core/config.py).
echo "--- Production model (resolved from design_search/core/config.py) ---"
python -c "
from design_search.core.config import ServingConfig
c = ServingConfig()
print(f'  tokenizer = {c.tokenizer_ckpt}')
print(f'  dynamics  = {c.dynamics_ckpt}')
print(f'  mu = {c.mu}, lmbda = {c.lmbda}')
print(f'  speeds = {c.speeds}')
"
echo

# ---------- stage tracking ----------
declare -A STAGE_STATUS
run_stage() {
    local n="$1"; local name="$2"; local skip_path="$3"; shift 3
    local cmd=("$@")
    echo
    echo "--- Stage $n: $name ---"
    if [[ -e "$skip_path" ]]; then
        echo "[SKIP] output exists: $skip_path"
        STAGE_STATUS[$n]="SKIP"
        return 0
    fi
    echo ">>> ${cmd[*]}"
    if "${cmd[@]}"; then
        echo "[OK] stage $n complete"
        STAGE_STATUS[$n]="OK"
        return 0
    else
        local ec=$?
        echo "[FAIL] stage $n exit code $ec"
        STAGE_STATUS[$n]="FAIL($ec)"
        return $ec
    fi
}

# ---------- Stage 1: beam search ----------
run_stage 1 "Beam search" "$OUT/results.json" \
    python -m design_search.beam.search \
        --seed "$SEED_PLATE" \
        --B "$B" --K "$K" --D "$D" --steps "$STEPS" \
        --random-seed "$RANDOM_SEED" \
        --mesh-res "$MESH_RES" \
        --batch-size "$BATCH_SIZE" \
        --score-formula "$SCORE_FORMULA" \
        --lambda-vf "$LAMBDA_VF" \
        --min-radius "$MIN_RADIUS" \
        --output-dir "$OUT" \
    || echo "[stage 1] failed; downstream stages will be skipped or may fail"

# ---------- Stage 2: FEM validation (panel b) ----------
# Tolerated: even on partial FEM failure we proceed. fem_results.json may have
# zero-score entries that beam/figure.py filters out.
if [[ -f "$OUT/results.json" ]]; then
    run_stage 2 "FEM validation (panel b convergence)" \
        "$OUT/fem_validation/fem_results.json" \
        python -m design_search.beam.fem_validate \
            --beam-results "$OUT/results.json" \
            --output-dir "$OUT/fem_validation" \
            --n-steps "$STEPS" \
            --mesh-res "$MESH_RES" \
        || echo "[stage 2] partial failure tolerated, continuing"
else
    echo "[SKIP] stage 2: no $OUT/results.json"
    STAGE_STATUS[2]="SKIP"
fi

# ---------- Stage 3: per-candidate bench (panel d) ----------
# Runs AFTER stages 1+2 so the GPU is idle and there is no contention. Both
# bars of panel (d) are measured the same way: one candidate at a time, LEIA
# and FEM each run in-process with nothing else running, at MESH_RES (default
# 10). The paper plots this measurement as lines across several resolutions;
# this orchestrator draws one resolution per run (set DESIGN_SEARCH_MESH_RES
# and combine the bench.json summaries for the lines).
# DESIGN_SEARCH_BENCH_REUSE_FEM=1 copies stage 2's parallel-pool times for the
# best-per-iteration plates instead of re-solving them (saves hours at high
# resolution, but those times include pool contention); panel (d) then prints
# a note that part of the FEM times were pooled.
if [[ -f "$OUT/results.json" ]]; then
    BENCH_FEM_FLAGS=()
    if [[ "${DESIGN_SEARCH_BENCH_REUSE_FEM:-0}" == "1" && -f "$OUT/fem_validation/fem_results.json" ]]; then
        BENCH_FEM_FLAGS+=(--fem-from-stage2 "$OUT/fem_validation/fem_results.json")
    fi
    run_stage 3 "Per-candidate timing bench (panel d)" \
        "$OUT/bench/bench.json" \
        python -m design_search.beam.bench \
            --beam-results "$OUT/results.json" \
            --output-dir "$OUT/bench" \
            --n-random "$N_RANDOM" \
            --random-seed "$RANDOM_SEED" \
            --steps "$STEPS" \
            --mesh-res "$MESH_RES" \
            "${BENCH_FEM_FLAGS[@]}" \
        || echo "[stage 3] failed"
else
    echo "[SKIP] stage 3: no $OUT/results.json"
    STAGE_STATUS[3]="SKIP"
fi

# ---------- Stage 4: filmstrip ----------
if [[ -f "$OUT/results.json" ]]; then
    run_stage 4 "Filmstrip rendering" \
        "$OUT/filmstrip/filmstrip_meta.json" \
        xvfb-run -a python -m design_search.beam.figure prepare-filmstrip \
            --beam-results "$OUT/results.json" \
            --output-dir "$OUT/filmstrip" \
        || echo "[stage 4] failed"
else
    echo "[SKIP] stage 4: no $OUT/results.json"
    STAGE_STATUS[4]="SKIP"
fi

# ---------- Stage 5: figure (always runs — cheap and reads everything else) ----------
echo
echo "--- Stage 5: Figure assembly ---"
FIG_CMD=(
    python -m design_search.beam.figure figure
        --beam-results "$OUT/results.json"
        --output "$OUT/figure4.pdf"
        --filmstrip-dir "$OUT/filmstrip"
)
[[ -f "$OUT/fem_validation/fem_results.json" ]] && \
    FIG_CMD+=(--fem-results "$OUT/fem_validation/fem_results.json")
[[ -f "$OUT/bench/bench.json" ]] && \
    FIG_CMD+=(--bench-results "$OUT/bench/bench.json")

if [[ -f "$OUT/results.json" ]]; then
    echo ">>> ${FIG_CMD[*]}"
    if "${FIG_CMD[@]}"; then
        STAGE_STATUS[5]="OK"
        echo "[OK] figure: $OUT/figure4.pdf"
    else
        STAGE_STATUS[5]="FAIL"
        echo "[FAIL] figure assembly"
    fi
else
    echo "[SKIP] stage 5: no $OUT/results.json"
    STAGE_STATUS[5]="SKIP"
fi

# ---------- summary ----------
echo
echo "================================================================"
echo "  Summary  —  $(date -Iseconds)"
echo "================================================================"
for n in 1 2 3 4 5; do
    printf "  Stage %d: %s\n" "$n" "${STAGE_STATUS[$n]:-?}"
done
echo
echo "  Artifacts under: $OUT"
[[ -f "$OUT/figure4.pdf" ]] && echo "  Figure:           $OUT/figure4.pdf"
[[ -f "$OUT/figure4.png" ]] && echo "  Figure (PNG):     $OUT/figure4.png"
echo "================================================================"
