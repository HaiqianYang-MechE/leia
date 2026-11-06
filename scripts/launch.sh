#!/usr/bin/env bash
# Unified experiment launcher for LEIA (PyTorch side).
# Handles conda activation, unbuffered output, torchrun (DDP), xvfb (rendering).
#
# Usage:
#   bash scripts/launch.sh <command> [launcher-flags] [-- script-args...]
#
# Launcher flags (consumed by launch.sh, NOT forwarded to the script):
#   --gpus N            Wrap with torchrun --nproc_per_node=N (ddp commands only)
#   --gpu-ids 0,1,2     Set CUDA_VISIBLE_DEVICES
#   --master-port PORT  torchrun master port (default 29500)
#   --run-name NAME     Required for training commands; forwarded as --run-name NAME
#
# Commands:
#   train-tokenizer              Train PerceiverTokenizer (ddp)
#   train-dynamics               Train AdaLNContinuousDynamicsTransformer (ddp)
#   build-tokenizer-mmap         Build tokenizer training mmap from FEM NPZ + grad_u (cpu)
#   build-dynamics-mmap          Encode prevq tokens via tokenizer → dynamics training mmap (gpu)
#   build-microplate-visco-mmap  Build MicroPlate visco mmap from HDF5 (gpu)
#   precompute-grad-u            Precompute per-tet displacement gradients (cpu, parallel)
#   precompute-fps               Precompute FPS subsampling order (cpu, parallel)
#   precompute-tet-geom          Precompute tet shape function geometry (cpu, parallel)
#   eval                         Run scripts/eval.py with subcommand (gpu)
#                                Subcommands: tokenizer | e2e | gt-fd
#   render-e2e-dashboard         Comparison videos around leia.eval primitives (render)
#
# FEM data generation (separate envs) lives in `data_curation/fem/`. Run those
# scripts directly with the appropriate conda env activated; they are NOT
# routed through this launcher.
#
# Examples:
#   bash scripts/launch.sh train-tokenizer --gpus 8 --run-name v2-tok -- --mmap-dir data/... --wandb
#   bash scripts/launch.sh train-dynamics  --gpus 8 --run-name v2-dyn -- --mmap-dir data/... --wandb
#   bash scripts/launch.sh eval -- tokenizer --checkpoint checkpoints/tokenizer/production_v2/best.pt --mmap-dir data/...

set -euo pipefail

# ---------- command → script / category mapping ----------

resolve_command() {
    case "$1" in
        train-tokenizer)             SCRIPT="scripts/train_tokenizer.py";          CATEGORY="ddp"    ;;
        train-dynamics)              SCRIPT="scripts/train_dynamics.py";           CATEGORY="ddp"    ;;
        build-tokenizer-mmap)        SCRIPT="scripts/build_tokenizer_mmap.py";     CATEGORY="cpu"    ;;
        build-dynamics-mmap)         SCRIPT="scripts/build_dynamics_mmap.py";      CATEGORY="gpu"    ;;
        build-microplate-visco-mmap) SCRIPT="scripts/build_microplate_visco_mmap.py"; CATEGORY="gpu" ;;
        precompute-grad-u)           SCRIPT="scripts/precompute_grad_u.py";        CATEGORY="cpu"    ;;
        precompute-fps)              SCRIPT="scripts/precompute_fps.py";           CATEGORY="cpu"    ;;
        precompute-tet-geom)         SCRIPT="scripts/precompute_tet_geom.py";      CATEGORY="cpu"    ;;
        eval)                        SCRIPT="scripts/eval.py";                     CATEGORY="gpu"    ;;
        render-e2e-dashboard)        SCRIPT="scripts/render_e2e_dashboard.py";     CATEGORY="render" ;;
        *)  return 1 ;;
    esac
}

usage() {
    grep -E '^#( |$)' "$0" | sed 's/^# \?//'
    exit 0
}

# ---------- parse args ----------

if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    usage
fi

COMMAND="$1"; shift

if ! resolve_command "$COMMAND"; then
    echo "ERROR: unknown command '$COMMAND'. Run 'bash scripts/launch.sh --help' for usage." >&2
    exit 1
fi

GPUS=""
GPU_IDS=""
MASTER_PORT="29500"
RUN_NAME=""
SCRIPT_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --)             shift; SCRIPT_ARGS=("$@"); break ;;
        --gpus)         GPUS="$2";        shift 2 ;;
        --gpu-ids)      GPU_IDS="$2";     shift 2 ;;
        --master-port)  MASTER_PORT="$2"; shift 2 ;;
        --run-name)     RUN_NAME="$2";    shift 2 ;;
        *)
            echo "ERROR: unknown launcher flag '$1'. Use '--' to separate script args." >&2
            echo "       e.g. bash scripts/launch.sh $COMMAND -- $1 ..." >&2
            exit 1
            ;;
    esac
done

# ---------- validation ----------

REQUIRES_RUN_NAME=false
case "$COMMAND" in
    train-tokenizer|train-dynamics) REQUIRES_RUN_NAME=true ;;
esac

if $REQUIRES_RUN_NAME && [[ -z "$RUN_NAME" ]]; then
    echo "ERROR: --run-name is required for $COMMAND" >&2
    exit 1
fi

if [[ -n "$GPUS" ]] && [[ "$CATEGORY" != "ddp" ]]; then
    echo "ERROR: --gpus is only valid for DDP commands (train-tokenizer, train-dynamics)" >&2
    exit 1
fi

# ---------- environment setup ----------

# Activate the `leia` conda env unless it is already active. Uses the conda on
# PATH rather than a hardcoded install location.
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

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -n "$GPU_IDS" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi

# cd to repo root
cd "$(dirname "$0")/.."

# Make `import leia` available without pip install
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ---------- build final command ----------

CMD=()

if [[ "$CATEGORY" == "render" ]]; then
    CMD+=(xvfb-run -a)
fi

if [[ -n "$GPUS" ]]; then
    CMD+=(torchrun --nproc_per_node="$GPUS" --master_port="$MASTER_PORT")
else
    CMD+=(python)
fi

CMD+=("$SCRIPT")

if [[ -n "$RUN_NAME" ]]; then
    CMD+=(--run-name "$RUN_NAME")
fi

if [[ ${#SCRIPT_ARGS[@]} -gt 0 ]]; then
    CMD+=("${SCRIPT_ARGS[@]}")
fi

# ---------- execute ----------

echo ">>> $(printf '%q ' "${CMD[@]}")" >&2
if [[ -n "$GPU_IDS" ]]; then
    echo ">>> CUDA_VISIBLE_DEVICES=$GPU_IDS" >&2
fi

exec "${CMD[@]}"
