#!/usr/bin/env bash
# Activate a conda env from a non-interactive script, using the `conda` on
# PATH rather than a hardcoded install location.
#
# Usage (from a script that has already cd'd to the repo root):
#   source tests/_conda_activate.sh leia
#
# No-op when that env is already active. Fails with a clear message when it
# is not and `conda` is not on PATH (the caller's `set -e` then stops it).

_leia_conda_activate() {
    local env="$1"
    if [[ "${CONDA_DEFAULT_ENV:-}" == "$env" ]]; then
        return 0
    fi
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: conda env '$env' is not active and 'conda' is not on PATH." >&2
        echo "       Run 'conda activate $env' first (see README.md § Install)." >&2
        return 1
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$env"
}

_leia_conda_activate "${1:?usage: source tests/_conda_activate.sh <env>}"
