#!/usr/bin/env bash
# Promote a trained tokenizer run and a trained dynamics run into the served
# `production_v2` slot that eval, tests/, and design_search/ read:
#
#   checkpoints/tokenizer/production_v2 -> <tokenizer-run-dir>   (must contain best.pt)
#   checkpoints/dynamics/production_v2  -> <dynamics-run-dir>    (must contain best_ar.pt)
#
# Usage:
#   bash scripts/promote.sh <tokenizer-run-dir> <dynamics-run-dir>
#
# The slot entries are symlinks; re-run to switch runs. Refuses to touch a slot
# that is a real directory, so pre-existing checkpoints are never clobbered.
#
# Choosing which runs to serve is up to you: within a run the trainers keep
# best.pt (lowest val loss) / best_ar.pt (lowest AR-rollout error); across
# runs, compare their logged metrics. This script does not check that the
# dynamics run was trained against the given tokenizer (that is recorded only
# inside the .pt).

set -euo pipefail

usage() { grep -E '^#( |$)' "$0" | sed 's/^# \?//' >&2; exit 2; }
[[ $# -eq 2 ]] || usage

# Resolve the run dirs relative to the caller's cwd before moving to the repo root.
TOK_RUN="$(cd "$1" 2>/dev/null && pwd -P)" || { echo "ERROR: tokenizer run dir not found: $1" >&2; exit 1; }
DYN_RUN="$(cd "$2" 2>/dev/null && pwd -P)" || { echo "ERROR: dynamics run dir not found: $2" >&2; exit 1; }
cd "$(dirname "$0")/.."

[[ -f "$TOK_RUN/best.pt" ]]    || { echo "ERROR: $TOK_RUN has no best.pt (not a finished tokenizer run?)" >&2; exit 1; }
[[ -f "$DYN_RUN/best_ar.pt" ]] || { echo "ERROR: $DYN_RUN has no best_ar.pt (not a finished dynamics run?)" >&2; exit 1; }

TOK_SLOT=checkpoints/tokenizer/production_v2
DYN_SLOT=checkpoints/dynamics/production_v2

# Check both slots before touching either, so a refusal never leaves one
# model promoted and the other not.
for slot in "$TOK_SLOT" "$DYN_SLOT"; do
    if [[ -e "$slot" && ! -L "$slot" ]]; then
        echo "ERROR: $slot exists and is a real directory, not a symlink. Move it aside first." >&2
        exit 1
    fi
done

link() {  # link <slot> <target-dir>
    local slot="$1" target="$2"
    mkdir -p "$(dirname "$slot")"
    ln -sfn "$target" "$slot"
    echo "  $slot -> $(readlink "$slot")"
}

echo "Promoting into the production_v2 slot:"
link "$TOK_SLOT" "$TOK_RUN"
link "$DYN_SLOT" "$DYN_RUN"
echo "Serving: tokenizer best.pt from $(basename "$TOK_RUN"), dynamics best_ar.pt from $(basename "$DYN_RUN")"
