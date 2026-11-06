#!/usr/bin/env bash
# Wipe everything tests/* produced.
#
#   - tests/_artifacts/   preview + scratch outputs
#   - leaked checkpoint dirs from preview_train_small.sh (run-name prefix)

set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf tests/_artifacts/
rm -rf checkpoints/tokenizer/preview_tok_* 2>/dev/null || true
rm -rf checkpoints/dynamics/preview_dyn_* 2>/dev/null || true

echo "tests/ artifacts cleaned."
