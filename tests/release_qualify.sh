#!/usr/bin/env bash
# Run all smoke + preview scripts in sequence. Every script runs even if an
# earlier one fails; failures are listed at the end and make the exit non-zero.
# Run before tagging a release; ~20 min on 1 GPU.
#
# Does NOT run the paper-grade reproductions (README.md training / eval
# recipes, design_search/repro/ figure orchestrators) — those take hours to
# days. Run those separately when reproducing the paper.

set -euo pipefail
cd "$(dirname "$0")/.."

scripts=(
    tests/smoke_e2e.sh
    tests/smoke_tokenizer.sh
    tests/smoke_dynamics.sh
    tests/smoke_consumers.sh
    tests/smoke_eval.sh
    tests/preview_meshing.sh
    tests/preview_dashboard.sh
    tests/preview_fem.sh
    tests/preview_fem_solve.sh
    tests/preview_visco.sh
    tests/preview_visco_solve.sh
    tests/preview_train_small.sh
)

echo "==============================================================="
echo "  release_qualify — running ${#scripts[@]} scripts"
echo "==============================================================="

n_pass=0; n_skip=0; skipped=(); failed=()
for s in "${scripts[@]}"; do
    echo ""
    echo "==============================================================="
    echo "  $s"
    echo "==============================================================="
    t0=$(date +%s)
    set +e; bash "$s"; rc=$?; set -e
    dt=$(($(date +%s) - t0))
    if (( rc == 0 )); then
        printf "  -> %s OK (%dm%ds)\n" "$s" $((dt/60)) $((dt%60)); n_pass=$((n_pass + 1))
    elif (( rc == 77 )); then
        printf "  -> %s SKIPPED (%dm%ds)\n" "$s" $((dt/60)) $((dt%60)); n_skip=$((n_skip + 1)); skipped+=("$s")
    else
        printf "  -> %s FAIL (exit %d, %dm%ds)\n" "$s" "$rc" $((dt/60)) $((dt%60)); failed+=("$s (exit $rc)")
    fi
done

echo ""
echo "==============================================================="
if (( ${#failed[@]} > 0 )); then
    echo "  FAIL — release_qualify (${#failed[@]} failed, ${n_pass} passed, ${n_skip} skipped)"
    for s in "${failed[@]}"; do echo "    failed: $s"; done
    for s in "${skipped[@]}"; do echo "    skipped: $s"; done
    echo "==============================================================="
    exit 1
elif (( n_pass == 0 )); then
    echo "  release_qualify: nothing ran — all ${n_skip} scripts skipped (missing data or checkpoints)"
    echo "==============================================================="
    exit 1
elif (( n_skip == 0 )); then
    echo "  ALL PASS — release_qualify (${n_pass} scripts)"
else
    echo "  PASS with ${n_skip} skipped — release_qualify (${n_pass} passed)"
    for s in "${skipped[@]}"; do echo "    skipped: $s"; done
fi
echo "==============================================================="
echo "Inspect tests/_artifacts/ for preview outputs."
