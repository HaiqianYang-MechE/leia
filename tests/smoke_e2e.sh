#!/usr/bin/env bash
# End-to-end smoke: full leia/ package import + every script's --help works.

set -euo pipefail

cd "$(dirname "$0")/.."

source tests/_conda_activate.sh leia

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo ">>> import leia"
python -c "
import leia
import leia.metrics
import leia.losses, leia.training, leia.distributed
import leia.tokenizer, leia.dynamics, leia.datasets, leia.stress, leia.stress_torch
print(f'leia v{leia.__version__} OK')
print(f'public API: {[x for x in dir(leia) if not x.startswith(\"_\")][:10]}...')
"

echo ""
echo ">>> scripts/*.py --help"
for script in scripts/*.py; do
    if err=$(python "$script" --help 2>&1 >/dev/null); then
        echo "  OK: $script"
    else
        echo "  FAIL: $script"
        echo "$err" | tail -15 | sed 's/^/      /'
        exit 1
    fi
done

echo ""
echo ">>> every design_search module importable"
python - <<'PYEOF'
import importlib, pkgutil
import design_search
n_ok = n_fail = 0
for m in pkgutil.walk_packages(design_search.__path__, "design_search."):
    try:
        importlib.import_module(m.name)
        n_ok += 1
    except Exception as e:
        print(f"  FAIL: {m.name}: {type(e).__name__}: {str(e)[:80]}")
        n_fail += 1
print(f"  {n_ok} OK, {n_fail} fail")
assert n_ok > 0, "no design_search modules found"
assert n_fail == 0, f"{n_fail} design_search modules failed import"
PYEOF

echo ""
echo "PASS smoke_e2e"
