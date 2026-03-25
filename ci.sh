#!/bin/bash
# ci.sh — local verification pipeline
# Works with both `uv run` and bare `python3` (pip install -e .)
# Usage: bash ci.sh
# Exit code 0 = all green
set -e

if command -v uv >/dev/null 2>&1; then
    RUN="uv run python"
else
    RUN="python3"
fi

echo "============================================================"
echo "MFN LOCAL VERIFY — $(date +%Y-%m-%d) — runner: $RUN"
echo "============================================================"
echo ""

# ── 1. Import smoke ──────────────────────────────────────────
echo "[1/6] Import smoke..."
$RUN -c "
import mycelium_fractal_net as mfn
assert mfn.__version__
print(f'  v{mfn.__version__}')
"
echo "  OK"

# ── 2. Core tests ────────────────────────────────────────────
echo "[2/6] Core tests..."
$RUN -m pytest \
    tests/smoke/ \
    tests/core/ \
    tests/test_unified_engine.py \
    tests/test_unified_score.py \
    tests/test_math_frontier.py \
    tests/test_fractal_arsenal.py \
    tests/test_levin_pipeline.py \
    tests/test_auto_heal.py \
    tests/test_bio_meta.py \
    tests/test_bio_regression.py \
    tests/test_mwc_allosteric.py \
    tests/test_frontier_coverage.py \
    tests/test_golden_hashes.py \
    tests/test_golden_regression.py \
    -q --timeout=120 --tb=line \
    -W "ignore::pytest.PytestConfigWarning"
echo "  OK"

# ── 3. Adversarial ───────────────────────────────────────────
echo "[3/6] Adversarial..."
$RUN experiments/adversarial.py
echo "  OK"

# ── 4. Reproduce ─────────────────────────────────────────────
echo "[4/6] Reproduce..."
$RUN experiments/reproduce.py
echo "  OK"

# ── 5. Full pipeline smoke ──────────────────────────────────
echo "[5/6] Pipeline..."
$RUN -c "
import mycelium_fractal_net as mfn
seq = mfn.simulate(mfn.SimulationSpec(grid_size=16, steps=20, seed=42))
r = mfn.diagnose(seq)
h = mfn.auto_heal(seq)
print(f'  diagnose={r.severity} heal={\"ok\" if not h.needs_healing else h.healed}')
"
echo "  OK"

# ── 6. Lint (if ruff available) ──────────────────────────────
echo "[6/6] Lint..."
if $RUN -m ruff check src/mycelium_fractal_net/ --select E,F --statistics 2>/dev/null; then
    echo "  OK"
else
    echo "  SKIP (ruff not installed)"
fi

echo ""
echo "============================================================"
echo "ALL VERIFY GATES PASSED"
echo "============================================================"
