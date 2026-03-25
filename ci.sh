#!/bin/bash
# ci.sh — unified local verification pipeline
# Usage: bash ci.sh
# Exit code 0 = all green. Non-zero = failure.
set -e

RUN="${RUN:-uv run python}"
echo "============================================================"
echo "MFN LOCAL VERIFY PIPELINE"
echo "============================================================"
echo ""

echo "[1/6] Lint"
$RUN -m ruff check src/mycelium_fractal_net/bio/ src/mycelium_fractal_net/analytics/ src/mycelium_fractal_net/core/unified_engine.py src/mycelium_fractal_net/core/diagnostic_memory.py
echo "  OK"

echo "[2/6] Types (strict)"
$RUN -m mypy src/mycelium_fractal_net/bio/ --strict --ignore-missing-imports
echo "  OK"

echo "[3/6] Core tests"
$RUN -m pytest tests/test_bio_extension.py tests/test_bio_meta.py \
    tests/test_bio_regression.py tests/test_levin_morphospace.py \
    tests/test_levin_memory.py tests/test_levin_persuasion.py \
    tests/test_levin_pipeline.py tests/test_levin_depth.py \
    tests/test_compute_reserve.py tests/test_bio_coverage_boost.py \
    tests/test_fractal_arsenal.py tests/test_fractal_dynamics.py \
    tests/test_unified_engine.py tests/test_diagnostic_memory.py \
    tests/test_math_frontier.py tests/test_metacognition.py \
    tests/test_input_guards.py tests/test_core_coverage_final.py \
    tests/test_frontier_coverage.py \
    tests/benchmarks/test_bio_gates.py \
    -q --timeout=120
echo "  OK"

echo "[4/6] Canonical reproduce"
$RUN experiments/reproduce.py
echo "  OK"

echo "[5/6] Adversarial validation"
$RUN experiments/adversarial.py
echo "  OK"

echo "[6/6] Import contracts"
uv run lint-imports
echo "  OK"

echo ""
echo "============================================================"
echo "ALL 6 VERIFY GATES PASSED"
echo "============================================================"
