#!/bin/bash
# scripts/check_bio.sh — unified bio quality gate (9 checks)
# Works on clean clone: uv sync --group dev --extra bio && bash scripts/check_bio.sh
set -e

RUN="${RUN:-uv run python}"
echo "=== BIO QUALITY GATE ==="
echo "runner: $RUN"
echo ""

echo "1/9 Lint"
$RUN -m ruff check src/mycelium_fractal_net/bio/
$RUN -m ruff format --check src/mycelium_fractal_net/bio/

echo "2/9 Types (strict)"
$RUN -m mypy src/mycelium_fractal_net/bio/ --strict --ignore-missing-imports

echo "3/9 Unit tests"
$RUN -m pytest tests/test_bio_extension.py tests/test_bio_meta.py -q --timeout=60

echo "4/9 Regression tests"
$RUN -m pytest tests/test_bio_regression.py -q --timeout=60

echo "5/9 Property tests (fast)"
BIO_HYPOTHESIS_PROFILE=fast $RUN -m pytest tests/test_bio_properties_fast.py -q --timeout=60

echo "6/9 Stateful tests"
BIO_HYPOTHESIS_PROFILE=fast $RUN -m pytest tests/test_bio_stateful.py -q --timeout=60

echo "7/9 Levin + depth + reserve tests"
$RUN -m pytest tests/test_levin_morphospace.py tests/test_levin_memory.py \
    tests/test_levin_persuasion.py tests/test_levin_pipeline.py \
    tests/test_levin_depth.py tests/test_compute_reserve.py \
    tests/test_bio_coverage_boost.py -q --timeout=60

echo "8/9 Benchmark gates (calibrated)"
$RUN -m pytest tests/benchmarks/test_bio_gates.py -v --timeout=60

echo "9/9 Branch coverage (bio/ ≥ 90%)"
$RUN -m pytest tests/test_bio_extension.py tests/test_bio_meta.py \
    tests/test_bio_regression.py tests/test_bio_properties_fast.py \
    tests/test_bio_stateful.py tests/test_levin_morphospace.py \
    tests/test_levin_memory.py tests/test_levin_persuasion.py \
    tests/test_levin_pipeline.py tests/test_levin_depth.py \
    tests/test_compute_reserve.py tests/test_bio_coverage_boost.py \
    tests/benchmarks/test_bio_gates.py \
    --cov=mycelium_fractal_net.bio --cov-branch --cov-fail-under=90 \
    -q --timeout=120

echo ""
echo "=== ALL 9 BIO GATES GREEN ==="
