#!/bin/bash
# scripts/check_bio.sh — unified bio quality gate
# Works on clean clone: uv sync --group dev && bash scripts/check_bio.sh
set -e

RUN="${RUN:-.venv/bin/python}"
echo "=== BIO QUALITY GATE ==="
echo "runner: $RUN"
echo ""

echo "1/7 Lint"
$RUN -m ruff check src/mycelium_fractal_net/bio/
$RUN -m ruff format --check src/mycelium_fractal_net/bio/

echo "2/7 Types"
$RUN -m mypy src/mycelium_fractal_net/bio/ --strict --ignore-missing-imports

echo "3/7 Unit tests"
$RUN -m pytest tests/test_bio_extension.py tests/test_bio_meta.py -q --timeout=60

echo "4/7 Regression tests"
$RUN -m pytest tests/test_bio_regression.py -q --timeout=60

echo "5/7 Property tests (fast)"
BIO_HYPOTHESIS_PROFILE=fast $RUN -m pytest tests/test_bio_properties_fast.py -q --timeout=60

echo "6/7 Stateful tests"
BIO_HYPOTHESIS_PROFILE=fast $RUN -m pytest tests/test_bio_stateful.py -q --timeout=60

echo "7/7 Benchmark gates (calibrated)"
$RUN -m pytest tests/benchmarks/test_bio_gates.py -v --timeout=60

echo ""
echo "=== ALL 7 BIO GATES GREEN ==="
