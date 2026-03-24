#!/bin/bash
# scripts/check_bio.sh — unified bio quality gate
# One command: lint + types + regression + property + stateful + benchmarks
set -e

PYTHON="${PYTHON:-.venv/bin/python}"
echo "=== BIO QUALITY GATE ==="
echo ""

echo "1. Lint"
$PYTHON -m ruff check src/mycelium_fractal_net/bio/
$PYTHON -m ruff format --check src/mycelium_fractal_net/bio/

echo "2. Types"
$PYTHON -m mypy src/mycelium_fractal_net/bio/ --strict --ignore-missing-imports

echo "3. Unit tests"
$PYTHON -m pytest tests/test_bio_extension.py tests/test_bio_meta.py -q --timeout=60

echo "4. Regression tests"
$PYTHON -m pytest tests/test_bio_regression.py -q --timeout=60

echo "5. Property tests (fast)"
BIO_HYPOTHESIS_PROFILE=fast $PYTHON -m pytest tests/test_bio_properties_fast.py -q --timeout=60

echo "6. Stateful tests"
BIO_HYPOTHESIS_PROFILE=fast $PYTHON -m pytest tests/test_bio_stateful.py -q --timeout=60

echo "7. Benchmark gates"
$PYTHON -m pytest tests/benchmarks/test_bio_gates.py -v --timeout=60

echo ""
echo "=== ALL BIO GATES GREEN ==="
