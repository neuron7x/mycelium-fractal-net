#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Run: bash scripts/bootstrap_local.sh" >&2
  exit 1
fi
source .venv/bin/activate

RUN_ID="$(date +%F_%H-%M-%S)"
OUT="artifacts/runs/fullcheck_${RUN_ID}"
mkdir -p "$OUT" artifacts/local artifacts/runs
FULL_LOG="$OUT/full.log"
SUMMARY="$OUT/summary.txt"
ERRORS="$OUT/errors_only.log"
STATUS="$OUT/status.tsv"
HTML_REPORT="$OUT/pytest-report.html"
JUNIT_XML="$OUT/junit.xml"
COVERAGE_XML="$OUT/coverage.xml"
FINAL_MD="$OUT/final_readable_report.md"

exec > >(tee -a "$FULL_LOG") 2>&1

echo "===== START ${RUN_ID} ====="
echo "ROOT=$ROOT"
echo "OUT=$OUT"

failed=0
run_step() {
  local name="$1"
  shift
  echo
  echo "========== ${name} =========="
  "$@"
  local rc=$?
  echo "========== EXIT ${rc} :: ${name} =========="
  printf '%s\t%s\n' "$name" "$rc" >> "$STATUS"
  [[ $rc -eq 0 ]] || failed=1
  return 0
}

run_step ENV python -V
run_step PIP_CHECK python -m pip check
run_step DOCTOR python scripts/dev_doctor.py
run_step COMPILEALL python -m compileall src scripts tests validation benchmarks
run_step RUFF python -m ruff check .
run_step BLACK python -m black --check .
run_step ISORT python -m isort . --check-only
run_step MYPY python -m mypy src
run_step IMPORT_LINTER lint-imports
run_step OPENAPI_EXPORT python scripts/export_openapi.py
run_step OPENAPI_CONTRACT python scripts/check_openapi_contract.py
run_step VERIFY_MATRIX python scripts/verify_matrix.py
run_step DOCS_DRIFT python scripts/docs_drift_check.py
run_step PYTEST python -m pytest tests -vv -rA --maxfail=0 --tb=short --durations=50 --junitxml="$JUNIT_XML" --cov=src/mycelium_fractal_net --cov=src/analytics --cov-report=term-missing --cov-report=xml:"$COVERAGE_XML" --html="$HTML_REPORT" --self-contained-html
run_step CONTRACT_TESTS python -m pytest -vv tests/test_cli_workflows.py tests/test_public_api_structure.py tests/test_integration_api_cli.py
run_step VALIDATION python validation/run_validation_experiments.py
run_step NEUROCHEM_CONTROLS python validation/neurochem_controls.py
run_step SIMULATE mfn simulate --grid-size 24 --steps 16 --with-history --include-arrays --output "$OUT/simulate.json"
run_step EXTRACT mfn extract --grid-size 24 --steps 16 --output "$OUT/extract.json"
run_step DETECT mfn detect --grid-size 24 --steps 16 --output "$OUT/detect.json"
run_step FORECAST mfn forecast --grid-size 24 --steps 16 --horizon 4 --output "$OUT/forecast.json"
run_step COMPARE mfn compare --grid-size 24 --steps 16 --output "$OUT/compare.json"
run_step REPORT mfn report --grid-size 24 --steps 16 --horizon 4 --output "$OUT/report_manifest.json" --output-root "$OUT/report_runs"
run_step SHOWCASE python scripts/showcase_run.py
run_step CRITICALITY_SWEEP python scripts/criticality_sweep.py
run_step BASELINE_PARITY python scripts/baseline_parity.py
run_step BENCHMARK_CORE python benchmarks/benchmark_core.py
run_step BENCHMARK_SCALABILITY python benchmarks/benchmark_scalability.py
run_step BENCHMARK_QUALITY python benchmarks/benchmark_quality.py
run_step SECURITY_PIP_AUDIT python -m pip_audit
run_step SECURITY_BANDIT python -m bandit -r src scripts validation -x tests,.venv
run_step SECURITY_CHECKOV python -m checkov -d . --quiet
run_step SBOM python scripts/generate_sbom.py
run_step ATTEST python scripts/attest_artifacts.py
run_step RELEASE_PREP python scripts/release_prep.py
run_step RELEASE_PROOF python scripts/release_proof.py

grep -nEA3 -B3 "FAIL|FAILED|ERROR|Traceback|Exception|AssertionError|ModuleNotFoundError|ImportError|RETURN_CODE=[1-9]" "$FULL_LOG" > "$ERRORS" || true
tail -n 250 "$FULL_LOG" > "$SUMMARY" || true

python - <<PY
from pathlib import Path
status_path = Path(r"$STATUS")
rows = []
if status_path.exists():
    for line in status_path.read_text().splitlines():
        if not line.strip():
            continue
        name, code = line.split('\t', 1)
        rows.append((name, int(code)))
pass_rows = [name for name, code in rows if code == 0]
fail_rows = [name for name, code in rows if code != 0]
out = Path(r"$FINAL_MD")
out.write_text(
    "# Full System Check\n\n"
    + f"- Overall: {'PASS' if not fail_rows else 'FAIL'}\n"
    + f"- Passed stages: {len(pass_rows)}\n"
    + f"- Failed stages: {len(fail_rows)}\n\n"
    + "## Failed stages\n"
    + ("\n".join(f"- {name}" for name in fail_rows) if fail_rows else "- none")
    + "\n\n## Passed stages\n"
    + ("\n".join(f"- {name}" for name in pass_rows) if pass_rows else "- none")
    + f"\n\n## Key files\n- full log: {Path(r'$FULL_LOG').name}\n- summary: {Path(r'$SUMMARY').name}\n- errors only: {Path(r'$ERRORS').name}\n- status: {Path(r'$STATUS').name}\n- pytest html: {Path(r'$HTML_REPORT').name}\n- junit: {Path(r'$JUNIT_XML').name}\n- coverage: {Path(r'$COVERAGE_XML').name}\n",
    encoding='utf-8'
)
PY

printf '{\n  "run_id": "%s",\n  "out": "%s",\n  "full_log": "%s",\n  "summary": "%s",\n  "errors_only": "%s",\n  "status_tsv": "%s",\n  "pytest_html": "%s",\n  "junit_xml": "%s",\n  "coverage_xml": "%s",\n  "final_report_md": "%s",\n  "overall": "%s"\n}\n' \
  "$RUN_ID" "$OUT" "$FULL_LOG" "$SUMMARY" "$ERRORS" "$STATUS" "$HTML_REPORT" "$JUNIT_XML" "$COVERAGE_XML" "$FINAL_MD" "$([[ $failed -eq 0 ]] && echo PASS || echo FAIL)" > "$OUT/manifest.json"

echo
echo "===== FINISH $(date) ====="
echo "FULL_LOG=$FULL_LOG"
echo "SUMMARY=$SUMMARY"
echo "ERRORS_ONLY=$ERRORS"
echo "STATUS=$STATUS"
echo "FINAL_REPORT=$FINAL_MD"
echo "PYTEST_HTML=$HTML_REPORT"
echo "JUNIT_XML=$JUNIT_XML"
echo "COVERAGE_XML=$COVERAGE_XML"
if [[ $failed -eq 0 ]]; then
  echo "OVERALL=PASS"
else
  echo "OVERALL=FAIL"
fi
exit $failed
