UV ?= uv
RUN := $(UV) run
SHELL := /usr/bin/env bash

.PHONY: bootstrap fullcheck quickstart sync doctor test verify validate simulate extract detect forecast compare report benchmark api demo-scenarios showcase release-proof contracts openapi sbom baseline-parity docs-drift clean

bootstrap:
	bash scripts/bootstrap_local.sh

fullcheck:
	bash scripts/full_system_check.sh

quickstart: bootstrap fullcheck

ml-bootstrap:
	bash scripts/bootstrap_local.sh

sync:
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) sync --group dev --group security; \
	else \
		echo "uv not found; falling back to local bootstrap"; \
		bash scripts/bootstrap_local.sh; \
	fi

doctor:
	@if command -v $(UV) >/dev/null 2>&1 && [[ -f uv.lock ]]; then \
		$(RUN) python scripts/dev_doctor.py; \
	elif [[ -f .venv/bin/activate ]]; then \
		source .venv/bin/activate && python scripts/dev_doctor.py; \
	else \
		echo "No environment found. Run 'make bootstrap' first." && exit 1; \
	fi

test:
	@if command -v $(UV) >/dev/null 2>&1 && [[ -f uv.lock ]]; then \
		$(RUN) pytest -q; \
	else \
		source .venv/bin/activate && python -m pytest -q; \
	fi

verify:
	@if command -v $(UV) >/dev/null 2>&1 && [[ -f uv.lock ]]; then \
		$(RUN) python scripts/dev_doctor.py; \
		$(RUN) lint-imports; \
		$(RUN) python scripts/export_openapi.py; \
		$(RUN) python scripts/check_openapi_contract.py; \
		$(RUN) python scripts/verify_matrix.py; \
	else \
		source .venv/bin/activate && python scripts/dev_doctor.py; \
		source .venv/bin/activate && lint-imports; \
		source .venv/bin/activate && python scripts/export_openapi.py; \
		source .venv/bin/activate && python scripts/check_openapi_contract.py; \
		source .venv/bin/activate && python scripts/verify_matrix.py; \
	fi

validate:
	@source .venv/bin/activate && python validation/run_validation_experiments.py && python validation/neurochem_controls.py

simulate:
	@source .venv/bin/activate && mfn simulate --grid-size 24 --steps 16 --with-history --include-arrays --output artifacts/local/simulate.json

extract:
	@source .venv/bin/activate && mfn extract --grid-size 24 --steps 16 --output artifacts/local/extract.json

detect:
	@source .venv/bin/activate && mfn detect --grid-size 24 --steps 16 --output artifacts/local/detect.json

forecast:
	@source .venv/bin/activate && mfn forecast --grid-size 24 --steps 16 --horizon 4 --output artifacts/local/forecast.json

compare:
	@source .venv/bin/activate && mfn compare --grid-size 24 --steps 16 --output artifacts/local/compare.json

report:
	@source .venv/bin/activate && mfn report --grid-size 24 --steps 16 --horizon 4 --output artifacts/local/report_manifest.json --output-root artifacts/runs

benchmark:
	@source .venv/bin/activate && python benchmarks/benchmark_core.py && python benchmarks/benchmark_scalability.py && python benchmarks/benchmark_quality.py

api:
	@source .venv/bin/activate && mfn api --host 127.0.0.1 --port 8000

demo-scenarios:
	@source .venv/bin/activate && python scripts/release_prep.py

showcase:
	@source .venv/bin/activate && python scripts/showcase_run.py && python scripts/criticality_sweep.py

contracts:
	@source .venv/bin/activate && lint-imports && python scripts/export_openapi.py && python scripts/check_openapi_contract.py && python -m pytest -q tests/test_cli_workflows.py tests/test_public_api_structure.py tests/test_integration_api_cli.py

openapi:
	@source .venv/bin/activate && python scripts/export_openapi.py && python scripts/check_openapi_contract.py

sbom:
	@source .venv/bin/activate && python scripts/generate_sbom.py && python scripts/attest_artifacts.py

release-proof:
	@source .venv/bin/activate && python scripts/release_proof.py

baseline-parity:
	@source .venv/bin/activate && python scripts/showcase_run.py && python scripts/baseline_parity.py

docs-drift:
	@source .venv/bin/activate && python scripts/docs_drift_check.py

clean:
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info artifacts/bootstrap artifacts/runs/fullcheck_* pytest-report.html .coverage coverage.xml junit.xml
