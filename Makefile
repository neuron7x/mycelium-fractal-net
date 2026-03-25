UV ?= uv
RUN := $(UV) run
SHELL := /usr/bin/env bash

.PHONY: bootstrap fullcheck quickstart sync doctor test lint typecheck security coverage verify validate simulate extract detect forecast compare report benchmark api demo-scenarios showcase release-proof contracts openapi sbom baseline-parity docs-drift clean selfcheck

# ═══════════════════════════════════════════════════════════════
#  Setup
# ═══════════════════════════════════════════════════════════════

bootstrap:
	bash scripts/bootstrap_local.sh

sync:
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) sync --group dev --group security; \
	else \
		echo "uv not found; falling back to local bootstrap"; \
		bash scripts/bootstrap_local.sh; \
	fi

doctor:
	$(RUN) python scripts/dev_doctor.py

quickstart: bootstrap fullcheck

# ═══════════════════════════════════════════════════════════════
#  Quality gates
# ═══════════════════════════════════════════════════════════════

lint:
	$(RUN) ruff check src/ tests/ scripts/ benchmarks/
	$(RUN) ruff format --check src/ tests/ scripts/ benchmarks/

typecheck:
	$(RUN) mypy src/mycelium_fractal_net/types/ src/mycelium_fractal_net/security/ src/mycelium_fractal_net/core/ src/mycelium_fractal_net/analytics/ src/mycelium_fractal_net/neurochem/ src/mycelium_fractal_net/bio/ --strict --ignore-missing-imports

security:
	$(RUN) bandit -r src/mycelium_fractal_net/ -c pyproject.toml -q
	$(RUN) pip-audit --strict --desc

test:
	$(RUN) pytest -q

coverage:
	$(RUN) pytest --cov=mycelium_fractal_net --cov-branch --cov-report=term-missing --cov-fail-under=80 -q

# ═══════════════════════════════════════════════════════════════
#  Verification & validation
# ═══════════════════════════════════════════════════════════════

reproduce:
	$(RUN) python experiments/reproduce.py

adversarial:
	$(RUN) python experiments/adversarial.py

localci:
	bash ci.sh

verify: lint typecheck reproduce adversarial
	$(RUN) lint-imports

validate:
	$(RUN) python validation/run_validation_experiments.py
	$(RUN) python validation/neurochem_controls.py

fullcheck: lint typecheck test verify security
	@echo "All quality gates passed."

# ═══════════════════════════════════════════════════════════════
#  Pipeline commands
# ═══════════════════════════════════════════════════════════════

simulate:
	$(RUN) mfn simulate --grid-size 24 --steps 16 --with-history --include-arrays --output artifacts/local/simulate.json

extract:
	$(RUN) mfn extract --grid-size 24 --steps 16 --output artifacts/local/extract.json

detect:
	$(RUN) mfn detect --grid-size 24 --steps 16 --output artifacts/local/detect.json

forecast:
	$(RUN) mfn forecast --grid-size 24 --steps 16 --horizon 4 --output artifacts/local/forecast.json

compare:
	$(RUN) mfn compare --grid-size 24 --steps 16 --output artifacts/local/compare.json

report:
	$(RUN) mfn report --grid-size 24 --steps 16 --horizon 4 --output artifacts/local/report_manifest.json --output-root artifacts/runs

# ═══════════════════════════════════════════════════════════════
#  Benchmarks & showcase
# ═══════════════════════════════════════════════════════════════

benchmark:
	$(RUN) python benchmarks/benchmark_core.py
	$(RUN) python benchmarks/benchmark_scalability.py
	$(RUN) python benchmarks/benchmark_quality.py
	$(RUN) python benchmarks/benchmark_cognitive.py

api:
	$(RUN) mfn api --host 127.0.0.1 --port 8000

demo-scenarios:
	$(RUN) python scripts/release_prep.py

showcase:
	$(RUN) python scripts/showcase_run.py
	$(RUN) python scripts/criticality_sweep.py

# ═══════════════════════════════════════════════════════════════
#  Contracts & release
# ═══════════════════════════════════════════════════════════════

contracts:
	$(RUN) lint-imports
	$(RUN) python scripts/export_openapi.py
	$(RUN) python scripts/check_openapi_contract.py
	$(RUN) pytest -q tests/test_cli_workflows.py tests/test_public_api_structure.py tests/test_integration_api_cli.py

openapi:
	$(RUN) python scripts/export_openapi.py
	$(RUN) python scripts/check_openapi_contract.py

sbom:
	$(RUN) python scripts/generate_sbom.py
	$(RUN) python scripts/attest_artifacts.py

release-proof:
	$(RUN) python scripts/release_proof.py

baseline-parity:
	$(RUN) python scripts/showcase_run.py
	$(RUN) python scripts/baseline_parity.py

docs-drift:
	$(RUN) python scripts/docs_drift_check.py

# ═══════════════════════════════════════════════════════════════
#  Cleanup
# ═══════════════════════════════════════════════════════════════

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .import_linter_cache build dist *.egg-info artifacts/bootstrap artifacts/runs/fullcheck_* pytest-report.html .coverage coverage.xml junit.xml __pycache__

# ═══════════════════════════════════════════════════════════════
#  Self-Check — single command to validate everything
# ═══════════════════════════════════════════════════════════════
selfcheck:
	$(RUN) python scripts/selfcheck.py

selfcheck-quick:
	$(RUN) python scripts/selfcheck.py --quick
