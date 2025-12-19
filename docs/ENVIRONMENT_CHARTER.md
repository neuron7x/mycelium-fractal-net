# Environment Charter — MyceliumFractalNet

This document defines the standards for dependency management, configuration, and data handling to keep the project deterministic, reproducible, and secure.

## Toolchain & Versions
- Supported Python: **3.10–3.12** (`Dockerfile` targets Python 3.10-slim).
- Direct dependencies are pinned in `pyproject.toml` and `requirements.txt`.
- The authoritative resolved set is `requirements.lock` (consumed by the Docker build).

## Dependency Policy
- **No floating ranges.** All entries use `==`.
- Canonical install order:
  1. `pip install -r requirements.lock` (preferred for CI/builds)
  2. `pip install -e ".[dev]"` (when developing)
- Regenerate the lock after intentional upgrades:
  ```bash
  pip install -e ".[dev]"
  python scripts/generate_lock.py
  ```
- Do not add new dependencies without pinning and re-locking.

## Configuration & Secrets
- Canonical env surface is documented in `configs/.env.example`; use it only as a template.
- **Secrets never live in the repo.** Provide `MFN_API_KEY`/`MFN_API_KEYS` via secret manager or CI secrets. Prod/staging must keep `MFN_API_KEY_REQUIRED=true`.
- CORS and rate limiting defaults: permissive only in `dev`, explicit origins/limits required in `staging`/`prod`.
- Logging defaults to structured JSON outside `dev`; request bodies are excluded unless explicitly enabled.

## Data Handling
- Follow `data/README.md`: keep committed fixtures tiny and representative; large datasets remain external.
- Generated artifacts under `data/scenarios/` stay gitignored; clean with `git clean -fd data/scenarios/` when needed.

## Verification Checklist
- `ruff check .`
- `pytest`
- `python scripts/generate_lock.py` (after dependency changes)
- Confirm Kubernetes/infra configs keep secrets external (see `k8s.yaml`).
