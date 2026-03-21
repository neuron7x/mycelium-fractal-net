# Morphology-aware Field Intelligence Engine

Deterministic structural analytics engine for 2D field simulation, morphology-aware feature extraction, anomaly/regime intelligence, short-horizon forecasting, morphology comparison, and canonical report generation.

## Quickstart (2 commands, full local visibility)
```bash
bash scripts/bootstrap_local.sh
bash scripts/full_system_check.sh
```

This path installs the editable project, dev/security/reporting tooling, and the ML dependency path by default. On Linux x86_64 it prefers the CPU PyTorch wheel to avoid the accidental CUDA-heavy install path.

## What you get after the run
Artifacts land under `artifacts/runs/fullcheck_<timestamp>/` with:
- `full.log` — complete raw stdout/stderr
- `summary.txt` — short tail of the run
- `errors_only.log` — concentrated failure digest
- `status.tsv` — exit code per stage
- `final_readable_report.md` — human-readable pass/fail summary
- `pytest-report.html` — browsable test report
- `junit.xml` — CI-friendly test feed
- `coverage.xml` — machine-readable coverage
- `manifest.json` — run index

## Install controls
```bash
INSTALL_ML=1 bash scripts/bootstrap_local.sh
INSTALL_ML=0 bash scripts/bootstrap_local.sh
INSTALL_ACCEL=1 bash scripts/bootstrap_local.sh
TORCH_CHANNEL=cpu bash scripts/bootstrap_local.sh
```

## Canonical local commands
```bash
make bootstrap
make fullcheck
```

## Simulation API

The core simulation API accepts an optional `neuromodulation=None` parameter on `SimulationSpec` for opt-in neuromodulation contour. When omitted, the baseline reaction-diffusion engine runs without neuromodulation state.

## Release pipeline

The full release pipeline runs:
1. `scripts/showcase_run.py` — generates showcase artifacts under `artifacts/release/`
2. `scripts/baseline_parity.py` — baseline parity verification
3. `scripts/attest_artifacts.py` — Ed25519 artifact attestation

Final output: `artifacts/release/release_manifest.json` with provenance and attestation references.

## Notes
- `uv` remains optional; the local bootstrap path does not depend on `uv.lock` consistency.
- Full verification expects ML surfaces to be present.
- Public API signatures for `compute_nernst_potential` and `simulate_mycelium_field` preserve introspection contracts.
- `SECURITY_CHECKOV` runs through `python -m checkov` so the runner does not depend on a shell-visible binary.
