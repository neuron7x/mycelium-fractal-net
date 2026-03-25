# Release Notes — v4.1.0

**Release date:** 2026-03-22
**Codename:** Morphology-aware Field Intelligence Engine

## Highlights

- **Causal Validation Gate** — 44 falsifiable rules verify that every pipeline conclusion follows from data. Perturbation stability tested with 3 noise seeds. Reports are blocked if causal consistency fails.

- **Typed Analytics** — 30 frozen dataclasses replace untyped dictionaries throughout the type system. All public APIs have complete type signatures.

- **Hardened CI/CD** — 5 GitHub Actions workflows (ci with 8 jobs, release, security, benchmarks, ci-reusable) with Python 3.10–3.13 matrix testing, coverage gating (82%+), security scanning (bandit + pip-audit + gitleaks), and import boundary verification.

- **Config Governance** — All 87 decision thresholds externalized to `configs/detection_thresholds_v1.json` with schema validation. Zero magic numbers in decision paths. Every causal verdict carries a `provenance_hash` for traceability.

- **Neuromodulation** — GABA-A tonic inhibition, serotonergic plasticity, and MWC allosteric model with occupancy conservation. Six canonical profiles. Backward-compatible opt-in.

## What's New

### For Users
- Fluent API: `seq.detect()`, `seq.extract()`, `seq.forecast()`, `seq.compare()`
- Pretty CLI with colored output and `--json` flag
- 6 neuromodulation profiles accessible from CLI, API, and SDK
- Full pipeline report with Ed25519-signed artifact bundles

### For Developers
- 24-category Ruff linting (up from 3)
- 16 pre-commit hooks (up from 6)
- `make fullcheck` — one command for all quality gates
- `CONTRIBUTING.md` with development workflow and code standards
- `SECURITY.md` with vulnerability disclosure policy

### For Researchers
- 41 causal rules with scientific references (Hodgkin-Huxley 1952, IEEE 754-2019)
- Perturbation stability verification (label drift under epsilon=1e-6)
- Deterministic fingerprinting for reproducibility
- Scientific validation experiments in `validation/`

## Upgrade Guide

**From v4.0.0:** No breaking changes. All v4.0.0 code works unmodified. New features are opt-in.

```python
# v4.0.0 code continues to work
seq = mfn.simulate(mfn.SimulationSpec(grid_size=64, steps=32, seed=42))

# v4.1.0 additions (optional)
seq.detect()   # Fluent API
seq.forecast(4)
```

## Full Changelog

See [CHANGELOG.md](CHANGELOG.md) for the complete list of changes.
