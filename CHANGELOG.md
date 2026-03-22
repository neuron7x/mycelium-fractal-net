# Changelog

All notable changes to MyceliumFractalNet are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- CI pipeline rewritten: 9-job GitHub Actions with Python 3.10–3.13 matrix, coverage gating, security scanning, import contract verification, and benchmark tracking.
- Ruff lint rules expanded from 3 to 24 categories (bugbear, bandit, simplify, print detection, complexity, and more).
- mypy `ignore_errors` removed from all modules — type errors are now visible and trackable.
- Coverage report enforces `fail_under = 80` with branch coverage and precision.
- Pre-commit hooks expanded from 6 to 16 (bandit, import-linter, mypy, check-yaml/toml/json, no-commit-to-branch, debug-statements).
- All `assert` statements in production code replaced with explicit `RuntimeError` / `ValueError` raises.
- `print()` calls in core modules replaced with `logging.getLogger(__name__)` or `sys.stdout.write()`.
- Optional dependency loader narrowed from `except Exception` to `except ImportError`.
- Makefile modernized: all targets use `uv run`, new `lint`, `typecheck`, `security`, `coverage` targets.
- pytest `--strict-markers --strict-config` enforced.

### Added
- `SECURITY.md` — vulnerability disclosure policy with response timeline.
- `CONTRIBUTING.md` — development workflow, code standards, PR process.
- `.github/workflows/ci-reusable.yml` — reusable workflow for showcase, attestation, and scientific controls.
- Bandit configuration in `pyproject.toml`.
- Coverage `exclude_lines` for pragmas, `TYPE_CHECKING`, and overloads.
- Ruff per-file ignores for tests, scripts, CLI, experiments, and benchmarks.

## [4.1.0] — 2026-03-22

### Added
- **Causal Validation Gate** — 41 falsifiable rules verifying cause-effect consistency across 7 pipeline stages (SIM, EXT, DET, FOR, CMP, XST, PTB).
- **Perturbation stability** — automatic label stability check under ε=10⁻⁶ noise with 3 independent seeds.
- **Typed analytics** — 30 frozen dataclasses replacing untyped `dict[str, float]` throughout the type system.
- **Descriptor cache** — LRU by `runtime_hash`, eliminating redundant 14× recomputation per pipeline run.
- **Detection constants** — 62 named thresholds with versioned config (`configs/detection_thresholds_v1.json`).
- **Golden regression tests** — 18 deterministic output checks for simulation, extraction, and detection.
- **Benchmark performance gates** — 4 gated tests with baseline + margin enforcement.
- **Security hardening** — CSP, HSTS, request body limits, output sanitization, error scrubbing on API surface.
- **Fluent API** — `seq.detect()`, `seq.forecast()`, `seq.compare()`, `seq.extract()` chainable from `FieldSequence`.
- **Pretty CLI** — colored terminal output with `--json` flag for machine consumption.
- **NeuromodulationStateSnapshot** — typed snapshot with occupancy conservation law verification.
- **SimulationMetrics** — typed replacement for untyped engine metadata dictionaries.
- **Import boundary contracts** — 7 rules enforced by import-linter in CI and pre-commit.
- **Artifact attestation** — Ed25519 deterministic signing for release manifests and evidence packs.
- **Scenario presets** — synthetic morphology, sensor grid anomaly, regime transition scenarios.

### Changed
- **RDE refactored** — `reaction_diffusion_engine.py` (952 lines) split into 3 focused modules: engine, config, compat.
- **Import chain** — PyTorch decoupled from core; CPU-only install works without `[ml]` extra.
- **CI** — full test suite execution instead of 4 cherry-picked files.
- **Warning policy** — `UserWarning` and `FutureWarning` treated as errors in pytest.
- **Type repr** — all core types show one-line semantic summary instead of raw field dump.

### Fixed
- `manifest.json` missing from artifact list in report pipeline.
- `SyntaxWarning` from unescaped regex in `test_federated.py`.
- Import contracts: 3/4 → 7/7 KEPT (artifact bundle crypto resolved via importlib).
- Descriptor recomputation: 14× → 1× per pipeline run via hash-based LRU cache.

## [4.0.0] — 2026-03-17

### Added
- Neuromodulation integration — GABA-A tonic inhibition, serotonergic plasticity, MWC allosteric model.
- Reaction-diffusion engine with CFL stability analysis and adaptive alpha guard.
- Memmap history backend for large-grid temporal trajectories.
- Ed25519 artifact signing with deterministic seed from `configs/crypto.yaml`.
- 6-stage canonical pipeline: simulate → extract → detect → forecast → compare → report.
- OpenAPI v2 contract with neuromodulation surface coverage.
- Profile registry: 6 canonical neuromodulation profiles.
- Calibration task framework for biophysical parameter validation.
- Scientific validation experiments (`validation/`).
- Benchmark suite with core, scalability, and quality benchmarks.

[Unreleased]: https://github.com/neuron7x/mycelium-fractal-net/compare/v4.1.0...HEAD
[4.1.0]: https://github.com/neuron7x/mycelium-fractal-net/compare/v4.0.0...v4.1.0
[4.0.0]: https://github.com/neuron7x/mycelium-fractal-net/releases/tag/v4.0.0
