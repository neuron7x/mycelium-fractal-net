# Changelog

All notable changes to MyceliumFractalNet are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.2.0] — 2026-03-24

### Added
- **Bio layer** — 5 peer-reviewed biological mechanisms: Physarum adaptive conductivity (Tero 2010), hyphal anastomosis (Du 2019), FitzHugh-Nagumo excitable signaling (Adamatzky 2023), fat-tailed spore dispersal (Clark 1999), Keller-Segel chemotaxis (Boswell 2003)
- **Memory-Augmented CMA-ES** — HDV episodic memory (Kanerva 2009) + CMA-ES optimizer for BioConfig parameter search
- **mfn.diagnose()** — unified diagnostic API: detect + EWS + forecast + causal + intervention + narrative in one call
- **mfn.early_warning()** — critical transition detection via autocorrelation, variance ratio, skewness
- **mfn.ensemble_diagnose()** — statistically hardened diagnosis across multiple seeds with CI95
- **mfn.inverse_synthesis()** — reverse parameter synthesis via coordinate descent
- **mfn.watch()** — continuous monitoring with callback-driven control
- **DiagnosisDiff** — temporal diff between diagnostic reports with trend analysis
- **diagnose_streaming()** — generator that yields each pipeline step
- **Causal Intervention Planner** — Pareto-optimal intervention search with robustness evaluation
- **Live demo** — `python -m mycelium_fractal_net` with Rich terminal output
- **RESULTS.md** — reproducible benchmark numbers with DOI references
- Property-based tests (Hypothesis), stateful tests (RuleBasedStateMachine)
- Calibrated benchmark gates (relative to bio_baseline.json × 3.0 multiplier)
- 8 import boundary contracts (bio isolation added)

### Changed
- Module decomposition: model.py (1329→13 LOC), api.py (1062→937), config.py (810→318)
- mypy --strict: 87 files, 0 errors (was 20 files)
- Ruff: 1595→36 (97.7% reduction)
- Causal rule coverage: 46/46 (100%)
- 4 golden hash profiles locked (+ balanced_criticality)
- README rewritten with live demo command and bio layer documentation
- Architectural debt register formalized

### Performance
- Physarum solver: 28.9ms → 3.0ms (9.6× via sparse matrix caching + splu)
- Memory query: 1.4ms → 0.07ms (20× via pre-allocated matrix)
- MetaOptimizer eval: ~300ms → 8ms (37×)
- Fitness function: flat → discriminating (additive 5-component formula)

### Fixed
- NaN propagation in params_to_bio_config
- HDV encoder overflow for extreme float inputs
- BioMemory dirty flag unconditional rebuild
- structural_error drift in README/API docs
- 10 stale "4.1.0" version references

## [Unreleased]

### Changed — Cycle 2 Hardening

#### TASK-03: Neurochem Config Typing
- Replaced `TypedDict` configs (`GABAAKineticsConfig`, `SerotonergicKineticsConfig`, `ObservationNoiseConfig`) with frozen dataclasses with explicit defaults.
- Added `NeuromodulationConfig` frozen dataclass replacing `dict[str, Any]` in engine path.
- Removed `dict` from `step_neuromodulation_state()` signature — accepts only typed configs.
- Eliminated all `.get()` calls in neurochem runtime path (14 occurrences).
- Removed `_neuromod_get` and `_neuromod_sub` helper functions from `reaction_diffusion_engine.py`.
- `ReactionDiffusionConfig.neuromodulation` and `SimulationConfig.neuromodulation` now typed as `NeuromodulationConfig | None`.
- Backward compatibility: dict input auto-converts via `NeuromodulationConfig.from_dict()`.
- Added unit tests: valid typed config, missing optional config, invalid type rejection.

#### TASK-01: MWC Finalization
- Removed legacy `affinity_um` parameter from `mwc_fraction()` (was documented as unused).
- Added explicit literature mapping for all MWC parameters (Chang et al. 1996, Gielen & Bhatt 2019, Bhatt et al. 2021).
- Added monotonicity tests across full concentration range (3 test cases).
- Added EC50 comparison test against published data.

#### TASK-02: Constants Finalization
- Named bare tolerance literals: `_NUMERICAL_EPS`, `_NUMERICAL_DIVISOR_GUARD` in stability.py and forecast.py.
- All neurochem constants already named and categorized (biophysical, numerical stability, empirical calibration) in `neurochem/constants.py`.

#### TASK-15: Strict Typing for Core + Analytics + Neurochem
- `mypy --strict` now passes for `neurochem/` (0 errors) and `analytics/` (0 errors).
- `core/` strict typing enabled — only frozen modules (turing, federated, stdp) excluded.
- Added `disallow_untyped_defs`, `warn_return_any`, `no_implicit_optional`, `strict_equality` for core/analytics/neurochem.
- CI gate blocks merge on mypy regression.

#### TASK-17: Core Dependency Minimization
- Moved `fastapi`, `websockets`, `pandas`, `pyarrow`, `prometheus_client`, `httpx` from core dependencies to optional extras.
- New extras: `[api]`, `[data]`, `[metrics]`, `[ws]`.
- Core install requires only: numpy, sympy, pydantic, cryptography.
- Added core smoke test (`test_core_smoke.py`) verifying simulate/extract/detect/forecast work without optional deps.
- Made pandas import lazy in `types/features.py`.

#### CI Improvements
- Added `core-only` CI job that runs tests without optional ML deps.
- mypy strict CI gate now covers `core/`, `analytics/`, `neurochem/` (blocking).

### Changed
- CI pipeline: 5 workflows (ci.yml 8 jobs, release.yml, security.yml, benchmarks.yml, ci-reusable.yml) with Python 3.10–3.13 matrix, coverage gating (80%), security scanning, import contracts, benchmark tracking.
- Ruff lint rules expanded from 3 to 24 categories (bugbear, bandit, simplify, print detection, complexity, and more).
- mypy `ignore_errors` removed from all modules — type errors are now visible and trackable.
- Coverage: 78.93% → 82.21% branch (+62 targeted tests across cli_display, cli_doctor, compat, config, features, grid_ops, insight_architect).
- Pre-commit hooks expanded from 6 to 16 (bandit, import-linter, mypy, check-yaml/toml/json, no-commit-to-branch, debug-statements).
- All `assert` statements in production code replaced with explicit `RuntimeError` / `ValueError` raises.
- `print()` calls in core modules replaced with `logging.getLogger(__name__)` or `sys.stdout.write()`.
- Optional dependency loader narrowed from `except Exception` to `except ImportError`.
- All silent downgrades (`except: pass`) replaced with `logging.warning()`.
- Makefile modernized: all targets use `uv run`, new `lint`, `typecheck`, `security`, `coverage` targets.
- pytest `--strict-markers --strict-config` enforced.
- benchmark_core.py: CPU-first (no torch dependency), ML benchmarks gated behind `_has_torch()`.
- All 87 decision thresholds (detect + compare + forecast) loaded from `configs/detection_thresholds_v1.json` via `detection_config.py`.

### Added
- `detection_config.py` — config-driven threshold loader with schema validation, fallback defaults, `CONFIG_HASH` for provenance.
- Causal gate enhancements: `provenance_hash`, `engine_version`, `mode` field, `strict_release` / `strict_api` modes, replay consistency.
- `SECURITY.md` — vulnerability disclosure policy with response timeline.
- `CONTRIBUTING.md` — development workflow, code standards, PR process.
- `RELEASE_CANDIDATE_CHECKLIST.md` — 10-gate release sign-off matrix.
- `docs/RELEASE_GOVERNANCE.md` — 12-gate release criteria, change classification, performance budgets, reproducibility sheet.
- 5 CI/CD workflows: ci (8 jobs), release, security (weekly), benchmarks, ci-reusable.
- Bandit configuration in `pyproject.toml`.
- E2E release pipeline test (simulate → extract → detect → forecast → compare → causal gate → report).
- Manifest tampering negative tests (SHA256 mismatch, forged hash, missing/extra artifact, verdict tampering).
- Property tests: replay determinism (5 operations), perturbation stability (5 seeds), causal mode semantics.
- Negative tests: NaN/Inf rejection, out-of-bounds causal failure, config schema validation.
- Config governance tests: schema validation, weight sums, required sections, loaded values match file.
- Release governance file existence checks (14 required files).
- New CPU-only benchmarks: pipeline_e2e, causal_gate latency, memory_simulation.

## [4.1.0] — 2026-03-22

### Added
- **Causal Validation Gate** — 44 falsifiable rules verifying cause-effect consistency across 7 pipeline stages (SIM, EXT, DET, FOR, CMP, XST, PTB).
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
