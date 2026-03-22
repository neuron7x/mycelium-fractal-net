# Changelog

All notable changes to MyceliumFractalNet are documented here.

## [4.1.0] — 2026-03-22

### Added
- **Causal Validation Gate** — 42 rules verifying cause-effect consistency across 7 pipeline stages
- **Perturbation stability** — automatic label stability check under 1e-6 noise (3 seeds)
- **Typed analytics** — 34 frozen dataclasses replacing untyped dict[str, float]
- **Descriptor cache** — LRU by runtime_hash, 14x→1x computation per pipeline
- **Detection constants** — 62 named thresholds with versioned config
- **Golden regression tests** — 18 deterministic output checks
- **Benchmark performance gates** — 4 tests with baseline + margin
- **Security hardening** — CSP, HSTS, body limits, output sanitization, error scrubbing
- **Fluent API** — `seq.detect()`, `seq.forecast()`, `seq.compare()`, `seq.extract()`
- **Pretty CLI** — colored terminal output, `--json` for machine use
- **NeuromodulationStateSnapshot** — typed state with occupancy conservation law
- **SimulationMetrics** — typed replacement for untyped engine metadata

### Changed
- **RDE split** — reaction_diffusion_engine.py (952 lines) → 3 focused modules
- **Import chain** — torch decoupled from core, CPU-only install works
- **CI** — full pytest suite instead of 4 cherry-picked files
- **Warning policy** — UserWarning=error, DeprecationWarning=ignore
- **Repr** — all core types show one line of meaning, not noise

### Fixed
- manifest.json missing from artifact_list
- SyntaxWarning from unescaped regex in test_federated.py
- Import contracts: 3/4 → 4/4 KEPT (artifact_bundle crypto via importlib)
- Descriptor recomputation: 14x → 1x per pipeline run

## [4.0.0] — 2026-03-17

### Added
- Neuromodulation integration (GABA-A, serotonergic, MWC allosteric model)
- Reaction-diffusion engine with CFL stability, adaptive alpha guard
- Memmap history backend for large grids
- Ed25519 artifact signing
- 6-stage pipeline: simulate → extract → detect → forecast → compare → report
