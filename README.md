<p align="center">
  <img src="docs/project-history/legacy-materials/assets/header.svg" alt="MyceliumFractalNet" width="100%" />
</p>

<h1 align="center">MyceliumFractalNet</h1>
<h3 align="center">Morphology-aware analytics engine with causal validation, adaptive bio physics, and Levin cognitive framework</h3>

<p align="center">
  <img src="https://img.shields.io/badge/verified_tests-224-brightgreen?style=flat-square" alt="Tests" />
  <img src="https://img.shields.io/badge/adversarial-6/6_pass-brightgreen?style=flat-square" alt="Adversarial" />
  <img src="https://img.shields.io/badge/causal_rules-46/46-blue?style=flat-square" alt="Causal Rules" />
  <img src="https://img.shields.io/badge/bio_mechanisms-8-orange?style=flat-square" alt="Bio" />
  <img src="https://img.shields.io/badge/import_contracts-8/8-blue?style=flat-square" alt="Contracts" />
  <img src="https://img.shields.io/badge/reproduce-deterministic-brightgreen?style=flat-square" alt="Reproduce" />
  <img src="https://img.shields.io/badge/mypy-strict-blue?style=flat-square" alt="Types" />
  <img src="https://img.shields.io/badge/Python-%E2%89%A53.10-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square" alt="License" /></a>
</p>

<br/>

> **MFN** simulates biological pattern formation on reaction-diffusion lattices, extracts morphological features, classifies regimes, forecasts evolution, and **proves that every conclusion is causally consistent** through 46 executable rules with DOI references. The bio layer implements 8 peer-reviewed mechanisms from Physarum transport to Levin's morphogenetic cognition.

<br/>

## Quickstart

```python
import mycelium_fractal_net as mfn

# Simulate a 32x32 reaction-diffusion field
seq = mfn.simulate(mfn.SimulationSpec(grid_size=32, steps=60, seed=42))

# One-call diagnosis: detect + EWS + forecast + causal gate + intervention plan
report = mfn.diagnose(seq)
print(report.summary())
# [DIAGNOSIS:INFO] anomaly=nominal(0.22) ews=approaching_transition(0.46) causal=pass

# Bio layer: Physarum + anastomosis + FHN + chemotaxis + dispersal
from mycelium_fractal_net.bio import BioExtension
bio = BioExtension.from_sequence(seq).step(n=5)
print(bio.report().summary())
# [BIO step=5] physarum: D_max=1.000 flux=0.007 | fhn: spiking=0.000 (0ms)

# Levin cognitive framework: morphospace + memory anonymization + persuasion
from mycelium_fractal_net.bio import LevinPipeline
report = LevinPipeline.from_sequence(seq).run()
print(report.summary())
# [LEVIN] pc1=0.944 S_B=0.96±0.03 | anon=0.064 fiedler=0.0025 | persuade=0.521 modes=10
```

---

## What Makes This Different

Most scientific Python packages simulate *or* analyze *or* validate. MFN does all three in a single pipeline with mathematical proof at every stage.

| Layer | What it does | How it's verified |
|-------|-------------|-------------------|
| **Simulation** | Reaction-diffusion PDE on N×N lattice | CFL stability, field bounds, NaN guard |
| **Analysis** | 57-dim morphology embedding, regime classification, EWS | 46 causal rules block invalid conclusions |
| **Bio Physics** | 8 mechanisms: Physarum, FHN, chemotaxis, anastomosis, dispersal, morphospace, memory diffusion, persuasion | Property-based tests (Hypothesis), stateful tests, calibrated benchmark gates |
| **Levin Framework** | PCA morphospace + basin stability + HDV memory anonymization + active inference + controllability Gramian | 5 mathematical invariants, 87% branch coverage, stress tested |

---

## Bio Layer: 8 Peer-Reviewed Mechanisms

<table>
<tr><th>Mechanism</th><th>Model</th><th>Reference</th></tr>
<tr><td><b>Physarum transport</b></td><td>Adaptive conductivity + Kirchhoff pressure solver</td><td>Tero et al. (2007) J. Theor. Biol.</td></tr>
<tr><td><b>Hyphal anastomosis</b></td><td>Tip growth + fusion + branching network</td><td>Glass et al. (2004) Microbiol. Mol. Biol. Rev.</td></tr>
<tr><td><b>FitzHugh-Nagumo</b></td><td>Excitable signaling with refractory dynamics</td><td>FitzHugh (1961) Biophys. J.</td></tr>
<tr><td><b>Chemotaxis</b></td><td>Keller-Segel gradient-following</td><td>Keller & Segel (1970) J. Theor. Biol.</td></tr>
<tr><td><b>Spore dispersal</b></td><td>Fat-tailed Levy flight kernel</td><td>Nathan et al. (2012) Ecol. Lett.</td></tr>
<tr><td><b>Morphospace</b></td><td>PCA state space + Monte Carlo basin stability</td><td>Menck et al. (2013) Nature Physics</td></tr>
<tr><td><b>Memory anonymization</b></td><td>Gap junction HDV diffusion (graph heat equation)</td><td>Levin (2023) Cognitive agency</td></tr>
<tr><td><b>Persuasion</b></td><td>Active inference + controllability Gramian</td><td>Friston & Levin (2015) Interface Focus</td></tr>
</table>

### Levin Pipeline

The three Levin modules form a unified cognitive framework for morphogenetic fields:

```
  Morphospace (PCA)          Memory Anonymization         Persuasion
  ┌──────────────────┐       ┌───────────────────────┐    ┌──────────────────────┐
  │ Field history     │       │ Per-cell HDV memory    │    │ Linearized dynamics  │
  │ → PCA projection  │       │ → Laplacian from       │    │ → Controllability    │
  │ → Basin stability │       │   Physarum conductance │    │   Gramian W_c        │
  │ → Attractor map   │       │ → Heat equation        │    │ → Persuadability     │
  │                   │       │   diffusion            │    │   score              │
  │ S_B = n_return/N  │       │ dM/dt = -α·L_g·M      │    │ → Intervention level │
  └──────────────────┘       └───────────────────────┘    └──────────────────────┘
           │                           │                            │
           └───────────────────────────┴────────────────────────────┘
                                       │
                              LevinPipeline.run()
                                       │
                              LevinReport.summary()
                              LevinReport.interpretation()
```

```python
from mycelium_fractal_net.bio import LevinPipeline

pipeline = LevinPipeline.from_sequence(seq)
report = pipeline.run(target_field=target)

print(report.interpretation())
# System is robust — S_B=0.96 means 96% of perturbations return to attractor
# | Memory is still individual (anonymity=0.06) — weak coupling or early stage
# | Persuadable system (10 controllable modes) — SIGNAL interventions effective
```

---

## Causal Validation

Every report includes machine-readable proof. 46 rules across 7 stages:

| Stage | Rules | Verified |
|-------|-------|----------|
| **SIM** | 11 | Field bounds, NaN/Inf, CFL, occupancy conservation, MWC monotonicity |
| **EXT** | 7 | Embedding validity, descriptor completeness, feature-group integrity |
| **DET** | 8 | Score bounds, label vocabulary, confidence range, evidence consistency |
| **FOR** | 7 | Horizon validity, prediction bounds, uncertainty envelope, damping |
| **CMP** | 6 | Distance non-negativity, cosine bounds, label-metric consistency |
| **XST** | 5 | Cross-stage logical coherence (regime/anomaly, neuromod/plasticity) |
| **PTB** | 2 | Label stability under perturbation (3 noise seeds) |

If a conclusion does not follow from data, the system **blocks it**. No exceptions.

---

## Performance

Hardware stress tested (32/32 operations passed, 0 failures, 0 memory leaks):

| Operation | N=16 | N=32 | N=64 |
|-----------|------|------|------|
| Simulation (30 steps) | 7ms | 8ms | 10ms |
| Physarum (10 steps) | 10ms | 32ms | 74ms |
| BioExtension.step(5) | — | 25ms | — |
| Morphospace (PCA fit) | 11ms | 17ms | 20ms |
| LevinPipeline.run() | 38ms | 63ms | — |
| Memory anonymization | 68ms | 141ms | 611ms |

Hot paths are vectorized (numpy stride tricks, sparse matmul), calibrated with gc.disable benchmark harness.

<details>
<summary><b>Benchmark architecture</b></summary>

```
benchmarks/
├── bio_baseline.json          # Calibrated baselines (machine-specific)
├── calibrate_bio.py           # Baseline generator (gc.disable, 200 samples)
├── stress_test.py             # 32-point stress escalation
└── stress_results.json        # Last run results

tests/benchmarks/
└── test_bio_gates.py          # 4 performance regression gates
                               # Adaptive multiplier: 50× sub-ms, 5× ms, 3× >5ms
```

</details>

---

## Engineering Quality

| Metric | Value |
|--------|-------|
| **Verified core tests** | 224 (bio + Levin + fractal + unified + math frontier + benchmarks) |
| **Branch coverage** | 92% bio/ (enforced ≥90% gate) |
| **Mypy** | `--strict` on 14 bio/ files, 0 errors |
| **Ruff** | 0 lint violations |
| **Import contracts** | 8/8 enforced (import-linter) |
| **Causal rules** | 46/46 verified |
| **Stress tests** | 32/32 (grid 8→96, steps 30→1000, memory leak check) |
| **Security** | pip-audit: 0 known vulnerabilities |
| **Property tests** | Hypothesis-based invariant verification |
| **Stateful tests** | RuleBasedStateMachine for BioMemory |
| **Frozen dataclasses** | 30 (immutable type system) |
| **Named constants** | 62 (zero magic numbers in detection) |

---

## Architecture

```
src/mycelium_fractal_net/
├── types/          30 frozen dataclasses — the type system
├── core/           PDE solver, detect, forecast, compare, causal validation, EWS
├── analytics/      Feature extraction, morphology (57-dim), connectivity
├── bio/            8 mechanisms + LevinPipeline + meta-optimizer
│   ├── physarum.py           Adaptive conductivity (Tero 2007)
│   ├── anastomosis.py        Hyphal network (Glass 2004)
│   ├── fhn.py                Excitable signaling (FitzHugh 1961)
│   ├── chemotaxis.py         Keller-Segel (1970)
│   ├── dispersal.py          Fat-tailed spores (Nathan 2012)
│   ├── morphospace.py        PCA + basin stability (Menck 2013)
│   ├── memory_anonymization.py  HDV diffusion (Levin 2023)
│   ├── persuasion.py         Active inference (Friston-Levin 2015)
│   ├── levin_pipeline.py     Unified entry point
│   ├── memory.py             HDV episodic memory (Kanerva 2009)
│   ├── evolution.py          CMA-ES parameter optimization
│   └── meta.py               Memory-augmented MA-CMA-ES
├── neurochem/      GABA-A kinetics, serotonergic plasticity, MWC model
├── security/       Input validation, encryption, audit, hardening
├── integration/    API server, adapters, schemas, authentication
├── pipelines/      Report generation, scenario presets
├── numerics/       Grid operations, Laplacian, CFL stability
└── cli.py          Terminal interface
```

8 import boundary contracts enforced by import-linter. No module exceeds 400 LOC without documented exemption.

---

## Installation

```bash
# Recommended (uv)
uv sync --group dev --extra bio

# Or pip
pip install -e ".[dev,bio]"

# Verify
python -c "import mycelium_fractal_net; print('ok')"
```

| Profile | Command | What you get |
|---------|---------|--------------|
| Core | `pip install -e .` | Simulation, detection, forecasting |
| Bio | `pip install -e ".[bio]"` | + Physarum, Levin, meta-optimizer |
| Dev | `pip install -e ".[dev,bio]"` | + pytest, ruff, mypy, hypothesis |
| Full | `pip install -e ".[full]"` | Everything (ML, API, security) |

**Requirements:** Python 3.10 — 3.13

---

## Verification

```bash
# One command — full local CI (lint + types + tests + reproduce + adversarial + contracts)
bash ci.sh

# Individual steps
uv run python experiments/reproduce.py     # Deterministic canonical reproduction
uv run python experiments/adversarial.py   # 6 adversarial invariants across 50+ seeds
make verify                                # Lint + types + reproduce + adversarial + contracts
```

See [docs/verification.md](docs/verification.md) and [docs/reproducibility.md](docs/reproducibility.md).

## Quality Gates

```bash
# Bio-specific gate (10 checks: lint, types, 7 test categories, coverage ≥90%)
bash scripts/check_bio.sh

# Full verification
make fullcheck
```

<details>
<summary><b>What check_bio.sh runs</b></summary>

```
1/7 Lint          ruff check + format
2/7 Types         mypy --strict (14 files)
3/7 Unit tests    bio extension + meta
4/7 Regression    correctness-only (no timing assertions)
5/7 Property      Hypothesis invariants
6/7 Stateful      RuleBasedStateMachine
7/7 Benchmarks    Calibrated gates (baseline × adaptive multiplier)
```

</details>

---

## CLI

```bash
mfn simulate --seed 42 --grid-size 64 --steps 32
mfn detect --seed 42 --grid-size 64 --steps 32
mfn report --seed 42 --grid-size 64 --steps 32 --output-root ./results
```

## REST API

```bash
mfn api --host 0.0.0.0 --port 8000
```

Endpoints: `/health` `/metrics` `/v1/simulate` `/v1/extract` `/v1/detect` `/v1/forecast` `/v1/compare` `/v1/report`

Full reference: [API Documentation](docs/API.md)

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | Layer definitions, module boundaries, dependency policies |
| [Public API Contract](docs/PUBLIC_API_CONTRACT.md) | Stable / deprecated / frozen surface classification |
| [Causal Validation](docs/CAUSAL_VALIDATION.md) | 46-rule catalog, failure semantics |
| [Quality Gate](docs/QUALITY_GATE.md) | 17 mandatory gates for PR and release |
| [Benchmarks](docs/BENCHMARKS.md) | Performance methodology and results |
| [Architectural Debt](docs/ARCHITECTURAL_DEBT.md) | Tracked debt with closure conditions |
| [Validation Report](docs/MFN_VALIDATION_REPORT.md) | Scientific validation methodology |
| [Changelog](CHANGELOG.md) | Version history |

---

## License

[MIT](LICENSE) — Yaroslav Vasylenko ([@neuron7x](https://github.com/neuron7x))
