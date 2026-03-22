<p align="center">
  <img src="docs/project-history/legacy-materials/assets/header.svg" alt="MFN" width="100%" />
</p>

<h1 align="center">Morphology-aware Field Intelligence Engine</h1>

<p align="center">
  <strong>Deterministic morphology intelligence for reaction-diffusion systems with causal verification.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/tests-1575_passed-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/causal_rules-42-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/coverage-82%25-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-≥3.10-3776ab?style=flat-square&logo=python&logoColor=white" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square" /></a>
</p>

---

## What it does

MFN simulates biological pattern formation, extracts morphology features, detects anomalies, forecasts evolution, compares states — and **proves that every conclusion is correct** before publishing it.

```python
import mycelium_fractal_net as mfn

seq = mfn.simulate(mfn.SimulationSpec(grid_size=64, steps=32, seed=42))

seq.detect()       # → AnomalyEvent(nominal, score=0.184, confidence=0.79, regime=stable)
seq.extract()      # → MorphologyDescriptor(D_box=1.549, instability=0.269, 57 dims)
seq.forecast(4)    # → ForecastResult(h=4, error=0.039)
seq.compare(seq)   # → ComparisonResult(near-identical, d=0.0000, cos=1.000)
```

Every result is verified by **42 causal rules** before it can be published. If a conclusion doesn't follow from the data — the system blocks it.

---

## Who needs this

| Domain | Use case |
|--------|----------|
| **Computational neuroscience** | GABA-A receptor kinetics, serotonergic plasticity, neuromodulation with occupancy conservation laws |
| **Drug discovery** | In-silico screening: does a compound stabilize or destabilize neural network patterns? |
| **Anomaly detection** | Regime classification (stable / transitional / critical / reorganized / pathological) with causal evidence |
| **Scientific computing** | Verified pipelines: every artifact signed, every decision traceable, every threshold versioned |

---

## What makes it different

<table>
<tr><td>🔬</td><td><strong>Not a black box.</strong> Every detection comes with evidence payload, contributing features, and confidence score. You know <em>why</em> the system decided what it decided.</td></tr>
<tr><td>🔗</td><td><strong>Causal Validation Gate.</strong> 42 rules verify cause-effect consistency across 7 pipeline stages. Perturbation stability tested with 3 noise seeds. No other scientific computing library has this.</td></tr>
<tr><td>🔒</td><td><strong>Artifact provenance.</strong> Ed25519 signatures on every report. Sign-then-verify enforcement. Immutable audit trail.</td></tr>
<tr><td>⚡</td><td><strong>Fast.</strong> Full pipeline (simulate → extract → detect → forecast → compare → causal validation) in 77ms for 64×64 grid. 27M cells/sec simulation throughput.</td></tr>
<tr><td>🧬</td><td><strong>Real biophysics.</strong> Nernst equation, Turing morphogenesis, GABA-A MWC allosteric model, serotonergic plasticity — not toy approximations.</td></tr>
</table>

---

## Pipeline

```
SimulationSpec
     │
     ▼
  simulate ──→ FieldSequence(64x64, 32 steps, [-71.0, -3.6] mV)
     │
     ├──→ extract  ──→ MorphologyDescriptor(57 dims, 7 feature groups)
     ├──→ detect   ──→ AnomalyEvent(nominal/watch/anomalous + regime)
     ├──→ forecast ──→ ForecastResult(predicted states + uncertainty)
     ├──→ compare  ──→ ComparisonResult(distance + topology drift)
     │
     ▼
  Causal Validation Gate ──→ 42 rules, pass/degraded/fail
     │
     ▼
  report ──→ 21 artifacts (JSON, SVG, HTML, MD, signatures, causal_validation.json)
```

---

## Install

```bash
git clone https://github.com/neuron7x/mycelium-fractal-net.git
cd mycelium-fractal-net
pip install -e ".[dev]"
```

Or with full extras (ML + acceleration):
```bash
pip install -e ".[full]"
```

---

## CLI

```bash
# Simulate
mfn simulate --seed 42 --grid-size 64 --steps 32

# Detect anomalies
mfn detect --seed 42 --grid-size 64 --steps 32

# Full report with artifacts
mfn report --seed 42 --grid-size 64 --steps 32 --output-root ./results

# Machine-readable output
mfn --json simulate --seed 42
```

<details>
<summary>Example CLI output</summary>

```
Simulation
──────────────
  Grid:     64x64
  Steps:    32
  Seed:     42
  Alpha:    0.18
  Field:    [-71.0, -3.6] mV  mean=-50.2 mV
  History:  no
  Hash:     a1b2c3d4e5f6g7h8

Detection
─────────────
  Anomaly:  nominal  score=0.184  conf=0.79
  Regime:   stable  score=0.650
  Drivers:  dynamic_threshold, instability_index, near_transition_score
```

</details>

---

## API

```bash
mfn api --host 0.0.0.0 --port 8000
```

| Endpoint | What it does |
|----------|-------------|
| `POST /simulate` | Run simulation, return field statistics |
| `POST /nernst` | Compute Nernst ion potential |
| `POST /validate` | Full validation cycle (requires torch) |
| `GET /health` | Health check |
| `GET /metrics` | Prometheus metrics |

Security: API key auth, rate limiting, CORS, CSP headers, request size limits, output sanitization.

---

## Neuromodulation

Optional GABA-A and serotonergic neuromodulation with biophysical kinetics:

```python
spec = mfn.SimulationSpec(
    grid_size=64, steps=32, seed=42,
    neuromodulation=mfn.NeuromodulationSpec(
        profile="gabaa_tonic_muscimol_alpha1beta3",
        enabled=True,
        gabaa_tonic=mfn.GABAATonicSpec(
            agonist_concentration_um=0.5,
            shunt_strength=0.3,
        ),
    ),
)
seq = mfn.simulate(spec)
seq.detect()  # → AnomalyEvent with neuromodulation-aware regime detection
```

Occupancy conservation (resting + active + desensitized = 1.0) is enforced at runtime and verified by the causal gate.

---

## Causal Validation

Every report includes `causal_validation.json` — a machine-readable proof that conclusions follow from data:

```
CausalValidation(pass, 33/33 rules, 0E 0W)
```

42 rules across 7 stages: simulation bounds, extraction consistency, detection causality, forecast plausibility, comparison coherence, cross-stage logic, perturbation stability.

If a rule fails — the report is blocked. No silent errors. No hidden drift.

[Full rule catalog →](docs/CAUSAL_VALIDATION.md)

---

## Engineering

| Metric | Value |
|--------|-------|
| Tests | 1,575 passed |
| Coverage | 82% |
| Typed dataclasses | 34 |
| Import contracts | 4/4 enforced |
| Causal rules | 42 |
| Named constants | 62 (zero magic numbers in detection) |
| Golden regression tests | 18 |
| Benchmark gates | 4 |
| Security headers | 8 types |
| Bandit scan | 0 medium/high issues |

---

## Project structure

```
src/mycelium_fractal_net/
├── core/           # PDE solver, detect, forecast, compare, report, causal validation
├── analytics/      # Feature extraction, morphology, connectivity, temporal
├── neurochem/      # GABA-A kinetics, serotonergic plasticity, MWC model
├── types/          # 34 frozen dataclasses — the type system
├── integration/    # API adapters, schemas, auth, rate limiting
├── security/       # Input validation, encryption, audit, hardening
├── pipelines/      # Report generation, scenario presets
└── cli.py          # Terminal interface with color output
```

---

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — layering, contracts, boundaries
- [Causal Validation](docs/CAUSAL_VALIDATION.md) — 42 rules, failure semantics, release criteria
- [API Reference](docs/API.md) — endpoints, schemas, security
- [ADRs](docs/adr/) — architectural decision records

---

## License

MIT — Yaroslav Vasylenko ([@neuron7x](https://github.com/neuron7x))
