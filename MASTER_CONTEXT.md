# MFN + GNC+ Platform — Master Context

**Read this first.** Everything you need to understand the system in 2 minutes.

## What is this

MyceliumFractalNet (MFN) is a computational physics engine that simulates reaction-diffusion fields, analyzes them through topology/geometry/causality, and heals them when they break. GNC+ is its neuromodulatory control layer — 7 axes that modulate how the system thinks.

No other open-source package combines R-D simulation + TDA + causal validation + self-healing.

## Architecture

```
                    HomeostasisLoop (self-regulation)
                         │
                    SovereignGate (6-lens verification)
                         │
                    ThermodynamicKernel (F[u] + λ₁ gate)
                         │
                    ReactionDiffusionEngine
                         │
            ┌────────────┼────────────┐
         detect()    extract()    bio.step()
         46 rules    57-dim       8 mechanisms
            │            │            │
            └────────────┼────────────┘
                         │
            ┌────────────┼────────────┐
         observe()   diagnose()   auto_heal()
            │            │            │
            └────────────┼────────────┘
                         │
                    GNC+ Bridge
              7 modulators × 9 theta
```

## GNC+ — General Neuromodulatory Control

**7 axes** (what the system feels):

| Axis | Modulator | Role | Sigma effect |
|------|-----------|------|-------------|
| Glu | Glutamate | Plasticity | alpha+, rho+, tau- |
| GABA | GABA | Stability | alpha-, beta+, tau+, sigma_E-, sigma_U- |
| NA | Noradrenaline | Salience | beta-, sigma_E+, sigma_U+ |
| 5HT | Serotonin | Restraint | beta+, tau+, nu-, lambda_pe- |
| DA | Dopamine | Reward | alpha+, beta-, tau-, nu+, lambda_pe+ |
| ACh | Acetylcholine | Precision | rho+, beta+, sigma_E-, sigma_U- |
| Op | Opioid | Resilience | beta+, eta+ |

**9 theta parameters** (how the system thinks):
alpha(learning), rho(precision), beta(stability), tau(threshold),
nu(reward), sigma_E(expected uncertainty), sigma_U(unexpected uncertainty),
lambda_pe(prediction-error valence), eta(effort persistence)

**Omega** — 7×7 causal interaction matrix (adaptive, Hebbian update):
Glu↔GABA(-0.6), DA↔5HT(-0.4), NA↔ACh(+0.3), Op→ALL(+0.2)

**MesoController** — switches between micro(theta)/meso(balance)/macro(reset)

## Key MFN Metrics

| Metric | Value | What it means |
|--------|-------|---------------|
| Λ₂ | 1.926 (CV=1%) | H ∝ W₂^0.59 · I^0.86 power law |
| Λ₅ | 0.046 (CV=0.4%) | Integral HWI ratio — path invariant |
| Λ₆ | 1.323 (CV=0.9%) | Entropy decays 32.3% faster than W₂√I |
| γ_organoid | +1.487 ± 0.208 | First measurement on real brain organoids |
| D_box | 1.96 | Box-counting fractal dimension |

## Program Spine: A_H → B_X → D_T → T

```
A_H  Human Intervention Atlas    — maps interventions to ΔΘ
B_X  Cross-Species Causal Bridge — projects human/animal into shared space
D_T  Longitudinal Digital Twin   — predicts manifold trajectory x_{t+1:t+h}
T    Transfer to adaptive agents — AI deployment target
```

## Falsification Conditions (F1-F7)

| # | Condition | What kills the theory |
|---|-----------|----------------------|
| F1 | No stable mapping m_i → ΔΘ | Sign structure doesn't hold |
| F2 | R(m_i) not reproducible | Effects don't replicate |
| F3 | Omega has no explanatory power | Independent axes sufficient |
| F4 | Ψ has no predictive power | Manifold doesn't predict x_{t+1} |
| F5 | Θ not recoverable from Y | Parameters unidentifiable |
| F6 | B_X fails cross-species | No conservation across species |
| F7 | T = 0 | Transfer to AI doesn't work |

## Quick Start (3 commands)

```bash
pip install -e ".[bio,science]"

python -c "
import mycelium_fractal_net as mfn
seq = mfn.simulate(mfn.SimulationSpec(grid_size=32, steps=60, seed=42))
print(mfn.observe(seq))
print(mfn.SovereignGate().verify(seq))
"

python -c "
from mycelium_fractal_net.neurochem.gnc import compute_gnc_state, gnc_diagnose
state = compute_gnc_state({'Dopamine': 0.8, 'Serotonin': 0.3})
print(gnc_diagnose(state).summary())
"
```

## Status

- **v4.5.0** — 2,428+ tests, 82.6% coverage, ruff clean
- **56 GNC+ tests** — Sigma, Omega, F1-F7, MesoController, Bridge
- **10 γ-correlation tests** — synthetic 20-state validation
- **30 commits** this session
- Solo-authored in wartime Ukraine, 2024–2026

---

*Yaroslav Vasylenko / neuron7x*
