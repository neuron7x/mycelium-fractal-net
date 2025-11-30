# MFN Code Structure — Architecture-to-Code Mapping

**Document Version**: 1.0  
**Target Version**: MyceliumFractalNet v4.1.0  
**Last Updated**: 2025-11-30

---

## Overview

This document describes the mapping between the conceptual architecture documented in 
[ARCHITECTURE.md](ARCHITECTURE.md) and the actual code structure in the repository.
It serves as a guide for developers to understand where to find specific functionality
and where to add new features.

---

## Layer Model

MyceliumFractalNet follows a clean three-layer architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        External Layer                                │
│   api.py (FastAPI)  │  CLI (mycelium_fractal_net_v4_1.py)           │
│   Docker, k8s       │  Experiments, Analytics                       │
├─────────────────────────────────────────────────────────────────────┤
│                      Integration Layer                               │
│   src/mycelium_fractal_net/integration/                             │
│   - schemas.py (Pydantic models)                                    │
│   - adapters.py (Core ↔ External bridge)                            │
│   - service_context.py (Unified execution context)                  │
├─────────────────────────────────────────────────────────────────────┤
│                          Core Layer                                  │
│   src/mycelium_fractal_net/core/                                    │
│   - nernst.py (Nernst-Planck electrochemistry)                      │
│   - turing.py (Reaction-diffusion morphogenesis)                    │
│   - fractal.py (Fractal dimension, IFS)                             │
│   - stdp.py (Spike-Timing Dependent Plasticity)                     │
│   - federated.py (Byzantine/Krum aggregation)                       │
│   - stability.py (Lyapunov stability analysis)                      │
│   + membrane_engine.py, reaction_diffusion_engine.py, etc.          │
└─────────────────────────────────────────────────────────────────────┘
```

### Layer Boundaries

| From Layer | Can Import From | Cannot Import From |
|------------|-----------------|-------------------|
| Core | numpy, torch, sympy | FastAPI, uvicorn, integration |
| Integration | Core | External entry points |
| External | Integration, Core | — |

---

## Conceptual Module → Code Mapping

### 1. Nernst-Planck Electrochemistry

| Conceptual | Description | Code Path | Public API |
|------------|-------------|-----------|------------|
| Nernst equation | Membrane potential for ions | `core/nernst.py` | `compute_nernst_potential()` |
| Membrane dynamics | ODE integration | `core/membrane_engine.py` | `MembraneEngine` |
| Ion clamping | Numerical stability | `core/nernst.py` | `ION_CLAMP_MIN` |

**Key Functions:**
```python
from mycelium_fractal_net import compute_nernst_potential

E_K = compute_nernst_potential(
    z_valence=1,
    concentration_out_molar=5e-3,
    concentration_in_molar=140e-3,
    temperature_k=310.0
)  # Returns ~-0.089 V (-89 mV)
```

### 2. Turing Morphogenesis

| Conceptual | Description | Code Path | Public API |
|------------|-------------|-----------|------------|
| Activator-inhibitor | Reaction-diffusion PDEs | `core/turing.py` | `simulate_mycelium_field()` |
| Field simulation | 2D lattice dynamics | `core/reaction_diffusion_engine.py` | `ReactionDiffusionEngine` |
| Pattern threshold | Turing activation | `core/turing.py` | `TURING_THRESHOLD` |

**Key Functions:**
```python
from mycelium_fractal_net import simulate_mycelium_field
import numpy as np

rng = np.random.default_rng(42)
field, growth_events = simulate_mycelium_field(
    rng=rng,
    grid_size=64,
    steps=100,
    turing_enabled=True
)
```

### 3. Fractal Analysis

| Conceptual | Description | Code Path | Public API |
|------------|-------------|-----------|------------|
| Box-counting dimension | Fractal dimension estimate | `core/fractal.py` | `estimate_fractal_dimension()` |
| IFS generation | Iterated Function System | `core/fractal.py` | `generate_fractal_ifs()` |
| Lyapunov exponent | Stability metric | `core/stability.py` | `compute_lyapunov_exponent()` |

**Key Functions:**
```python
from mycelium_fractal_net import estimate_fractal_dimension, generate_fractal_ifs

# Box-counting dimension
binary = field > -0.060
D = estimate_fractal_dimension(binary)  # D ∈ [1.4, 1.9]

# IFS fractal
points, lyapunov = generate_fractal_ifs(rng, num_points=10000)
```

### 4. STDP Plasticity

| Conceptual | Description | Code Path | Public API |
|------------|-------------|-----------|------------|
| STDP learning rule | Bi & Poo (1998) | `core/stdp.py` | `STDPPlasticity` |
| Time constants | τ± = 20 ms | `core/stdp.py` | `STDP_TAU_PLUS`, `STDP_TAU_MINUS` |
| Amplitudes | A+ = 0.01, A- = 0.012 | `core/stdp.py` | `STDP_A_PLUS`, `STDP_A_MINUS` |

**Key Classes:**
```python
from mycelium_fractal_net import STDPPlasticity

stdp = STDPPlasticity(
    tau_plus=0.020,   # 20 ms
    tau_minus=0.020,  # 20 ms
    a_plus=0.01,
    a_minus=0.012
)
delta_w = stdp.compute_weight_update(pre_times, post_times, weights)
```

### 5. Federated Learning (Krum)

| Conceptual | Description | Code Path | Public API |
|------------|-------------|-----------|------------|
| Krum selection | Byzantine-robust selection | `core/federated.py` | `HierarchicalKrumAggregator.krum_select()` |
| Hierarchical aggregation | Two-level Krum + median | `core/federated.py` | `HierarchicalKrumAggregator.aggregate()` |
| Byzantine tolerance | f < (n-2)/2 | `core/federated.py` | `FEDERATED_BYZANTINE_FRACTION` |

**Key Functions:**
```python
from mycelium_fractal_net import aggregate_gradients_krum, HierarchicalKrumAggregator

# Convenience function
aggregated = aggregate_gradients_krum(gradients, num_clusters=100)

# Full control
aggregator = HierarchicalKrumAggregator(
    num_clusters=100,
    byzantine_fraction=0.2
)
aggregated = aggregator.aggregate(gradients)
```

### 6. Stability Analysis

| Conceptual | Description | Code Path | Public API |
|------------|-------------|-----------|------------|
| Lyapunov from history | Field divergence analysis | `core/stability.py` | `compute_lyapunov_exponent()` |
| Stability check | λ < 0 = stable | `core/stability.py` | `is_stable()` |
| Stability metrics | Comprehensive analysis | `core/stability.py` | `compute_stability_metrics()` |

**Key Functions:**
```python
from mycelium_fractal_net.core.stability import compute_stability_metrics, is_stable

metrics = compute_stability_metrics(field_history)
print(f"Lyapunov: {metrics['lyapunov_exponent']:.3f}")
print(f"Stable: {is_stable(metrics['lyapunov_exponent'])}")
```

---

## Directory Structure

```
mycelium-fractal-net/
├── src/mycelium_fractal_net/
│   ├── __init__.py              # Public API (aligned with README)
│   ├── model.py                 # Neural network, validation
│   ├── config.py                # Configuration management
│   ├── core/                    # Pure numerical engines
│   │   ├── __init__.py          # Core API exports
│   │   ├── nernst.py            # Nernst-Planck electrochemistry
│   │   ├── turing.py            # Turing morphogenesis
│   │   ├── fractal.py           # Fractal dimension, IFS
│   │   ├── stdp.py              # STDP plasticity
│   │   ├── federated.py         # Byzantine/Krum aggregation
│   │   ├── stability.py         # Lyapunov analysis
│   │   ├── membrane_engine.py   # Membrane potential engine
│   │   ├── reaction_diffusion_engine.py  # Turing engine
│   │   ├── fractal_growth_engine.py      # IFS engine
│   │   ├── exceptions.py        # Custom exceptions
│   │   ├── types.py             # Simulation types
│   │   └── field.py             # Field data structures
│   ├── integration/             # Schema, adapters
│   │   ├── __init__.py
│   │   ├── schemas.py           # Pydantic models
│   │   ├── adapters.py          # Core ↔ External bridge
│   │   └── service_context.py   # Execution context
│   ├── analytics/               # Feature extraction
│   │   ├── __init__.py
│   │   └── fractal_features.py
│   ├── numerics/                # Numerical utilities
│   ├── experiments/             # Experiment utilities
│   └── pipelines/               # Data pipelines
├── analytics/                   # Top-level analytics module
├── api.py                       # FastAPI server
├── mycelium_fractal_net_v4_1.py # CLI entrypoint
├── tests/
│   ├── test_public_api_structure.py  # API structure tests
│   ├── test_layer_boundaries.py      # Architecture tests
│   └── ...                           # Other tests
├── configs/                     # Configuration presets
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md
│   ├── MFN_SYSTEM_ROLE.md
│   ├── MFN_MATH_MODEL.md
│   ├── MFN_CODE_STRUCTURE.md    # This document
│   └── ...
└── ...
```

---

## Import Patterns

### Recommended: Import from main package

```python
# Best practice: import from mycelium_fractal_net
from mycelium_fractal_net import (
    compute_nernst_potential,
    simulate_mycelium_field,
    estimate_fractal_dimension,
    STDPPlasticity,
    HierarchicalKrumAggregator,
)
```

### Direct module imports (for specialized use)

```python
# Direct from core domain modules
from mycelium_fractal_net.core.nernst import compute_nernst_potential
from mycelium_fractal_net.core.turing import simulate_mycelium_field
from mycelium_fractal_net.core.fractal import estimate_fractal_dimension
from mycelium_fractal_net.core.stdp import STDPPlasticity
from mycelium_fractal_net.core.federated import HierarchicalKrumAggregator
from mycelium_fractal_net.core.stability import compute_lyapunov_exponent
```

### Engine-level imports (for full control)

```python
# Low-level engine access
from mycelium_fractal_net.core import (
    MembraneEngine,
    MembraneConfig,
    ReactionDiffusionEngine,
    ReactionDiffusionConfig,
    FractalGrowthEngine,
    FractalConfig,
)
```

---

## Adding New Features

### Adding a new numerical algorithm to core/

1. Create a new module in `src/mycelium_fractal_net/core/` (e.g., `new_algorithm.py`)
2. Add exports to `core/__init__.py`
3. Add re-exports to main `__init__.py` if it's public API
4. Add tests in `tests/` directory
5. Document in this file

### Adding a new API endpoint

1. Add request/response schemas in `integration/schemas.py`
2. Add adapter function in `integration/adapters.py`
3. Add endpoint in `api.py`
4. Add tests

### Adding new configuration options

1. Update config dataclasses in `core/` modules or `config.py`
2. Update `integration/service_context.py` if needed
3. Update documentation

---

## Test Coverage

| Module | Test File(s) | Focus |
|--------|--------------|-------|
| Public API | `test_public_api_structure.py` | Import structure, signatures, smoke tests |
| Layer boundaries | `test_layer_boundaries.py` | No infrastructure in core, module existence |
| Nernst | `test_nernst.py` | Physics validation, edge cases |
| Turing | `test_morphogenesis.py` | Pattern formation, stability |
| Fractal | `test_fractal_dimension.py` | Box-counting, IFS |
| STDP | `test_stdp.py` | Weight updates, timing |
| Federated | `test_federated.py` | Krum selection, Byzantine tolerance |
| Stability | `test_lyapunov.py` | Lyapunov computation |

---

## References

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture overview |
| [MFN_SYSTEM_ROLE.md](MFN_SYSTEM_ROLE.md) | System role and boundaries |
| [MFN_MATH_MODEL.md](MFN_MATH_MODEL.md) | Mathematical formalization |
| [NUMERICAL_CORE.md](NUMERICAL_CORE.md) | Numerical engine details |
| [TECHNICAL_AUDIT.md](TECHNICAL_AUDIT.md) | Implementation status |

---

*Document maintained by: MFN Development Team*  
*Last updated: 2025-11-30*
