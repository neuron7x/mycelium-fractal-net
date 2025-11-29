# MFN Integration Specification

## Overview

This document defines the integration specification for MyceliumFractalNet (MFN) v4.1 into
the main repository. It establishes the package structure, public API, dependencies, and
a roadmap for complete integration across 7 pull requests.

**Document Version**: 1.0  
**Target Version**: MyceliumFractalNet v4.1.0  
**Status**: PR-1 (Integration Specification & Package Scaffold)

---

## 1. Repository Structure Analysis

### 1.1 Package Schema

The repository follows the `src` layout pattern:

```
mycelium-fractal-net/
├── src/
│   └── mycelium_fractal_net/      # Main package
│       ├── __init__.py            # Public API exports
│       ├── model.py               # Core implementation
│       └── core/                  # Numerical engines
│           ├── __init__.py
│           ├── exceptions.py
│           ├── membrane_engine.py
│           ├── reaction_diffusion_engine.py
│           └── fractal_growth_engine.py
├── analytics/                     # Feature extraction module
│   ├── __init__.py
│   └── fractal_features.py
├── experiments/                   # Dataset generation
│   ├── __init__.py
│   ├── generate_dataset.py
│   └── inspect_features.py
└── tests/                         # Test suite
    ├── core/                      # Core engine tests
    ├── test_analytics/            # Analytics tests
    ├── integration/               # Integration tests
    └── test_*.py                  # Component tests
```

### 1.2 Test Structure

Tests are organized in `tests/` with pytest as the test runner:

- **Component tests**: `tests/test_*.py` - Individual module tests
- **Core tests**: `tests/core/` - Numerical engine tests
- **Analytics tests**: `tests/test_analytics/` - Feature extraction tests
- **Integration tests**: `tests/integration/` - Package-level smoke tests

### 1.3 Code Style Standards

| Tool | Configuration | Purpose |
|------|---------------|---------|
| ruff | `pyproject.toml` | Linting (E, F, I rules) |
| mypy | strict mode | Type checking |
| black | line-length 100 | Formatting |
| isort | implicit | Import sorting |
| pytest | `-q --disable-warnings` | Testing |

**Python Version**: ≥3.10

---

## 2. MFN Source Tree

### 2.1 Current Structure (`mfn_source_tree`)

```
mycelium-fractal-net/
├── src/mycelium_fractal_net/
│   ├── __init__.py                 # Package entry, exports 35+ public symbols
│   ├── model.py                    # ~1013 LOC: Core algorithms
│   │   ├── compute_nernst_potential()
│   │   ├── simulate_mycelium_field()
│   │   ├── estimate_fractal_dimension()
│   │   ├── generate_fractal_ifs()
│   │   ├── compute_lyapunov_exponent()
│   │   ├── STDPPlasticity
│   │   ├── SparseAttention
│   │   ├── HierarchicalKrumAggregator
│   │   ├── MyceliumFractalNet
│   │   └── run_validation()
│   └── core/
│       ├── __init__.py             # Core engine exports
│       ├── exceptions.py           # StabilityError, ValueOutOfRangeError, etc.
│       ├── membrane_engine.py      # MembraneEngine: Nernst/GHK equations
│       ├── reaction_diffusion_engine.py  # ReactionDiffusionEngine: Turing PDEs
│       └── fractal_growth_engine.py      # FractalGrowthEngine: IFS/DLA
├── analytics/
│   ├── __init__.py                 # 7 exported symbols
│   └── fractal_features.py         # 18 fractal features, FeatureVector class
├── experiments/
│   ├── __init__.py                 # Dataset generation exports
│   ├── generate_dataset.py         # Parameter sweep pipeline
│   └── inspect_features.py         # Exploratory analysis utilities
├── docs/
│   ├── ARCHITECTURE.md             # System architecture
│   ├── MATH_MODEL.md               # Mathematical formalization
│   ├── NUMERICAL_CORE.md           # Numerical implementation details
│   ├── FEATURE_SCHEMA.md           # Feature extraction schema
│   └── ROADMAP.md                  # Development roadmap
├── configs/
│   ├── small.yaml                  # Development config
│   ├── medium.yaml                 # Testing config
│   └── large.yaml                  # Production config
├── api.py                          # FastAPI server
├── mycelium_fractal_net_v4_1.py    # CLI entry point
├── Dockerfile                      # Container build
└── k8s.yaml                        # Kubernetes deployment
```

### 2.2 Target Integration Tree (`target_integration_tree`)

The current structure is already well-organized. No major restructuring needed.
Target structure maintains compatibility:

```
src/mycelium_fractal_net/
├── __init__.py                     # Public API (maintained)
├── model.py                        # Core implementation (maintained)
└── core/                           # Numerical engines (maintained)
    ├── __init__.py
    ├── exceptions.py
    ├── membrane_engine.py
    ├── reaction_diffusion_engine.py
    └── fractal_growth_engine.py

tests/integration/                  # Integration tests (NEW)
├── __init__.py
└── test_imports.py                 # Smoke tests

docs/                               # Documentation
├── MFN_INTEGRATION_SPEC.md         # This document (NEW)
└── (existing docs maintained)
```

---

## 3. Public API Specification

### 3.1 Top-Level Package Name

**Package**: `mycelium_fractal_net`

Import: `import mycelium_fractal_net` or `from mycelium_fractal_net import ...`

### 3.2 Core Functions

#### 3.2.1 `compute_nernst_potential`

```python
def compute_nernst_potential(
    z_valence: int,
    concentration_out_molar: float,
    concentration_in_molar: float,
    temperature_k: float = 310.0,
) -> float:
    """
    Compute membrane equilibrium potential using Nernst equation.

    Parameters
    ----------
    z_valence : int
        Ion valence (K⁺ = +1, Ca²⁺ = +2, Cl⁻ = -1).
    concentration_out_molar : float
        Extracellular concentration in mol/L.
    concentration_in_molar : float
        Intracellular concentration in mol/L.
    temperature_k : float
        Temperature in Kelvin (default: 310 K ≈ 37°C).

    Returns
    -------
    float
        Membrane potential in Volts.

    Example
    -------
    >>> E_K = compute_nernst_potential(1, 5e-3, 140e-3)  # K⁺
    >>> print(f"{E_K * 1000:.1f} mV")  # ≈ -89 mV
    """
```

#### 3.2.2 `simulate_mycelium_field`

```python
def simulate_mycelium_field(
    rng: np.random.Generator,
    grid_size: int = 64,
    steps: int = 64,
    alpha: float = 0.18,
    spike_probability: float = 0.25,
    turing_enabled: bool = True,
    turing_threshold: float = 0.75,
    quantum_jitter: bool = False,
    jitter_var: float = 0.0005,
) -> tuple[NDArray, int]:
    """
    Simulate mycelium-like potential field with Turing morphogenesis.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded random number generator for reproducibility.
    grid_size : int
        Spatial resolution N×N (default: 64).
    steps : int
        Number of simulation timesteps (default: 64).
    alpha : float
        Diffusion coefficient (default: 0.18, stable range: 0.05-0.24).
    spike_probability : float
        Per-step probability of growth event (default: 0.25).
    turing_enabled : bool
        Enable activator-inhibitor dynamics (default: True).
    turing_threshold : float
        Pattern activation threshold (default: 0.75).
    quantum_jitter : bool
        Enable stochastic noise (default: False).
    jitter_var : float
        Noise variance (default: 0.0005).

    Returns
    -------
    field : NDArray[float64]
        Final potential field in Volts, shape (N, N).
    growth_events : int
        Number of growth events during simulation.

    Example
    -------
    >>> rng = np.random.default_rng(42)
    >>> field, events = simulate_mycelium_field(rng, grid_size=64, steps=100)
    >>> print(f"Range: [{field.min()*1000:.1f}, {field.max()*1000:.1f}] mV")
    """
```

#### 3.2.3 `estimate_fractal_dimension`

```python
def estimate_fractal_dimension(
    binary_field: NDArray[np.bool_],
    min_box_size: int = 2,
    max_box_size: int | None = None,
    num_scales: int = 5,
) -> float:
    """
    Estimate box-counting fractal dimension of binary pattern.

    Parameters
    ----------
    binary_field : NDArray[bool]
        Binary 2D field, shape (N, N).
    min_box_size : int
        Minimum box size (default: 2).
    max_box_size : int | None
        Maximum box size (default: N//2).
    num_scales : int
        Number of logarithmic scales for regression (default: 5).

    Returns
    -------
    float
        Estimated fractal dimension D ∈ [0, 2].
        Typical mycelium patterns: D ∈ [1.4, 1.9].

    Example
    -------
    >>> binary = field > -0.060  # Threshold at -60 mV
    >>> D = estimate_fractal_dimension(binary)
    >>> print(f"D = {D:.3f}")  # Expected: ~1.5-1.6
    """
```

### 3.3 Core Classes

#### 3.3.1 `MyceliumFractalNet`

```python
class MyceliumFractalNet(nn.Module):
    """
    Neural network with fractal dynamics, STDP plasticity, and sparse attention.

    Parameters
    ----------
    input_dim : int
        Input feature dimension (default: 4).
    hidden_dim : int
        Hidden layer dimension (default: 32).
    use_sparse_attention : bool
        Enable top-k sparse attention (default: True).
    use_stdp : bool
        Enable STDP modulation (default: True).

    Architecture
    ------------
    Input → Linear → ReLU → SparseAttention → Linear → ReLU → Linear → Output

    Example
    -------
    >>> model = MyceliumFractalNet(input_dim=4, hidden_dim=32)
    >>> x = torch.randn(8, 4)  # batch=8, features=4
    >>> out = model(x)  # shape: (8, 1)
    """
```

#### 3.3.2 `HierarchicalKrumAggregator`

```python
class HierarchicalKrumAggregator:
    """
    Byzantine-robust federated learning aggregator using hierarchical Krum.

    Parameters
    ----------
    num_clusters : int
        Number of client clusters (default: 100).
    byzantine_fraction : float
        Expected fraction of Byzantine clients (default: 0.2).
    sample_fraction : float
        Fraction of clients to sample when n > 1000 (default: 0.1).

    Methods
    -------
    aggregate(client_gradients, rng=None) -> Tensor
        Aggregate gradients with Byzantine robustness.

    Example
    -------
    >>> aggregator = HierarchicalKrumAggregator(num_clusters=50)
    >>> gradients = [torch.randn(100) for _ in range(1000)]
    >>> result = aggregator.aggregate(gradients)
    """
```

### 3.4 Core Numerical Engines

| Engine | Config Class | Metrics Class | Purpose |
|--------|--------------|---------------|---------|
| `MembraneEngine` | `MembraneConfig` | `MembraneMetrics` | Nernst/GHK potential |
| `ReactionDiffusionEngine` | `ReactionDiffusionConfig` | `ReactionDiffusionMetrics` | Turing PDEs |
| `FractalGrowthEngine` | `FractalConfig` | `FractalMetrics` | IFS/DLA growth |

---

## 4. Dependencies

### 4.1 Required Dependencies

| Package | Version | Purpose | Status |
|---------|---------|---------|--------|
| numpy | ≥1.24 | Numerical computing | In project |
| torch | ≥2.0.0 | Neural networks | In project |
| sympy | ≥1.12 | Symbolic verification | In project |

### 4.2 Optional Dependencies

| Package | Version | Purpose | Status |
|---------|---------|---------|--------|
| fastapi | ≥0.109.0 | REST API | In requirements.txt |
| uvicorn | ≥0.27.0 | ASGI server | In requirements.txt |
| pydantic | ≥2.0.0 | Data validation | In requirements.txt |

### 4.3 Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | ≥7.0 | Testing |
| pytest-cov | ≥4.0 | Coverage |
| ruff | ≥0.5.0 | Linting |
| mypy | ≥1.8.0 | Type checking |
| black | ≥24.0.0 | Formatting |
| isort | ≥5.12.0 | Import sorting |
| hypothesis | (optional) | Property-based testing |
| scipy | (optional) | Scientific computing |

---

## 5. Resource Constraints

### 5.1 Typical Simulation Parameters

| Configuration | Grid Size | Steps | Expected Time | RAM Usage |
|---------------|-----------|-------|---------------|-----------|
| Small (dev) | 32×32 | 32 | ~0.1s | ~10 MB |
| Medium (test) | 64×64 | 64 | ~0.5s | ~50 MB |
| Large (prod) | 128×128 | 128 | ~2s | ~200 MB |

### 5.2 Computational Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Nernst potential | O(1) | O(1) |
| Field simulation | O(N² × T) | O(N²) |
| Fractal dimension | O(N² × k) | O(N²) |
| IFS generation | O(n_points) | O(n_points) |
| Krum aggregation | O(n² × d) | O(n × d) |

Where: N = grid size, T = steps, k = scales, n = clients, d = gradient dim.

### 5.3 Numerical Stability Constraints

| Parameter | Valid Range | Default | Notes |
|-----------|-------------|---------|-------|
| Diffusion α | 0.05–0.24 | 0.18 | CFL: α ≤ 0.25 |
| Turing D_a | 0.01–0.5 | 0.1 | D_a < D_max |
| Turing D_i | 0.01–0.3 | 0.05 | D_i < D_a |
| IFS scale | 0.2–0.5 | — | Contraction |
| Ion clamp | ≥1e-6 | 1e-6 | Prevents log(0) |

---

## 6. PR Roadmap

### PR-1: Integration Specification & Package Scaffold (This PR)

**Scope**: Documentation and minimal code structure.

**Deliverables**:
- [x] `docs/MFN_INTEGRATION_SPEC.md` — This document
- [x] `tests/integration/__init__.py` — Test package init
- [x] `tests/integration/test_imports.py` — Smoke tests

**Verification**:
```bash
pytest tests/integration/test_imports.py -v
```

---

### PR-2: Core Simulation Transfer

**Scope**: Ensure `model.py` core functions are properly tested and documented.

**Deliverables**:
- Integration tests for `simulate_mycelium_field()`
- Integration tests for `compute_nernst_potential()`
- Docstring improvements for public API
- Type annotations verification

**Acceptance Criteria**:
- All smoke tests pass
- 100% type coverage on public API
- Nernst potential: E_K ≈ -89 mV ± 2 mV

---

### PR-3: Numerical Schemes Formalization

**Scope**: Validate and document numerical stability of PDE solvers.

**Deliverables**:
- Stability tests for reaction-diffusion engine
- CFL condition verification tests
- Boundary condition tests (periodic)
- Performance benchmarks for grid sizes 32-256

**Acceptance Criteria**:
- No NaN/Inf after 1000+ steps
- Turing patterns form reproducibly (seed-deterministic)
- Explicit stability bounds documented

---

### PR-4: Fractal Analytics & Feature Engineering

**Scope**: Validate and extend `analytics` module.

**Deliverables**:
- Integration with `analytics.compute_features()`
- Validation of 18-feature extraction
- Box-counting dimension tests
- Feature range validation (D ∈ [1.4, 1.9] for biological patterns)

**Acceptance Criteria**:
- Feature extraction completes for all grid sizes
- R² > 0.9 for dimension regression
- No NaN in any feature

---

### PR-5: Experimental Dataset Generation

**Scope**: Validate and document dataset generation pipeline.

**Deliverables**:
- Parameter sweep automation tests
- Dataset schema validation
- Parquet output verification
- Reproducibility tests (seeded generation)

**Acceptance Criteria**:
- `generate_dataset.py` produces valid parquet files
- All 18 features present in output
- Deterministic with fixed seed

---

### PR-6: System Integration

**Scope**: Integration with external consumers (API, CLI).

**Deliverables**:
- API endpoint tests (`api.py`)
- CLI validation (`mycelium_fractal_net_v4_1.py`)
- Docker build verification
- Integration with external modules (if any)

**Acceptance Criteria**:
- `/validate` endpoint returns valid metrics
- CLI `--mode validate` passes
- Docker container runs successfully

---

### PR-7: Optimization, Profiling & Finalization

**Scope**: Performance optimization and production readiness.

**Deliverables**:
- Performance profiling report
- Memory optimization (if needed)
- API stability tests
- Load testing for federated learning (simulated 1M clients)
- Final documentation review

**Acceptance Criteria**:
- Medium config: <1s end-to-end
- No memory leaks in long-running tests
- All CI checks pass
- Documentation complete and accurate

---

## 7. Verification Commands

### Run Smoke Tests

```bash
pytest tests/integration/test_imports.py -v
```

### Run Full Test Suite

```bash
pytest -q
```

### Run Linters

```bash
ruff check .
mypy src/mycelium_fractal_net
```

### Run Validation CLI

```bash
python mycelium_fractal_net_v4_1.py --mode validate --seed 42 --epochs 1
```

---

## 8. References

- [ARCHITECTURE.md](ARCHITECTURE.md) — System architecture
- [MATH_MODEL.md](MATH_MODEL.md) — Mathematical formalization
- [NUMERICAL_CORE.md](NUMERICAL_CORE.md) — Numerical implementation
- [FEATURE_SCHEMA.md](FEATURE_SCHEMA.md) — Feature extraction schema
- [ROADMAP.md](ROADMAP.md) — Development roadmap

---

*Document maintained by: Integration Team*  
*Last updated: 2025-11-29*
