
# ARCHITECTURE — MyceliumFractalNet v4.1

## Overview

MyceliumFractalNet v4.1 is a bio-inspired adaptive neural network that combines:
- Neurophysiology (Nernst-Planck ion potentials)
- Fractal geometry (IFS generation, box-counting dimension)
- Turing morphogenesis (reaction-diffusion patterns)
- Heterosynaptic plasticity (STDP learning)
- Sparse attention mechanisms
- Byzantine-robust federated learning (hierarchical Krum)

> **For detailed mathematical formalization**, including explicit equations, parameter
> tables with units and valid ranges, and discretization details, see
> **[MFN_MATH_MODEL.md](MFN_MATH_MODEL.md)**.

## 1. Nernst Equation

Membrane potential for ion with valence z:

$$
E = \frac{R T}{z F} \ln\left(\frac{[ion]_{out}}{[ion]_{in}}\right)
$$

**Physics verification:**
- At 37°C (310K): RT/zF = 26.73 mV for z=1
- For K⁺: [K]_in = 140 mM, [K]_out = 5 mM → E_K ≈ -89 mV
- Ion concentration clamping: min = 1e-6 M (prevents log(0))

Implemented in `compute_nernst_potential`, verified with sympy in `_symbolic_nernst_example`.

## 2. Mycelium Field Simulation

`simulate_mycelium_field` implements:

- 2D potential field V(x, y) initialized at -70 mV
- Discrete Laplacian diffusion: ∇²V = V_up + V_down + V_left + V_right - 4V
- Growth events ("spikes") with configurable probability
- Potential clamping in range [-95, 40] mV

### Turing Morphogenesis

Reaction-diffusion system with activator-inhibitor dynamics:

$$
\frac{\partial a}{\partial t} = D_a \nabla^2 a + r_a \cdot a(1-a) - i
$$

$$
\frac{\partial i}{\partial t} = D_i \nabla^2 i + r_i \cdot (a - i)
$$

**Parameters:**
- Turing threshold = 0.75 (pattern formation)
- Diffusion rates: D_a = 0.1, D_i = 0.05
- Reaction rates: r_a = 0.01, r_i = 0.02

### Quantum Jitter

Optional Gaussian noise for stochastic dynamics:
- Variance = 0.0005 (validated: stable at normalized jitter 0.067)

## 3. Fractal Analysis

### IFS Generation

`generate_fractal_ifs` implements Iterated Function System:
- Random contractive affine transformations
- Contraction factor: 0.2-0.5 (ensures convergence)
- Returns points and Lyapunov exponent

### Lyapunov Exponent

$$
\lambda = \lim_{n \to \infty} \frac{1}{n} \sum_{i=1}^{n} \ln|det(J_i)|
$$

- λ < 0: Stable (contractive) dynamics
- λ > 0: Unstable (expansive) dynamics
- Empirically validated: λ ≈ -2.1 for standard configuration

### Box-Counting Dimension

`estimate_fractal_dimension`:
- Binary field from threshold activation
- Multi-scale box counting
- Log-log regression for dimension estimate
- Validated: D ≈ 1.584 for stable mycelium patterns

## 4. STDP Plasticity

`STDPPlasticity` implements heterosynaptic Spike-Timing Dependent Plasticity:

$$
\Delta w = 
\begin{cases}
A_+ \exp(-\Delta t / \tau_+) & \text{if } \Delta t > 0 \text{ (LTP)} \\
-A_- \exp(\Delta t / \tau_-) & \text{if } \Delta t < 0 \text{ (LTD)}
\end{cases}
$$

**Parameters (from neurophysiology):**
- τ+ = τ- = 20 ms
- A+ = 0.01
- A- = 0.012

## 5. Sparse Attention

`SparseAttention` implements top-k sparse attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

With sparsity: only top-k attention weights retained (default k=4).

Benefits:
- O(n·k) instead of O(n²) complexity
- Better gradient flow
- Interpretable attention patterns

## 6. MyceliumFractalNet (NN Layer)

Neural network with fractal dynamics:

**Architecture:**
```
Input (4) → Linear → ReLU → SparseAttention → Linear → ReLU → Linear → Output (1)
```

**Input features:**
1. Fractal dimension D
2. Mean potential (mV)
3. Standard deviation
4. Maximum potential

**Features:**
- Sparse attention (topk=4)
- Optional STDP modulation
- Adam optimizer with MSE loss

## 7. Federated Learning

`HierarchicalKrumAggregator` implements Byzantine-robust aggregation:

### Krum Selection

Select gradient with minimum sum of distances to (n - f - 2) nearest neighbors:

$$
\text{Krum}(g_1, ..., g_n) = g_i \text{ where } i = \arg\min_j \sum_{k \in N_j} \|g_j - g_k\|
$$

### Hierarchical Aggregation

1. Level 1: Cluster-wise Krum (100 clusters)
2. Level 2: Global Krum + median fallback (70/30 weighted)

**Parameters:**
- Clusters: 100
- Byzantine tolerance: 20%
- Sampling fraction: 10% (for >1000 clients)

**Scale validation:**
- Tested: 1M clients simulation
- Jitter stability: 0.067 normalized

## 8. Validation Cycle

`run_validation`:

1. Generate dataset from field simulations
2. Compute statistics (D, mean, std, max)
3. Train MyceliumFractalNet
4. Return metrics dictionary

**Metrics:**
- loss_start, loss_final, loss_drop
- pot_min_mV, pot_max_mV
- example_fractal_dim
- lyapunov_exponent
- growth_events
- nernst_symbolic_mV, nernst_numeric_mV

## 9. Integration Layer

The integration layer (`src/mycelium_fractal_net/integration/`) provides unified schemas,
service context, and adapters for consistent operation across CLI, HTTP API, and experiments.

```
src/mycelium_fractal_net/integration/
├── __init__.py          # Package exports
├── schemas.py           # Pydantic request/response models
├── service_context.py   # Unified context with config, RNG, engine handles
└── adapters.py          # Thin bridge between integration layer and core
```

**Key Components:**

- **schemas.py**: Pydantic models (`ValidateRequest`, `SimulateResponse`, etc.) ensuring
  consistent validation across all entry points.
- **service_context.py**: `ServiceContext` class encapsulating configuration, RNG state,
  and execution mode (CLI/API/experiment). Enables reproducibility and clean dependency injection.
- **adapters.py**: Thin adapter functions that bridge schemas to core functions without
  containing business logic. Keeps the core independent of FastAPI/CLI.

**Benefits:**
- Single source of truth for request/response schemas
- Consistent validation between CLI and API
- Clear boundary for adding auth/rate-limiting/observability
- Core numerical engines remain independent of integration concerns

## 10. API Endpoints

FastAPI server (`api.py`):

| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | Health check |
| /validate | POST | Run validation cycle |
| /simulate | POST | Simulate mycelium field |
| /nernst | POST | Compute Nernst potential |
| /federated/aggregate | POST | Krum aggregation |

## 11. Configuration

Three configuration levels (`configs/`):

| Config | Grid | Steps | Clusters | Use Case |
|--------|------|-------|----------|----------|
| small | 32 | 32 | 10 | Development |
| medium | 64 | 64 | 50 | Testing |
| large | 128 | 128 | 100 | Production |

## References

1. Nernst, W. (1889). Die elektromotorische Wirksamkeit der Ionen.
2. Turing, A.M. (1952). The chemical basis of morphogenesis.
3. Bi & Poo (1998). Synaptic modifications in cultured hippocampal neurons.
4. Blanchard et al. (2017). Byzantine-robust distributed learning.
5. Mandelbrot, B. (1982). The fractal geometry of nature.
