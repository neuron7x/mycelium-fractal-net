# NUMERICAL_CORE.md — Numerical Implementation Guide

## Overview

This document describes the numerical core engines for MyceliumFractalNet v4.1.
The core provides stable, parameterized implementations for:

1. **Membrane Engine** — ODE integration for membrane potentials
2. **Reaction-Diffusion Engine** — PDE solver for Turing morphogenesis
3. **Fractal Growth Engine** — IFS/DLA for fractal pattern generation

Each engine is designed with:
- Explicit stability conditions
- NaN/Inf detection with custom exceptions
- Configurable parameters via dataclasses
- Metrics collection for monitoring
- Deterministic operation with `random_seed`

Reference: [ARCHITECTURE.md](ARCHITECTURE.md) for mathematical specifications.

---

## 1. Membrane Engine

### Mathematical Model

Membrane potential dynamics based on Nernst-Planck ion dynamics:

$$
E = \frac{RT}{zF} \ln\left(\frac{[ion]_{out}}{[ion]_{in}}\right)
$$

Membrane potential evolution (passive decay):

$$
\frac{dV}{dt} = \frac{V_{rest} - V}{\tau} + I_{ext}
$$

### Numerical Scheme

| Scheme | Order | Stability Condition | Use Case |
|--------|-------|---------------------|----------|
| Euler | 1st | `dt < τ` | Fast, simple |
| RK4 | 4th | `dt < 2τ` | Accurate |

**Default parameters:**
- `dt = 0.001 s` (1 ms)
- `τ = 0.010 s` (10 ms)
- `V_rest = -70 mV`
- `V_min = -95 mV`, `V_max = +40 mV`

### Usage

```python
from mycelium_fractal_net.core import MembraneEngine, MembraneEngineConfig

config = MembraneEngineConfig(
    dt=0.001,
    tau=0.010,
    v_rest=-0.070,
    random_seed=42,
)
engine = MembraneEngine(config)

# Simulate 100 neurons for 1000 steps
V, metrics = engine.simulate(n_neurons=100, steps=1000)

print(f"Mean potential: {metrics.v_mean * 1000:.1f} mV")
print(f"Execution time: {metrics.execution_time_s:.3f} s")
```

### Stability Checks

- NaN/Inf detection after each step
- Automatic clamping to `[V_min, V_max]`
- Raises `NumericalInstabilityError` if unstable

---

## 2. Reaction-Diffusion Engine

### Mathematical Model

Activator-inhibitor reaction-diffusion (Turing morphogenesis):

$$
\frac{\partial a}{\partial t} = D_a \nabla^2 a + r_a \cdot a(1-a) - i
$$

$$
\frac{\partial i}{\partial t} = D_i \nabla^2 i + r_i \cdot (a - i)
$$

### Spatial Discretization

5-point Laplacian stencil with periodic boundary conditions:

$$
\nabla^2 u \approx \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{\Delta x^2}
$$

### Stability Condition (CFL)

For explicit Euler:

$$
\Delta t \leq \frac{\Delta x^2}{4 \cdot \max(D_a, D_i)}
$$

With `Δx = 1` (unit grid spacing) and default diffusion `D = 0.1`:
- `dt_max = 2.5` (CFL number < 1 for stability)

**Default parameters:**
- `grid_size = 64`
- `dt = 0.1`
- `steps = 100`
- `D_a = 0.1`, `D_i = 0.05`
- `r_a = 0.01`, `r_i = 0.02`
- `turing_threshold = 0.75`

### Boundary Conditions

| Type | Description |
|------|-------------|
| `PERIODIC` | Wrap-around (default) |
| `NEUMANN` | Zero-flux (∂u/∂n = 0) |
| `DIRICHLET` | Fixed boundary (u = 0) |

### Usage

```python
from mycelium_fractal_net.core import ReactionDiffusionEngine, ReactionDiffusionConfig

config = ReactionDiffusionConfig(
    grid_size=64,
    steps=200,
    dt=0.1,
    random_seed=42,
)
engine = ReactionDiffusionEngine(config)

activator, inhibitor, metrics = engine.simulate()

print(f"Pattern fraction: {metrics.pattern_fraction:.2%}")
print(f"CFL number: {metrics.cfl_number:.3f}")
```

### Field Coupling Mode

Couple Turing patterns to membrane potential field:

```python
field = np.full((64, 64), -0.070)  # -70 mV
activator, inhibitor, field_out, metrics = engine.simulate_with_field(
    field, field_coupling=0.005
)
```

---

## 3. Fractal Growth Engine

### Mathematical Model

**Iterated Function System (IFS):**

$$
x_{n+1} = A_i \cdot x_n + b_i
$$

Where affine maps have contraction factor `s ∈ [0.2, 0.5]`.

**Lyapunov exponent:**

$$
\lambda = \lim_{n \to \infty} \frac{1}{n} \sum_{i=1}^{n} \ln|det(J_i)|
$$

Stability condition: `λ < 0` (contractive dynamics).

**Box-counting dimension:**

$$
D = \lim_{\epsilon \to 0} \frac{\ln N(\epsilon)}{\ln(1/\epsilon)}
$$

**Default parameters:**
- `num_points = 10000`
- `num_transforms = 4`
- `contraction_min = 0.2`, `contraction_max = 0.5`
- Expected `D ≈ 1.585` (mycelium networks)

### Usage

```python
from mycelium_fractal_net.core import FractalGrowthEngine, FractalGrowthConfig

config = FractalGrowthConfig(
    num_points=10000,
    num_transforms=4,
    random_seed=42,
)
engine = FractalGrowthEngine(config)

points, metrics = engine.generate_ifs()

print(f"Lyapunov exponent: {metrics.lyapunov_exponent:.4f}")
print(f"Fractal dimension: {metrics.fractal_dimension:.3f}")
print(f"Stable: {metrics.is_stable}")
```

### Diffusion-Limited Aggregation (DLA)

```python
config = FractalGrowthConfig(
    grid_size=64,
    max_iterations=500,
    random_seed=42,
)
engine = FractalGrowthEngine(config)

grid, metrics = engine.generate_dla()

print(f"Particles: {metrics.dla_particles}")
print(f"Fill fraction: {metrics.grid_fill_fraction:.3f}")
```

---

## 4. Exception Hierarchy

```
StabilityError (base)
├── ValueOutOfRangeError
│   - value, min_bound, max_bound
└── NumericalInstabilityError
    - field_name, nan_count, inf_count
```

### Handling Exceptions

```python
from mycelium_fractal_net.core import (
    MembraneEngine,
    MembraneEngineConfig,
    NumericalInstabilityError,
    StabilityError,
)

try:
    engine = MembraneEngine(config)
    V, metrics = engine.simulate(n_neurons=100, steps=1000)
except NumericalInstabilityError as e:
    print(f"NaN/Inf at step {e.step}: {e.nan_count} NaN, {e.inf_count} Inf")
except StabilityError as e:
    print(f"Stability error: {e}")
```

---

## 5. Configuration Reference

### MembraneEngineConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dt` | float | 0.001 | Time step (s) |
| `v_rest` | float | -0.070 | Resting potential (V) |
| `v_min` | float | -0.095 | Min potential (V) |
| `v_max` | float | 0.040 | Max potential (V) |
| `tau` | float | 0.010 | Membrane time constant (s) |
| `temperature_k` | float | 310.0 | Temperature (K) |
| `integration_scheme` | enum | EULER | EULER or RK4 |
| `check_stability` | bool | True | Enable NaN/Inf checks |
| `ion_clamp_min` | float | 1e-6 | Min ion concentration (M) |
| `random_seed` | int | None | RNG seed |

### ReactionDiffusionConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid_size` | int | 64 | Grid size (N×N) |
| `dt` | float | 0.1 | Time step |
| `steps` | int | 100 | Number of steps |
| `d_activator` | float | 0.1 | Activator diffusion |
| `d_inhibitor` | float | 0.05 | Inhibitor diffusion |
| `r_activator` | float | 0.01 | Activator reaction rate |
| `r_inhibitor` | float | 0.02 | Inhibitor reaction rate |
| `turing_threshold` | float | 0.75 | Pattern threshold |
| `boundary` | enum | PERIODIC | Boundary condition |
| `check_stability` | bool | True | Enable checks |
| `random_seed` | int | None | RNG seed |

### FractalGrowthConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_points` | int | 10000 | IFS points |
| `num_transforms` | int | 4 | Affine transforms |
| `contraction_min` | float | 0.2 | Min contraction |
| `contraction_max` | float | 0.5 | Max contraction |
| `max_iterations` | int | 1000 | DLA iterations |
| `grid_size` | int | 64 | DLA grid size |
| `dla_enabled` | bool | False | Enable DLA mode |
| `check_stability` | bool | True | Enable checks |
| `random_seed` | int | None | RNG seed |
| `box_min_size` | int | 2 | Min box size |
| `box_num_scales` | int | 5 | Number of scales |

---

## 6. Performance Benchmarks

| Engine | Configuration | Time (s) |
|--------|--------------|----------|
| Membrane | 100 neurons, 1000 steps | < 0.1 |
| Reaction-Diffusion | 64×64, 200 steps | < 1.0 |
| IFS | 10000 points | < 0.1 |
| DLA | 32×32, 100 iterations | < 3.0 |

All benchmarks on CPU (Intel i7 / M1).

---

## 7. Reproducibility

For deterministic results:

1. **Always set `random_seed`** in config
2. Use `engine.reset()` to restore initial state
3. Same seed + same parameters = identical results

```python
# Reproducible workflow
config = MembraneEngineConfig(random_seed=42)
engine = MembraneEngine(config)

V1, _ = engine.simulate(n_neurons=50, steps=100)
engine.reset()
V2, _ = engine.simulate(n_neurons=50, steps=100)

assert np.array_equal(V1, V2)  # Always True
```

---

## 8. Integration with Existing API

The core engines integrate with the existing `mycelium_fractal_net` API:

```python
from mycelium_fractal_net import simulate_mycelium_field
from mycelium_fractal_net.core import (
    ReactionDiffusionEngine,
    ReactionDiffusionConfig,
)

# New core API
config = ReactionDiffusionConfig(grid_size=64, steps=100, random_seed=42)
engine = ReactionDiffusionEngine(config)
activator, inhibitor, metrics = engine.simulate()

# Existing API (compatible)
rng = np.random.default_rng(42)
field, events = simulate_mycelium_field(rng, grid_size=64, steps=100)
```

---

## References

1. Nernst, W. (1889). Die elektromotorische Wirksamkeit der Ionen.
2. Turing, A.M. (1952). The chemical basis of morphogenesis.
3. Mandelbrot, B. (1982). The fractal geometry of nature.
4. Barnsley, M. (1988). Fractals Everywhere.
5. Press et al. (2007). Numerical Recipes 3rd Ed.
