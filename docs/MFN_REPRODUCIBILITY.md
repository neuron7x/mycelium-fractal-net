# MFN Reproducibility Guide

This document describes how MyceliumFractalNet ensures deterministic, reproducible simulations and experiments.

## Overview

MFN provides three layers of reproducibility:

1. **RNG Control Layer** — Unified random number generator management
2. **Canonical Configurations** — Serializable config types with seed fields
3. **Run Registry** — File-based experiment tracking with metadata

## RNG Control Layer

### Creating Reproducible RNG Contexts

```python
from mycelium_fractal_net.infra.rng import create_rng, set_global_seed

# Create an isolated RNG context
rng_ctx = create_rng(seed=42)
rng = rng_ctx.numpy_rng

# Use the RNG for reproducible operations
value = rng.random()
```

### Global Seed Setting

For full reproducibility across all random operations:

```python
from mycelium_fractal_net.infra.rng import set_global_seed

# Set global seed (affects NumPy, Python random, and PyTorch if available)
set_global_seed(42)
```

### Forking RNG Contexts

Create independent child contexts for parallel operations:

```python
rng_ctx = create_rng(seed=42)

# Fork for child operations
child_ctx = rng_ctx.fork()

# Child has independent state
child_rng = child_ctx.numpy_rng
```

### State Serialization

Save and restore RNG state:

```python
# Save state
state = rng_ctx.get_state()

# Restore state
from mycelium_fractal_net.infra.rng import RNGContext
restored_ctx = RNGContext.from_state(state)
```

## Configuration Types

All MFN configurations support reproducibility through:

- Required `seed` field (or `base_seed` for dataset configs)
- `to_dict()` / `from_dict()` methods for serialization

### SimulationConfig

```python
from mycelium_fractal_net import SimulationConfig

config = SimulationConfig(
    grid_size=64,
    steps=100,
    alpha=0.18,
    spike_probability=0.25,
    turing_enabled=True,
    seed=42,  # For reproducibility
)

# Run simulation - deterministic with same seed
result = run_mycelium_simulation(config)
```

### DatasetConfig

```python
from mycelium_fractal_net import DatasetConfig

config = DatasetConfig(
    num_samples=200,
    grid_sizes=[32, 64],
    base_seed=42,  # Base seed for all samples
)

# Generate dataset - each sample has derived seed
```

### ScenarioConfig

```python
from mycelium_fractal_net.pipelines.scenarios import ScenarioConfig

config = ScenarioConfig(
    name="my_experiment",
    grid_size=64,
    steps=100,
    base_seed=42,
)
```

## Run Registry

The Run Registry tracks experiment metadata for audit and reproducibility.

### Basic Usage

```python
from mycelium_fractal_net.infra.run_registry import RunRegistry

registry = RunRegistry()

# Start a run
run = registry.start_run(
    config=my_config,
    run_type="validation",
    seed=42,
    command="python validate.py --seed 42",
)

# ... run experiment ...

# Log metrics
registry.log_metrics(run, {
    "loss": 0.5,
    "accuracy": 0.95,
    "fractal_dim": 1.58,
})

# End run
registry.end_run(run, status="success")
```

### Directory Structure

```
runs/
└── YYYYMMDD_HHMMSS_<shortid>/
    ├── config.json     # Full configuration
    ├── meta.json       # Run metadata
    └── metrics.json    # Key metrics
```

### meta.json Fields

| Field | Description |
|-------|-------------|
| `run_id` | Unique run identifier |
| `run_type` | Type of run (validation, benchmark, experiment) |
| `timestamp` | ISO format start time |
| `git_commit` | Git commit hash |
| `command` | Command that started the run |
| `env` | Environment (dev/stage/prod) |
| `seed` | Random seed used |
| `status` | running/success/failed/cancelled |
| `end_time` | ISO format end time (after completion) |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MFN_RUN_REGISTRY_ENABLED` | Enable/disable registry | `true` |
| `MFN_RUN_REGISTRY_DIR` | Override runs directory | `runs/` |

### Retrieving Run Data

```python
# Get specific run
run_data = registry.get_run("20241201_123456_abc123")

# List recent runs
runs = registry.list_runs(run_type="validation", limit=10)
```

## CLI Usage

### Setting Seed via Command Line

```bash
# Validation CLI
python mycelium_fractal_net_v4_1.py --mode validate --seed 42

# Dataset generation
python -m mycelium_fractal_net.experiments.generate_dataset --seed 42
```

## Reproducing an Experiment

### From Run Registry

1. Find the run:
```python
registry = RunRegistry()
run_data = registry.get_run("20241201_123456_abc123")
```

2. Extract configuration:
```python
config = run_data["config"]
seed = run_data["meta"]["seed"]
```

3. Re-run with same config:
```python
from mycelium_fractal_net import SimulationConfig, run_mycelium_simulation

sim_config = SimulationConfig(**config)
result = run_mycelium_simulation(sim_config)
```

### From Config File

```python
import json
from mycelium_fractal_net import SimulationConfig

# Load config
with open("runs/20241201_123456_abc123/config.json") as f:
    config_dict = json.load(f)

# Create and run
config = SimulationConfig(**config_dict)
result = run_mycelium_simulation(config)
```

## Guarantees

MFN provides the following reproducibility guarantees:

| Component | Guarantee |
|-----------|-----------|
| `run_mycelium_simulation` | Same config + seed → identical `field` and `growth_events` |
| `run_mycelium_simulation_with_history` | Same config + seed → identical history trajectory |
| `estimate_fractal_dimension` | Deterministic for same input |
| `compute_nernst_potential` | Deterministic (no randomness) |
| Dataset generation | Same `base_seed` → identical dataset |

## Limitations

### PyTorch/CUDA Determinism

Full determinism with PyTorch requires additional settings:

```python
import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Note: Some CUDA operations may not have deterministic implementations.

### Floating-Point Precision

Very small differences (< 1e-10) may occur due to:
- Different CPU instruction orderings
- Different compiler optimizations
- Different hardware (Intel vs AMD)

For comparison, use tolerances:
```python
import numpy as np
np.testing.assert_allclose(result1.field, result2.field, rtol=1e-10)
```

## Best Practices

1. **Always set seed explicitly** in production configs
2. **Use RNGContext** for isolated random state
3. **Log configs to Run Registry** for audit trails
4. **Include git commit** in experiment metadata
5. **Test determinism** as part of CI/CD

## Reference

- `src/mycelium_fractal_net/infra/rng.py` — RNG control layer
- `src/mycelium_fractal_net/infra/run_registry.py` — Run registry
- `src/mycelium_fractal_net/config.py` — Configuration types
- `tests/repro/` — Reproducibility test suite
