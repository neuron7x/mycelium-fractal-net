# MFN Reproducibility Guide

This document describes how MyceliumFractalNet (MFN) ensures reproducible simulations, experiments, and results.

## Overview

MFN provides a comprehensive reproducibility framework consisting of:

1. **Unified RNG Control Layer** — Centralized random number generator management
2. **Run Registry** — File-based tracking of experimental runs with metadata
3. **Canonical Configuration Types** — Serializable config objects with `to_dict()`/`from_dict()` methods
4. **Deterministic Seeds** — Every simulation/experiment can be seeded for exact reproduction

## Quick Start

### Running a Reproducible Simulation

```python
from mycelium_fractal_net import (
    SimulationConfig,
    run_mycelium_simulation_with_history,
    create_rng,
)

# Option 1: Use seed in config
config = SimulationConfig(
    grid_size=64,
    steps=100,
    seed=42,  # Deterministic seed
    turing_enabled=True,
)
result = run_mycelium_simulation_with_history(config)

# Option 2: Use RNGContext for more control
from mycelium_fractal_net import create_rng, simulate_mycelium_field

rng_ctx = create_rng(seed=42)
field, growth_events = simulate_mycelium_field(
    rng_ctx.numpy_rng,
    grid_size=64,
    steps=100,
)
```

### Using the Run Registry

```python
from mycelium_fractal_net import RunRegistry, SimulationConfig

# Create registry (default: runs/ directory)
registry = RunRegistry()

# Start a run
config = SimulationConfig(grid_size=64, steps=100, seed=42)
run = registry.start_run(config, run_type="simulation", seed=42)

# ... execute simulation ...

# Log metrics
registry.log_metrics(run, {
    "loss": 0.123,
    "fractal_dim": 1.58,
    "growth_events": 42,
})

# End the run
registry.end_run(run, status="success")
```

## RNG Control Layer

### Module: `mycelium_fractal_net.infra.rng`

The RNG module provides centralized control over random number generators across numpy, Python's random module, and PyTorch.

#### Key Functions

| Function | Description |
|----------|-------------|
| `create_rng(seed)` | Create an RNGContext with synchronized generators |
| `set_global_seed(seed)` | Set global seed for all libraries |
| `get_numpy_rng(seed)` | Get a numpy Generator with given seed |

#### RNGContext Class

```python
from mycelium_fractal_net.infra.rng import RNGContext, create_rng

# Create context
ctx = create_rng(seed=42)

# Access numpy generator
values = ctx.numpy_rng.random(10)

# Fork for independent operations
forked_ctx = ctx.fork(offset=100)  # seed = 142

# Reset to original state
ctx.reset()

# Restore Python random state before context was created
ctx.restore_original_state()
```

### Global Seeding

For code that uses legacy `np.random.*` or `random.*` APIs:

```python
from mycelium_fractal_net import set_global_seed

set_global_seed(42)
# Now np.random.random() and random.random() are deterministic
```

## Run Registry

### Module: `mycelium_fractal_net.infra.run_registry`

The run registry provides lightweight, file-based tracking of experimental runs.

### Directory Structure

```
runs/
└── 20250530_143022_a1b2c3d4/
    ├── config.json     # Full configuration
    ├── meta.json       # Run metadata
    └── metrics.json    # Logged metrics
```

### meta.json Schema

```json
{
  "run_id": "20250530_143022_a1b2c3d4",
  "run_type": "simulation",
  "timestamp": "2025-05-30T14:30:22.123456",
  "git_commit": "abc1234567890...",
  "command": null,
  "env": "dev",
  "seed": 42,
  "status": "success",
  "end_timestamp": "2025-05-30T14:30:25.654321",
  "duration_seconds": 3.53
}
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MFN_RUN_REGISTRY_ENABLED` | Enable/disable registry | `true` |
| `MFN_RUN_REGISTRY_DIR` | Override runs directory | `runs/` |

### Disabling the Registry

```bash
# Disable for CI or testing
export MFN_RUN_REGISTRY_ENABLED=false
```

Or programmatically:

```python
registry = RunRegistry(enabled=False)
```

## Configuration Serialization

All MFN config types support `to_dict()` and `from_dict()` methods for serialization:

### SimulationConfig

```python
from mycelium_fractal_net import SimulationConfig

config = SimulationConfig(grid_size=64, steps=100, seed=42)

# Serialize to dict
data = config.to_dict()
# {'grid_size': 64, 'steps': 100, 'alpha': 0.18, 'spike_probability': 0.25, ...}

# Deserialize from dict
restored = SimulationConfig.from_dict(data)
```

### FeatureConfig

```python
from mycelium_fractal_net import FeatureConfig

config = FeatureConfig(
    min_box_size=2,
    num_scales=5,
    threshold_low_mv=-60.0,
)

data = config.to_dict()
restored = FeatureConfig.from_dict(data)
```

### DatasetConfig

```python
from mycelium_fractal_net import DatasetConfig
from pathlib import Path

config = DatasetConfig(
    num_samples=100,
    grid_sizes=[32, 64],
    base_seed=42,
    output_path=Path("data/dataset.parquet"),
)

data = config.to_dict()
restored = DatasetConfig.from_dict(data)
```

### ScenarioConfig

```python
from mycelium_fractal_net import ScenarioConfig

config = ScenarioConfig(
    name="my_scenario",
    grid_size=64,
    steps=100,
    base_seed=42,
)

data = config.to_dict()
restored = ScenarioConfig.from_dict(data)
```

## Reproducing a Run

To reproduce a previous run:

1. Load the config from `config.json`:

```python
import json
from mycelium_fractal_net import SimulationConfig, run_mycelium_simulation_with_history

# Load saved config
with open("runs/20250530_143022_a1b2c3d4/config.json") as f:
    config_data = json.load(f)

# Recreate config
config = SimulationConfig.from_dict(config_data)

# Run simulation — will produce identical results
result = run_mycelium_simulation_with_history(config)
```

2. Verify results match the original metrics in `metrics.json`.

## CLI Usage

The main CLI supports `--seed` parameter:

```bash
# Run validation with specific seed
python mycelium_fractal_net_v4_1.py --mode validate --seed 42
```

## Determinism Guarantees

### What is guaranteed:

- Same seed → Same simulation results (field values, growth events)
- Same config → Same feature extraction results
- Same seed + config → Same metrics

### What is NOT guaranteed:

- Cross-platform reproducibility (floating-point differences possible)
- PyTorch CUDA operations (GPU non-determinism by default)
- Different library versions may produce different results

### Enabling Full CUDA Determinism

For PyTorch GPU reproducibility (may impact performance):

```python
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Testing Reproducibility

MFN includes reproducibility tests in `tests/repro/`:

```bash
# Run all reproducibility tests
pytest tests/repro/ -v

# Test determinism
pytest tests/repro/test_determinism_small.py -v

# Test run registry
pytest tests/repro/test_run_registry_basics.py -v
```

## Best Practices

1. **Always specify seeds** — Use explicit seeds in configs for reproducibility
2. **Save configs** — Use RunRegistry or save configs manually
3. **Record git commits** — RunRegistry automatically captures git commit hash
4. **Use RNGContext** — For complex experiments with multiple RNG sources
5. **Document parameters** — Include all relevant parameters in config

## See Also

- [MFN_DATA_MODEL.md](MFN_DATA_MODEL.md) — Canonical data structures
- [MFN_DATA_PIPELINES.md](MFN_DATA_PIPELINES.md) — Dataset generation
- [TECHNICAL_AUDIT.md](TECHNICAL_AUDIT.md) — System health and validation
- [MFN_MATH_MODEL.md](MFN_MATH_MODEL.md) — Mathematical formalization
