# Migration Guide — MyceliumFractalNet v4.1

**Last Updated:** 2025-12-05  
**Target Audience:** Developers using MFN in their projects

---

## Overview

MyceliumFractalNet v4.1 has migrated to a `src/` layout for better package organization and distribution. This guide helps you update your code to use the new import paths.

## Summary of Changes

### What Changed

The root-level `analytics/` and `experiments/` modules have been **deprecated** in favor of the canonical package structure under `src/mycelium_fractal_net/`.

**Old structure (deprecated):**
```
mycelium-fractal-net/
├── analytics/          # ⚠️ DEPRECATED
│   └── fractal_features.py
├── experiments/        # ⚠️ DEPRECATED
│   └── generate_dataset.py
└── src/
    └── mycelium_fractal_net/
        ├── analytics/      # ✅ USE THIS
        └── experiments/    # ✅ USE THIS
```

**New structure (canonical):**
```
mycelium-fractal-net/
└── src/
    └── mycelium_fractal_net/
        ├── analytics/      # ✅ Canonical location
        ├── experiments/    # ✅ Canonical location
        ├── core/
        ├── crypto/
        └── ...
```

### Why This Change?

1. **PEP 517/518 compliance:** Modern Python packaging standards recommend `src/` layout
2. **Import clarity:** Prevents accidental imports of root-level modules during development
3. **Distribution:** Ensures only canonical package is distributed via PyPI
4. **Testing:** Encourages testing against installed package, not source tree

---

## Migration Instructions

### For Analytics Module

#### Old Import (Deprecated)
```python
from analytics import compute_features, FeatureConfig, FeatureVector
from analytics.fractal_features import compute_basic_stats
```

#### New Import (Recommended)
```python
from mycelium_fractal_net.analytics import compute_fractal_features, FeatureVector
from mycelium_fractal_net.analytics import compute_basic_stats
```

#### Alternative (via main package)
```python
from mycelium_fractal_net import compute_fractal_features, FeatureVector
```

### For Experiments Module

#### Old Import (Deprecated)
```python
from experiments import generate_dataset, SweepConfig
from experiments.generate_dataset import DatasetConfig
```

#### New Import (Recommended)
```python
from mycelium_fractal_net.experiments import generate_dataset
from mycelium_fractal_net.pipelines import get_preset_config
```

#### Alternative (via main package)
```python
from mycelium_fractal_net import run_scenario, get_preset_config
```

---

## API Changes

### Analytics Module

The analytics API has been streamlined. Here's the mapping:

| Old API (analytics) | New API (mycelium_fractal_net.analytics) | Notes |
|---------------------|-------------------------------------------|-------|
| `compute_features(field_history, config)` | `compute_fractal_features(simulation_result)` | Now takes SimulationResult instead of raw array |
| `FeatureConfig()` | Configuration passed via SimulationConfig | Simplified API |
| `FeatureVector` | `FeatureVector` | Same dataclass |
| `compute_fractal_features()` | `compute_fractal_features()` | Same function |
| `compute_basic_stats()` | `compute_basic_stats()` | Same function |
| `compute_temporal_features()` | Integrated into `compute_fractal_features()` | No separate function |
| `compute_structural_features()` | Integrated into `compute_fractal_features()` | No separate function |

### Experiments Module

The experiments module has been restructured into pipelines:

| Old API (experiments) | New API (mycelium_fractal_net) | Notes |
|-----------------------|----------------------------------|-------|
| `generate_dataset(sweep_config)` | `run_scenario(scenario_config)` | New scenario-based API |
| `SweepConfig` | `ScenarioConfig` | Renamed for clarity |
| `DatasetConfig` | `DatasetConfig` | Same, now in types module |

---

## Step-by-Step Migration

### Step 1: Update Imports

Replace all imports from `analytics` and `experiments` with `mycelium_fractal_net.analytics` and `mycelium_fractal_net.experiments`.

**Before:**
```python
from analytics import compute_features, FeatureVector
from experiments import generate_dataset
```

**After:**
```python
from mycelium_fractal_net.analytics import compute_fractal_features, FeatureVector
from mycelium_fractal_net.experiments import generate_dataset
```

### Step 2: Update API Calls (if needed)

If you were using `compute_features` with raw field_history:

**Before:**
```python
from analytics import compute_features, FeatureConfig

config = FeatureConfig()
features = compute_features(field_history, config)
```

**After:**
```python
from mycelium_fractal_net import run_mycelium_simulation_with_history, compute_fractal_features

result = run_mycelium_simulation_with_history(simulation_config)
features = compute_fractal_features(result)
```

### Step 3: Test Your Code

Run your test suite to ensure everything works:
```bash
pytest tests/
```

---

## Backward Compatibility

### Deprecation Timeline

- **v4.1.0 (current):** Root-level `analytics/` and `experiments/` are **DEPRECATED** but still functional
  - Imports will work but may show deprecation warnings in future versions
  - All documentation updated to use canonical imports
  
- **v4.2.0 (planned):** Deprecation warnings added
  - Importing from root-level modules will show warnings
  - All functionality preserved
  
- **v5.0.0 (future):** Root-level modules **REMOVED**
  - Only `mycelium_fractal_net.*` imports will work
  - Breaking change, requires migration

### How to Prepare

1. Update your imports now using this guide
2. Test thoroughly with v4.1.0
3. Plan migration before v5.0.0 release

---

## Common Issues

### Issue 1: ImportError after migration

**Error:**
```python
ImportError: cannot import name 'compute_features' from 'mycelium_fractal_net.analytics'
```

**Solution:**
`compute_features` was renamed to `compute_fractal_features`. Update your import:
```python
from mycelium_fractal_net.analytics import compute_fractal_features
```

### Issue 2: Different feature extraction results

**Problem:**
Feature values differ slightly after migration.

**Explanation:**
The new API uses `SimulationResult` which includes additional metadata. The computation is identical, but the input structure is different.

**Solution:**
Ensure you're passing the full `SimulationResult` from `run_mycelium_simulation_with_history()`, not just the field array.

### Issue 3: Missing FeatureConfig

**Error:**
```python
NameError: name 'FeatureConfig' is not defined
```

**Solution:**
`FeatureConfig` is no longer needed as a separate parameter. Configuration is now passed via `SimulationConfig`:
```python
from mycelium_fractal_net import make_simulation_config_default

config = make_simulation_config_default()
result = run_mycelium_simulation_with_history(config)
features = compute_fractal_features(result)
```

---

## Examples

### Example 1: Basic Feature Extraction

**Before (v4.0):**
```python
from analytics import compute_features, FeatureConfig
import numpy as np

field_history = np.random.randn(100, 64, 64)
config = FeatureConfig()
features = compute_features(field_history, config)
print(features.D_box)
```

**After (v4.1):**
```python
from mycelium_fractal_net import (
    run_mycelium_simulation_with_history,
    make_simulation_config_default,
    compute_fractal_features,
)

config = make_simulation_config_default()
result = run_mycelium_simulation_with_history(config)
features = compute_fractal_features(result)
print(features.D_box)
```

### Example 2: Dataset Generation

**Before (v4.0):**
```python
from experiments import generate_dataset, SweepConfig

config = SweepConfig(
    grid_sizes=[32, 64],
    step_counts=[50, 100],
    alpha_values=[0.1, 0.2],
)
dataset = generate_dataset(config)
```

**After (v4.1):**
```python
from mycelium_fractal_net import run_scenario, get_preset_config

config = get_preset_config("medium")
dataset = run_scenario(config)
```

### Example 3: Custom Feature Computation

**Before (v4.0):**
```python
from analytics.fractal_features import compute_basic_stats, compute_fractal_features

stats = compute_basic_stats(field)
fractal_dim = compute_fractal_features(binary_field)
```

**After (v4.1):**
```python
from mycelium_fractal_net.analytics import compute_basic_stats, compute_box_counting_dimension

stats = compute_basic_stats(field)
fractal_dim = compute_box_counting_dimension(binary_field)
```

---

## Testing Your Migration

### Unit Tests

Update your test imports and assertions:

**Before:**
```python
from analytics import compute_features

def test_feature_extraction():
    features = compute_features(field_history, config)
    assert features.D_box > 0
```

**After:**
```python
from mycelium_fractal_net import compute_fractal_features

def test_feature_extraction():
    features = compute_fractal_features(simulation_result)
    assert features.D_box > 0
```

### Integration Tests

Ensure your end-to-end tests use the new API:

```python
from mycelium_fractal_net import (
    make_simulation_config_default,
    run_mycelium_simulation_with_history,
    compute_fractal_features,
)

def test_full_pipeline():
    config = make_simulation_config_default()
    result = run_mycelium_simulation_with_history(config)
    features = compute_fractal_features(result)
    
    assert features.D_box > 1.0
    assert features.V_mean < 0
    assert 0 < features.f_active < 1
```

---

## Getting Help

If you encounter issues during migration:

1. **Check Documentation:** [docs/MFN_CODE_STRUCTURE.md](MFN_CODE_STRUCTURE.md)
2. **Review Examples:** [examples/](../examples/)
3. **Open Issue:** [GitHub Issues](https://github.com/neuron7x/mycelium-fractal-net/issues)

---

## Summary Checklist

- [ ] Replace all `from analytics import` with `from mycelium_fractal_net.analytics import`
- [ ] Replace all `from experiments import` with `from mycelium_fractal_net.experiments import` or use pipelines API
- [ ] Update `compute_features()` calls to `compute_fractal_features()`
- [ ] Remove `FeatureConfig` instantiation (use `SimulationConfig` instead)
- [ ] Update test imports and assertions
- [ ] Run full test suite to verify migration
- [ ] Update CI/CD scripts if needed

---

**Migration Status:** ✅ Recommended for all projects  
**Support:** Available through v4.x series  
**Deadline:** Complete before v5.0.0 release

---

**Questions?** Open an issue on [GitHub](https://github.com/neuron7x/mycelium-fractal-net/issues) or check the [troubleshooting guide](TROUBLESHOOTING.md).
