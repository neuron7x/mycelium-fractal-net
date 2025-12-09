# Namespace Migration Guide

## Breaking Change: Analytics Module Relocated

**Version**: 4.1.0  
**Date**: 2024-12-09

### Summary

The top-level `analytics` module has been moved into the `mycelium_fractal_net` namespace to eliminate namespace collision risks with external packages.

### Problem

The original codebase had a top-level `analytics/` module that was:
1. Part of the runtime (not dev-only)
2. Imported by core modules
3. Creating a namespace collision risk with external `analytics` packages in the Python ecosystem

This violated the principle of namespace hygiene for published packages.

### Solution

**All analytics functionality has been moved to `mycelium_fractal_net.analytics`**.

### Migration Required

#### Before (Old Code)
```python
from analytics import FeatureConfig, FeatureVector, compute_features
```

#### After (New Code)
```python
from mycelium_fractal_net.analytics import FeatureConfig, FeatureVector, compute_features
```

### Files Updated

The following internal files have been updated with the new import paths:
- `src/mycelium_fractal_net/types/features.py`
- `src/mycelium_fractal_net/experiments/generate_dataset.py`
- `src/mycelium_fractal_net/pipelines/scenarios.py`
- `src/mycelium_fractal_net/analytics/fractal_features.py` (docstrings)

### External Code Impact

If you have external code or scripts that import from `analytics`, you must update them:

```python
# OLD - Will fail
from analytics import FeatureConfig
from analytics.fractal_features import FeatureVector

# NEW - Correct
from mycelium_fractal_net.analytics import FeatureConfig
from mycelium_fractal_net.analytics.fractal_features import FeatureVector
```

### Verification

After upgrading, verify your code works:

```bash
# Install the new version
pip install --upgrade mycelium-fractal-net

# Test import
python -c "from mycelium_fractal_net.analytics import FeatureConfig; print('OK')"
```

### Why This Change Was Necessary

1. **Namespace Collision**: A top-level `analytics` module could conflict with other packages named `analytics` in the Python ecosystem
2. **Best Practice**: Python packages should not claim generic top-level names
3. **Production Safety**: Prevents silent import errors or unexpected behavior when multiple packages share names
4. **PEP 420 Compliance**: Proper namespace packaging structure

### Rollback

If you cannot immediately migrate, pin to the previous version:

```bash
pip install mycelium-fractal-net==4.0.x
```

However, this is not recommended as it contains security issues fixed in 4.1.0.

### Support

For questions or issues with this migration:
1. Review the examples in `examples/` directory
2. Check the test suite in `tests/` for usage patterns
3. Open an issue on GitHub with the `migration` label

### Related Documentation

- `CHANGELOG.md`: Complete list of breaking changes
- `SECURITY.md`: Security improvements in 4.1.0
- `docs/MFN_FEATURE_SCHEMA.md`: Analytics API documentation
