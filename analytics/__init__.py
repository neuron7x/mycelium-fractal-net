"""
Analytics module for MyceliumFractalNet.

⚠️ DEPRECATION WARNING ⚠️
This root-level analytics module is DEPRECATED and will be removed in v5.0.0.

Please migrate to the canonical import:
    from mycelium_fractal_net.analytics import compute_fractal_features, FeatureVector

See docs/MIGRATION_GUIDE.md for detailed migration instructions.

Provides feature extraction utilities for fractal analysis:
- Box-counting fractal dimension
- Basic field statistics
- Temporal dynamics features
- Structural/connectivity features
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Root-level 'analytics' module is deprecated and will be removed in v5.0.0. "
    "Please use 'from mycelium_fractal_net.analytics import ...' instead. "
    "See docs/MIGRATION_GUIDE.md for details.",
    DeprecationWarning,
    stacklevel=2,
)

from .fractal_features import (
    FEATURE_COUNT,
    FeatureConfig,
    FeatureVector,
    compute_basic_stats,
    compute_features,
    compute_fractal_features,
    compute_structural_features,
    compute_temporal_features,
)

__all__ = [
    "FEATURE_COUNT",
    "FeatureConfig",
    "FeatureVector",
    "compute_features",
    "compute_fractal_features",
    "compute_basic_stats",
    "compute_temporal_features",
    "compute_structural_features",
]
