"""
Analytics submodule for MyceliumFractalNet.

Provides feature extraction and analysis utilities for fractal analysis.

Main API:
- compute_features(field_history, config) -> FeatureVector
- compute_fractal_features(...) -> float
- compute_basic_stats(field) -> Dict[str, float]
- compute_temporal_features(field_history) -> Dict[str, float]
- compute_structural_features(field, thresholds) -> Dict[str, float]

Reference: docs/MFN_FEATURE_SCHEMA.md
"""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .fractal_features import (
    FEATURE_COUNT,
    FeatureConfig,
    FeatureVector,
    compute_basic_stats,
    compute_box_counting_dimension,
    compute_features,
    compute_fractal_features,
    compute_structural_features,
    compute_temporal_features,
)

if TYPE_CHECKING:
    pass


# Legacy type alias for compatibility
FeatureArray = NDArray[np.float64]
"""Type alias for 18-element feature array (see MFN_FEATURE_SCHEMA.md)."""

__all__ = [
    "FEATURE_COUNT",
    "FeatureConfig",
    "FeatureVector",
    "FeatureArray",
    "compute_box_counting_dimension",
    "compute_features",
    "compute_fractal_features",
    "compute_basic_stats",
    "compute_temporal_features",
    "compute_structural_features",
]
