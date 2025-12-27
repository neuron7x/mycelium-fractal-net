"""
Analytics submodule for MyceliumFractalNet.

Provides feature extraction and analysis utilities for fractal analysis.
Reference: docs/MFN_FEATURE_SCHEMA.md
"""

import numpy as np
from numpy.typing import NDArray

from .fractal_features import (
    FeatureConfig,
    FeatureVector,
    compute_basic_stats,
    compute_box_counting_dimension,
    compute_features,
    compute_features_from_result,
    compute_fractal_features,
    compute_structural_features,
    compute_temporal_features,
)
from .insight_architect import FractalInsightArchitect, Insight, InsufficientDataError

# Legacy type alias for compatibility
FeatureArray = NDArray[np.float64]
"""Type alias for 18-element feature array (see MFN_FEATURE_SCHEMA.md)."""

__all__ = [
    "FeatureVector",
    "FeatureConfig",
    "FeatureArray",
    "compute_box_counting_dimension",
    "compute_fractal_features",
    "compute_basic_stats",
    "compute_structural_features",
    "compute_temporal_features",
    "compute_features",
    "compute_features_from_result",
    "FractalInsightArchitect",
    "Insight",
    "InsufficientDataError",
]
