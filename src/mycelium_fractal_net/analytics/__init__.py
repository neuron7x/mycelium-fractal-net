"""
Analytics submodule for MyceliumFractalNet.

Provides feature extraction and analysis utilities for fractal analysis.

Main API:
- compute_fractal_features(result: SimulationResult) -> FeatureVector
- compute_box_counting_dimension(field, ...) -> float
- compute_basic_stats(field) -> Dict[str, float]

Reference: docs/MFN_FEATURE_SCHEMA.md
"""

import numpy as np
from numpy.typing import NDArray

from .fractal_features import (
    FeatureVector,
    compute_basic_stats,
    compute_box_counting_dimension,
    compute_fractal_features,
)
from .insight_architect import FractalInsightArchitect, Insight, InsufficientDataError


# Legacy type alias for compatibility
FeatureArray = NDArray[np.float64]
"""Type alias for 18-element feature array (see MFN_FEATURE_SCHEMA.md)."""

__all__ = [
    "FeatureVector",
    "FeatureArray",
    "compute_fractal_features",
    "compute_box_counting_dimension",
    "compute_basic_stats",
    "FractalInsightArchitect",
    "Insight",
    "InsufficientDataError",
]
