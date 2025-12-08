"""
Analytics module for MyceliumFractalNet.

Provides feature extraction utilities for fractal analysis:
- Box-counting fractal dimension
- Basic field statistics
- Temporal dynamics features
- Structural/connectivity features
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .fractal_features import (
    FEATURE_COUNT,
    FeatureConfig,
    FeatureVector,
    compute_basic_stats,
    compute_box_counting_dimension,
    compute_features,
    compute_fractal_features as _compute_fractal_features_low_level,
    compute_structural_features,
    compute_temporal_features,
)

if TYPE_CHECKING:
    from ..core.types import SimulationResult


def compute_fractal_features(
    result,  # SimulationResult or ndarray
    config: FeatureConfig | None = None,
) -> FeatureVector:
    """
    High-level wrapper to extract all 18 fractal features.

    This is the main user-facing API for feature extraction.

    Parameters
    ----------
    result : SimulationResult or ndarray
        Simulation result containing field and optionally history,
        or directly a field array (2D or 3D with history).
    config : FeatureConfig | None
        Configuration for feature extraction. If None, uses defaults.

    Returns
    -------
    FeatureVector
        Complete feature vector with all 18 features.

    Examples
    --------
    >>> from mycelium_fractal_net import run_mycelium_simulation, SimulationConfig
    >>> from mycelium_fractal_net.analytics import compute_fractal_features
    >>> result = run_mycelium_simulation(SimulationConfig(steps=100))
    >>> features = compute_fractal_features(result)
    >>> print(features['D_box'], features['V_mean'])
    """
    import numpy as np

    # Handle both SimulationResult and raw ndarray inputs
    if isinstance(result, np.ndarray):
        # Direct array input
        field_snapshots = result
    elif hasattr(result, 'has_history'):
        # SimulationResult input
        if result.has_history and result.history is not None:
            field_snapshots = result.history
        else:
            field_snapshots = result.field
    else:
        raise TypeError(
            f"Expected SimulationResult or ndarray, got {type(result)}"
        )

    return compute_features(field_snapshots, config=config)


__all__ = [
    "FEATURE_COUNT",
    "FeatureConfig",
    "FeatureVector",
    "compute_features",
    "compute_fractal_features",
    "compute_basic_stats",
    "compute_box_counting_dimension",
    "compute_temporal_features",
    "compute_structural_features",
]
