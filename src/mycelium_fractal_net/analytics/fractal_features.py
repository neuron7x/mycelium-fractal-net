"""
Fractal Feature Extraction for SimulationResult.

Provides integration between the core simulation results and fractal analytics.
Implements the API specified in MFN_FEATURE_SCHEMA.md.

Usage:
    >>> from mycelium_fractal_net import run_mycelium_simulation, SimulationConfig
    >>> from mycelium_fractal_net.analytics.fractal_features import compute_fractal_features
    >>> result = run_mycelium_simulation(SimulationConfig(steps=100))
    >>> features = compute_fractal_features(result)
    >>> print(features.values)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
from numpy.typing import NDArray

# Import the core feature computation from the analytics module
from analytics.fractal_features import FeatureConfig
from analytics.fractal_features import FeatureVector as AnalyticsFeatureVector
from analytics.fractal_features import (
    compute_basic_stats as _compute_basic_stats,
)
from analytics.fractal_features import (
    compute_features as _compute_features,
)
from analytics.fractal_features import (
    compute_fractal_features as _compute_fractal_dim,
)

if TYPE_CHECKING:
    from mycelium_fractal_net.core.types import SimulationResult

__all__ = [
    "FeatureVector",
    "compute_fractal_features",
    "compute_box_counting_dimension",
    "compute_basic_stats",
]


@dataclass
class FeatureVector:
    """
    Structured feature vector from fractal analysis.

    Contains all features defined in MFN_FEATURE_SCHEMA.md.
    Values are stored as a dictionary for easy access and serialization.

    Attributes
    ----------
    values : Dict[str, float]
        Dictionary mapping feature names to their values.
        All 18 features from MFN_FEATURE_SCHEMA.md are included.

    Examples
    --------
    >>> fv = FeatureVector(values={"D_box": 1.65, "V_mean": -70.0, ...})
    >>> fv.values["D_box"]
    1.65
    >>> fv.to_array()
    array([1.65, 0.95, -80.0, ...])
    """

    values: Dict[str, float] = field(default_factory=dict)

    # Feature names in canonical order (from MFN_FEATURE_SCHEMA.md)
    _FEATURE_NAMES: tuple[str, ...] = (
        "D_box",
        "D_r2",
        "V_min",
        "V_max",
        "V_mean",
        "V_std",
        "V_skew",
        "V_kurt",
        "dV_mean",
        "dV_max",
        "T_stable",
        "E_trend",
        "f_active",
        "N_clusters_low",
        "N_clusters_med",
        "N_clusters_high",
        "max_cluster_size",
        "cluster_size_std",
    )

    @classmethod
    def feature_names(cls) -> tuple[str, ...]:
        """Return canonical feature names in order."""
        return cls._FEATURE_NAMES

    def to_array(self) -> NDArray[np.float64]:
        """
        Convert to numpy array in canonical order.

        Returns
        -------
        NDArray[np.float64]
            Array of shape (18,) with features in order defined by MFN_FEATURE_SCHEMA.md.
        """
        return np.array(
            [float(self.values.get(name, 0.0)) for name in self._FEATURE_NAMES],
            dtype=np.float64,
        )

    @classmethod
    def from_analytics_vector(cls, av: AnalyticsFeatureVector) -> "FeatureVector":
        """
        Create FeatureVector from analytics module FeatureVector.

        Parameters
        ----------
        av : AnalyticsFeatureVector
            FeatureVector from the analytics module.

        Returns
        -------
        FeatureVector
            New FeatureVector with values from the analytics vector.
        """
        return cls(values=av.to_dict())

    def __contains__(self, key: str) -> bool:
        """Check if feature exists."""
        return key in self.values

    def __getitem__(self, key: str) -> float:
        """Get feature value by name."""
        return self.values[key]


def compute_box_counting_dimension(
    field: NDArray[np.floating[Any]],
    *,
    num_scales: int = 8,
    threshold: float = -0.060,
) -> float:
    """
    Compute box-counting fractal dimension for a 2D field.

    Uses the box-counting algorithm to estimate the fractal dimension D
    of the active region in the field. The active region is defined as
    cells with values above the threshold.

    Algorithm:
    1. Binarize field at threshold
    2. Count occupied boxes at multiple scales (box sizes)
    3. Fit log(N) vs log(1/ε) to estimate D

    For a fractal set: N(ε) ~ ε^(-D)
    where N is the number of occupied boxes and ε is the box size.

    Parameters
    ----------
    field : NDArray[np.floating]
        2D field array. Values should be in Volts (biological range: ~-0.095 to 0.040).
        Must be a square array.
    num_scales : int, optional
        Number of scales (box sizes) to use for regression. Default is 8.
        More scales provide better estimates but require larger fields.
    threshold : float, optional
        Threshold for binarization in Volts. Default is -0.060 V (-60 mV).
        Cells above this threshold are considered "active".

    Returns
    -------
    float
        Estimated fractal dimension D.
        - D ≈ 0 for empty/sparse fields
        - D ≈ 1 for line-like structures
        - D ∈ [1.4, 1.9] for biological mycelium patterns
        - D ≈ 2 for filled regions

    Raises
    ------
    ValueError
        If field is not 2D or not square.

    Notes
    -----
    Implementation details are described in docs/MFN_FEATURE_SCHEMA.md.

    Examples
    --------
    >>> import numpy as np
    >>> field = np.random.randn(64, 64) * 0.01 - 0.070
    >>> D = compute_box_counting_dimension(field)
    >>> print(f"Fractal dimension: {D:.3f}")
    """
    if field.ndim != 2:
        raise ValueError(f"field must be 2D, got {field.ndim}D")
    if field.shape[0] != field.shape[1]:
        raise ValueError(f"field must be square, got shape {field.shape}")

    # Convert threshold from Volts (user API) to mV (internal FeatureConfig expects mV)
    threshold_mv = threshold * 1000.0
    config = FeatureConfig(num_scales=num_scales, threshold_low_mv=threshold_mv)

    D_box, _ = _compute_fractal_dim(field, config)
    return D_box


def compute_basic_stats(field: NDArray[np.floating[Any]]) -> Dict[str, float]:
    """
    Compute basic statistics for a 2D field.

    Calculates min, max, mean, and std of the field values.
    All outputs are in millivolts (mV) for interpretability.

    Parameters
    ----------
    field : NDArray[np.floating]
        2D field array. Values should be in Volts.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        - "min": Minimum value in mV
        - "max": Maximum value in mV
        - "mean": Mean value in mV
        - "std": Standard deviation in mV

    Examples
    --------
    >>> import numpy as np
    >>> field = np.full((32, 32), -0.070)  # -70 mV constant field
    >>> stats = compute_basic_stats(field)
    >>> print(stats["mean"])  # Should be close to -70.0
    """
    V_min, V_max, V_mean, V_std, _, _ = _compute_basic_stats(field)
    return {
        "min": V_min,
        "max": V_max,
        "mean": V_mean,
        "std": V_std,
    }


def compute_fractal_features(result: "SimulationResult") -> FeatureVector:
    """
    Compute complete feature vector from SimulationResult.

    Extracts all 18 features defined in MFN_FEATURE_SCHEMA.md from
    the simulation result. Uses the final field state for static features
    and the full history (if available) for temporal features.

    This function does not modify the input result.

    Parameters
    ----------
    result : SimulationResult
        Simulation output containing at minimum the final field.
        If history is available, temporal features will be computed;
        otherwise temporal features will be set to defaults.

    Returns
    -------
    FeatureVector
        Complete feature vector with all 18 features:
        - Fractal: D_box, D_r2
        - Basic stats: V_min, V_max, V_mean, V_std, V_skew, V_kurt
        - Temporal: dV_mean, dV_max, T_stable, E_trend
        - Structural: f_active, N_clusters_low/med/high, max_cluster_size, cluster_size_std

    Notes
    -----
    - All voltage values are converted to mV in the output
    - NaN/Inf values are not expected (controlled by numerical core)
    - For valid results, D_box should be in [0, 2.5], biological range [1.4, 1.9]

    Examples
    --------
    >>> from mycelium_fractal_net import run_mycelium_simulation, SimulationConfig
    >>> from mycelium_fractal_net.analytics.fractal_features import compute_fractal_features
    >>> result = run_mycelium_simulation(SimulationConfig(steps=50, seed=42))
    >>> features = compute_fractal_features(result)
    >>> print(f"D_box: {features.values['D_box']:.3f}")
    >>> print(f"V_mean: {features.values['V_mean']:.1f} mV")

    See Also
    --------
    MFN_FEATURE_SCHEMA.md : Complete feature specification
    compute_box_counting_dimension : Box-counting algorithm details
    """
    # Import here to avoid circular imports
    from mycelium_fractal_net.core.types import SimulationResult as SR

    # Validate input type
    if not isinstance(result, SR):
        raise TypeError(
            f"Expected SimulationResult, got {type(result).__name__}. "
            "Use analytics.compute_features() for raw numpy arrays."
        )

    # Determine input data shape
    if result.has_history and result.history is not None:
        # Use full history for temporal features
        field_data = result.history
    else:
        # Single snapshot - temporal features will be defaults
        field_data = result.field

    # Compute features using the analytics module
    analytics_fv = _compute_features(field_data)

    # Convert to our FeatureVector format
    return FeatureVector.from_analytics_vector(analytics_fv)
