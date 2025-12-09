"""
Feature types for MyceliumFractalNet.

Defines the canonical 18-feature vector structure as specified in
MFN_FEATURE_SCHEMA.md. All feature names, order, and ranges are aligned
with the documentation.

Reference:
    - docs/MFN_FEATURE_SCHEMA.md — Complete feature definitions
    - docs/MFN_DATA_PIPELINES.md — Feature column schema
"""

from __future__ import annotations

# Re-export FeatureVector from analytics submodule (single source of truth)
# This is the canonical dataclass with individual feature fields (D_box, V_mean, etc.)
from mycelium_fractal_net.analytics.fractal_features import (
    FEATURE_COUNT,
    FeatureVector,
    compute_features,
    validate_feature_ranges,
)

# Canonical feature names in fixed order (per MFN_FEATURE_SCHEMA.md Section 3.2)
FEATURE_NAMES: list[str] = [
    "D_box",           # 1. Box-counting dimension
    "D_r2",            # 2. R² of dimension fit
    "V_min",           # 3. Minimum field value (mV)
    "V_max",           # 4. Maximum field value (mV)
    "V_mean",          # 5. Mean field value (mV)
    "V_std",           # 6. Field standard deviation (mV)
    "V_skew",          # 7. Field skewness
    "V_kurt",          # 8. Field kurtosis (excess)
    "dV_mean",         # 9. Mean rate of change (mV/step)
    "dV_max",          # 10. Max rate of change (mV/step)
    "T_stable",        # 11. Steps to quasi-stationary
    "E_trend",         # 12. Energy trend slope (mV²/step)
    "f_active",        # 13. Active cell fraction
    "N_clusters_low",  # 14. Clusters at -60mV
    "N_clusters_med",  # 15. Clusters at -50mV
    "N_clusters_high", # 16. Clusters at -40mV
    "max_cluster_size",# 17. Largest cluster size (cells)
    "cluster_size_std",# 18. Cluster size std dev (cells)
]

# Verify consistency with analytics module
assert len(FEATURE_NAMES) == FEATURE_COUNT, (
    f"FEATURE_NAMES length ({len(FEATURE_NAMES)}) must match "
    f"FEATURE_COUNT ({FEATURE_COUNT})"
)

__all__ = [
    "FeatureVector",
    "FEATURE_NAMES",
    "FEATURE_COUNT",
    "compute_features",
    "validate_feature_ranges",
]
