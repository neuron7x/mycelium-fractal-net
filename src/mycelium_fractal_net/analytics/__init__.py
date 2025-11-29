"""
Analytics submodule for MyceliumFractalNet.

Provides feature extraction and analysis utilities.
Re-exports FeatureVector for convenient access.
"""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


# Type alias for feature vectors compatible with numpy arrays
FeatureVector = NDArray[np.float64]
"""Type alias for 18-element feature array (see FEATURE_SCHEMA.md)."""

__all__ = ["FeatureVector"]
