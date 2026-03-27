"""Viability kernel — ellipsoidal capture basin approximation.

# APPROXIMATION: ellipsoidal capture basin, not exact Viab_K
# Exact viability kernel computation is NP-hard in general.
# We use Mahalanobis distance with horizon-dependent expansion.

B(S, W) ~ {x : Mahalanobis(x, S) <= 1 + kappa * W}

Read-only: does not modify system state.

Ref: Aubin (1991) Viability Theory, Vasylenko (2026)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from .types import NormSpace

__all__ = ["ViabilityKernel"]


class ViabilityKernel:
    """Ellipsoidal approximation of the capture basin.

    # APPROXIMATION: ellipsoidal capture basin, not exact Viab_K
    """

    def __init__(self, kappa: float = 0.1) -> None:
        self.kappa = kappa

    def contains(self, x: np.ndarray, norm: NormSpace) -> bool:
        """True if x is within the norm ellipsoid."""
        return norm.contains(x)

    def in_capture_basin(
        self,
        x: np.ndarray,
        norm: NormSpace,
        horizon: int = 10,
    ) -> bool:
        """True if x is within the expanded capture basin.

        # APPROXIMATION: ellipsoidal capture basin, not exact Viab_K
        Basin radius expands by kappa * horizon.
        """
        threshold = 1.0 + self.kappa * horizon
        return norm.mahalanobis(x) <= threshold

    def basin_radius(self, norm: NormSpace, horizon: int = 10) -> float:
        """Effective radius of the capture basin."""
        return 1.0 + self.kappa * horizon

    def distance_to_boundary(
        self,
        x: np.ndarray,
        norm: NormSpace,
        horizon: int = 10,
    ) -> float:
        """Signed distance to capture basin boundary. Negative = inside."""
        threshold = 1.0 + self.kappa * horizon
        return norm.mahalanobis(x) - threshold
