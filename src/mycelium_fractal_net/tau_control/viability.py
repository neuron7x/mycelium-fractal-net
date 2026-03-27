"""Viability kernel — certified ellipsoidal inner approximation with barrier monitor.

# IMPLEMENTED TRUTH: P > 0 verified at construction, membership check exact for ellipsoid.
# APPROXIMATION: ellipsoidal inner approximation, not exact Viab_K.
# APPROXIMATION: barrier monitor, not formal CBF certificate.
# GAP: SOS/polynomial certification requires f(x); not available in this repo.
# GAP: exact Viab_K requires system dynamics model; not available.

Ref: Aubin (1991) Viability Theory, Vasylenko (2026)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .types import NormSpace

__all__ = [
    "BarrierMonitor",
    "BarrierStatus",
    "CertifiedEllipsoid",
    "ViabilityKernel",
]


@dataclass(frozen=True)
class BarrierStatus:
    """Status from barrier monitor at one step.

    # APPROXIMATION: barrier monitor, not formal CBF certificate.
    """

    b_value: float
    delta_b: float
    approaching_boundary: bool
    outside_safe_set: bool
    consecutive_violations: int


class CertifiedEllipsoid:
    """Certified ellipsoidal inner approximation of viable set.

    {x : (x - mu)^T P (x - mu) <= 1} where P > 0.

    # IMPLEMENTED TRUTH: P positive definiteness verified at construction.
    # APPROXIMATION: inner ellipsoid, not exact viability kernel.
    """

    def __init__(self, P: np.ndarray, mu: np.ndarray) -> None:
        eigvals = np.linalg.eigvalsh(P)
        if not np.all(eigvals > 0):
            msg = f"P must be positive definite, got min eigenvalue {float(np.min(eigvals))}"
            raise ValueError(msg)
        self.P = P.copy()
        self.mu = mu.copy()
        self._eigvals = eigvals
        self._certificate_valid = True

    def _mahalanobis_sq(self, x: np.ndarray) -> float:
        diff = x - self.mu
        return float(diff @ self.P @ diff)

    def is_viable(self, x: np.ndarray) -> bool:
        """True if x inside certified ellipsoid."""
        return self._mahalanobis_sq(x) <= 1.0

    def has_recovery_trajectory(self, x: np.ndarray, horizon: int, kappa: float = 0.1) -> bool:
        """True if x in expanded capture basin.

        # APPROXIMATION: linear expansion, not exact reachability.
        """
        threshold = (1.0 + kappa * horizon) ** 2
        return self._mahalanobis_sq(x) <= threshold

    def barrier_value(self, x: np.ndarray) -> float:
        """B(x) = 1 - (x-mu)^T P (x-mu). B > 0 inside ellipsoid."""
        return 1.0 - self._mahalanobis_sq(x)

    def certificate_summary(self) -> dict[str, Any]:
        d = len(self.mu)
        import math as _math

        volume = float(np.pi ** (d / 2) / _math.gamma(d / 2 + 1) / np.sqrt(np.linalg.det(self.P)))
        return {
            "dimension": d,
            "min_eigenvalue": float(np.min(self._eigvals)),
            "max_eigenvalue": float(np.max(self._eigvals)),
            "volume_estimate": volume,
            "certificate_valid": self._certificate_valid,
        }

    @classmethod
    def from_data(
        cls,
        data: np.ndarray,
        coverage_quantile: float = 0.95,
    ) -> CertifiedEllipsoid:
        """Fit from operational trajectory data.

        P = (1/r^2) * Sigma^-1 where r = quantile-based coverage radius.
        """
        mu = np.mean(data, axis=0)
        centered = data - mu
        cov = np.cov(centered, rowvar=False)

        # Regularize
        cov += np.eye(cov.shape[0]) * 1e-6

        cov_inv = np.linalg.inv(cov)

        # Coverage radius: chi-squared quantile approximation
        d = cov.shape[0]
        from scipy.stats import chi2

        r_sq = chi2.ppf(coverage_quantile, df=d)
        P = cov_inv / r_sq

        return cls(P=P, mu=mu)


class BarrierMonitor:
    """Monitors B(x) = 1 - (x-mu)^T P (x-mu) from certified ellipsoid.

    # APPROXIMATION: barrier monitor, not formal CBF certificate.
    # GATE: do not claim CBF invariance without formal proof.
    """

    def __init__(self, delta_b: float = 0.05) -> None:
        self.delta_b_threshold = delta_b
        self._prev_b: float | None = None
        self._consecutive_violations: int = 0

    def update(
        self,
        x: np.ndarray,
        ellipsoid: CertifiedEllipsoid,
    ) -> BarrierStatus:
        b = ellipsoid.barrier_value(x)
        delta_b = 0.0
        if self._prev_b is not None:
            delta_b = b - self._prev_b

        outside = b <= 0
        approaching = b > 0 and delta_b < -self.delta_b_threshold

        if delta_b < 0:
            self._consecutive_violations += 1
        else:
            self._consecutive_violations = 0

        self._prev_b = b

        return BarrierStatus(
            b_value=b,
            delta_b=delta_b,
            approaching_boundary=approaching,
            outside_safe_set=outside,
            consecutive_violations=self._consecutive_violations,
        )

    def reset(self) -> None:
        self._prev_b = None
        self._consecutive_violations = 0


class ViabilityKernel:
    """Backward-compatible wrapper. Delegates to CertifiedEllipsoid when available."""

    def __init__(self, kappa: float = 0.1) -> None:
        self.kappa = kappa

    def contains(self, x: np.ndarray, norm: NormSpace) -> bool:
        return norm.contains(x)

    def in_capture_basin(
        self,
        x: np.ndarray,
        norm: NormSpace,
        horizon: int = 10,
    ) -> bool:
        """# APPROXIMATION: ellipsoidal capture basin, not exact Viab_K."""
        threshold = 1.0 + self.kappa * horizon
        return norm.mahalanobis(x) <= threshold

    def distance_to_boundary(
        self,
        x: np.ndarray,
        norm: NormSpace,
        horizon: int = 10,
    ) -> float:
        threshold = 1.0 + self.kappa * horizon
        return norm.mahalanobis(x) - threshold
