"""
Stability Analysis Module — Lyapunov Exponents and Stability Metrics.

This module provides stability analysis tools including Lyapunov exponent
computation from field evolution history and IFS dynamics.

Reference: MFN_MATH_MODEL.md Section 3.3 and Appendix B (Stability Analysis)

Lyapunov Exponent:
    λ = lim_{n→∞} (1/n) Σ ln|det(J_k)|

Interpretation:
    λ < 0: Stable (contractive) dynamics
    λ > 0: Unstable (expansive) dynamics
    
Expected values:
    MFN IFS fractals: λ ≈ -2.1 (stable)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# Re-export from fractal_growth_engine for stability metrics
from .fractal_growth_engine import (
    LYAPUNOV_STABLE_MAX,
    FractalGrowthEngine,
    FractalMetrics,
)


def compute_lyapunov_exponent(
    field_history: NDArray[Any],
    dt: float = 1.0,
) -> float:
    """
    Compute Lyapunov exponent from field evolution history.

    Measures exponential divergence/convergence of trajectories.
    Negative value indicates stable dynamics.

    Reference: MFN_MATH_MODEL.md Section 3.3

    Parameters
    ----------
    field_history : NDArray[Any]
        Array of shape (T, N, N) with field states over time.
    dt : float
        Time step between states.

    Returns
    -------
    float
        Estimated Lyapunov exponent.
        - λ < 0: Stable (contractive) dynamics
        - λ > 0: Unstable (expansive) dynamics
    """
    if len(field_history) < 2:
        return 0.0

    T = len(field_history)
    log_divergence = 0.0
    count = 0

    for t in range(1, T):
        diff = np.abs(field_history[t] - field_history[t - 1])
        norm_diff = np.sqrt(np.sum(diff**2))
        if norm_diff > 1e-10:
            log_divergence += np.log(norm_diff)
            count += 1

    if count == 0:
        return 0.0

    return log_divergence / (count * dt)


def is_stable(lyapunov: float) -> bool:
    """
    Check if Lyapunov exponent indicates stable dynamics.

    Parameters
    ----------
    lyapunov : float
        Lyapunov exponent value.

    Returns
    -------
    bool
        True if λ < 0 (stable/contractive dynamics).
    """
    return lyapunov < LYAPUNOV_STABLE_MAX


def compute_stability_metrics(
    field_history: NDArray[Any],
    dt: float = 1.0,
) -> dict[str, float]:
    """
    Compute comprehensive stability metrics from field history.

    Parameters
    ----------
    field_history : NDArray[Any]
        Array of shape (T, N, N) with field states over time.
    dt : float
        Time step between states.

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - lyapunov_exponent: Estimated Lyapunov exponent
        - is_stable: 1.0 if stable, 0.0 if unstable
        - max_deviation: Maximum field value deviation
        - mean_deviation: Mean field value deviation
        - variance_trend: Trend in variance over time
    """
    lyapunov = compute_lyapunov_exponent(field_history, dt)
    
    metrics = {
        "lyapunov_exponent": lyapunov,
        "is_stable": 1.0 if is_stable(lyapunov) else 0.0,
    }
    
    if len(field_history) >= 2:
        # Compute deviation statistics
        deviations = []
        for t in range(1, len(field_history)):
            diff = np.abs(field_history[t] - field_history[t - 1])
            deviations.append(np.mean(diff))
        
        metrics["max_deviation"] = float(np.max(deviations))
        metrics["mean_deviation"] = float(np.mean(deviations))
        
        # Variance trend (positive = increasing instability)
        variances = [float(np.var(field_history[t])) for t in range(len(field_history))]
        if len(variances) >= 2:
            variance_diff = np.diff(variances)
            metrics["variance_trend"] = float(np.mean(variance_diff))
        else:
            metrics["variance_trend"] = 0.0
    else:
        metrics["max_deviation"] = 0.0
        metrics["mean_deviation"] = 0.0
        metrics["variance_trend"] = 0.0
    
    return metrics


__all__ = [
    # Main functions
    "compute_lyapunov_exponent",
    "is_stable",
    "compute_stability_metrics",
    # Constants
    "LYAPUNOV_STABLE_MAX",
    # Re-exports for related functionality
    "FractalGrowthEngine",
    "FractalMetrics",
]
