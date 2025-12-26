"""
Stability Analysis Module — Lyapunov Exponents and Metrics.

This module provides the public API for dynamical stability analysis,
including Lyapunov exponent computation and stability metrics.

Conceptual domain: Dynamical systems, stability analysis

Reference:
    - docs/MFN_MATH_MODEL.md Section 3.3 (Lyapunov Exponent)
    - docs/ARCHITECTURE.md Section 3 (Fractal Analysis)

Mathematical Model:
    Lyapunov exponent:
        λ = lim_{n→∞} (1/n) Σ_{k=1}^{n} ln|det(J_k)|

    Interpretation:
        λ < 0: Stable (contractive) dynamics
        λ > 0: Unstable (expansive) dynamics

    Expected value for MFN IFS: λ ≈ -2.1 (stable)

Example:
    >>> import numpy as np
    >>> from mycelium_fractal_net.core.stability import compute_lyapunov_exponent
    >>> # Create field history (T timesteps, N×N grid)
    >>> history = np.random.randn(100, 64, 64) * 0.01
    >>> lyapunov = compute_lyapunov_exponent(history)
    >>> isinstance(lyapunov, float)
    True
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# Re-export the original implementation from model.py for backward compatibility
from ..model import compute_lyapunov_exponent

__all__ = [
    # Functions
    "compute_lyapunov_exponent",
    "is_stable",
    "compute_stability_metrics",
]


def is_stable(lyapunov_exponent: float, threshold: float = 0.0) -> bool:
    """
    Check if system is stable based on Lyapunov exponent.

    Parameters
    ----------
    lyapunov_exponent : float
        Computed Lyapunov exponent.
    threshold : float, optional
        Stability threshold, default 0.0.

    Returns
    -------
    bool
        True if stable (λ < threshold), False otherwise.

    Example
    -------
    >>> is_stable(-2.1)
    True
    >>> is_stable(0.5)
    False
    """
    return lyapunov_exponent < threshold


def compute_stability_metrics(
    field_history: NDArray[Any],
    dt: float = 1.0,
) -> dict[str, float]:
    """
    Compute comprehensive stability metrics from field evolution.

    Parameters
    ----------
    field_history : NDArray
        Array of shape (T, N, N) with field states over time.
    dt : float, optional
        Time step between states, default 1.0.

    Returns
    -------
    dict
        Dictionary containing:
        - lyapunov_exponent: Main stability indicator
        - is_stable: Boolean (as 1.0 or 0.0)
        - mean_change_rate: Average rate of change
        - max_change_rate: Maximum instantaneous change
        - final_std: Standard deviation of final state

    Example
    -------
    >>> history = np.random.randn(50, 32, 32) * 0.01
    >>> metrics = compute_stability_metrics(history)
    >>> 'lyapunov_exponent' in metrics
    True
    """
    if dt <= 0:
        raise ValueError("dt must be positive for stability metrics")

    lyapunov = compute_lyapunov_exponent(field_history, dt)

    # Compute change rates
    if len(field_history) >= 2:
        changes = np.abs(np.diff(field_history, axis=0))
        mean_change = float(np.mean(changes))
        max_change = float(np.max(changes))
    else:
        mean_change = 0.0
        max_change = 0.0

    # Final state statistics
    final_std = float(np.std(field_history[-1])) if len(field_history) > 0 else 0.0

    return {
        "lyapunov_exponent": lyapunov,
        "is_stable": 1.0 if is_stable(lyapunov) else 0.0,
        "mean_change_rate": mean_change / dt,
        "max_change_rate": max_change / dt,
        "final_std": final_std,
    }
