"""Wasserstein geometry as native metric for MFN state space.

Ref: Ito et al. (2025) Phys.Rev.Research 7:033011
     Peyre & Cuturi (2019) DOI:10.1561/2200000073
"""

from __future__ import annotations

import numpy as np

__all__ = ["ot_basin_stability", "wasserstein_distance", "wasserstein_trajectory_speed"]


def _field_to_distribution(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert 2D field to (coords, weights) for OT."""
    N = field.shape[0]
    x = np.arange(N, dtype=np.float64)
    xx, yy = np.meshgrid(x, x)
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)
    w = np.abs(field).ravel().astype(np.float64) + 1e-12
    w /= w.sum()
    return coords, w


def wasserstein_distance(
    field1: np.ndarray,
    field2: np.ndarray,
    method: str = "sliced",
    n_projections: int = 100,
) -> float:
    """Sliced W2 between two 2D fields. ~40ms for N=32."""
    import ot

    c1, a = _field_to_distribution(field1)
    c2, b = _field_to_distribution(field2)
    if method == "sliced":
        return float(ot.sliced_wasserstein_distance(c1, c2, a, b, n_projections))
    M = ot.dist(c1, c2)
    return float(np.sqrt(max(ot.emd2(a, b, M), 0)))


def wasserstein_trajectory_speed(
    history: np.ndarray,
    method: str = "sliced",
    stride: int = 1,
) -> np.ndarray:
    """W2 speed along trajectory — geometric Lyapunov function."""
    T = history.shape[0]
    speeds: list[float] = []
    for t in range(0, T - stride, stride):
        w = wasserstein_distance(history[t], history[t + stride], method=method)
        speeds.append(w)
    return np.array(speeds)


def ot_basin_stability(
    final_fields: list[np.ndarray],
    attractor_fields: list[np.ndarray],
    temperature: float = 1.0,
) -> np.ndarray:
    """Soft basin stability via W2 distance to attractors."""
    n_ic = len(final_fields)
    n_attr = len(attractor_fields)
    dists = np.zeros((n_ic, n_attr))

    for i, ff in enumerate(final_fields):
        for j, af in enumerate(attractor_fields):
            dists[i, j] = wasserstein_distance(ff, af, method="sliced")

    log_scores = -dists / temperature
    log_scores -= log_scores.max(axis=1, keepdims=True)
    scores = np.exp(log_scores)
    return scores / scores.sum(axis=1, keepdims=True)
