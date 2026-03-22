"""Grid operations for reaction-diffusion simulations."""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from mycelium_fractal_net.core.exceptions import NumericalInstabilityError

try:  # optional acceleration contour
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


class BoundaryCondition(Enum):
    PERIODIC = "periodic"
    NEUMANN = "neumann"
    DIRICHLET = "dirichlet"


def _use_accel(use_accel: bool | None) -> bool:
    if use_accel is not None:
        return bool(use_accel)
    return os.getenv("MFN_ENABLE_ACCEL_LAPLACIAN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _laplacian_numpy_periodic(field: NDArray[np.floating]) -> NDArray[np.floating]:
    up = np.roll(field, 1, axis=0)
    down = np.roll(field, -1, axis=0)
    left = np.roll(field, 1, axis=1)
    right = np.roll(field, -1, axis=1)
    return cast(NDArray[np.floating[Any]], up + down + left + right - 4.0 * field)


def _laplacian_numpy_neumann(field: NDArray[np.floating]) -> NDArray[np.floating]:
    up = np.empty_like(field)
    up[1:, :] = field[:-1, :]
    up[0, :] = field[0, :]

    down = np.empty_like(field)
    down[:-1, :] = field[1:, :]
    down[-1, :] = field[-1, :]

    left = np.empty_like(field)
    left[:, 1:] = field[:, :-1]
    left[:, 0] = field[:, 0]

    right = np.empty_like(field)
    right[:, :-1] = field[:, 1:]
    right[:, -1] = field[:, -1]
    return cast(NDArray[np.floating[Any]], up + down + left + right - 4.0 * field)


def _laplacian_numpy_dirichlet(field: NDArray[np.floating]) -> NDArray[np.floating]:
    up = np.pad(field[:-1, :], ((1, 0), (0, 0)), mode="constant", constant_values=0)
    down = np.pad(field[1:, :], ((0, 1), (0, 0)), mode="constant", constant_values=0)
    left = np.pad(field[:, :-1], ((0, 0), (1, 0)), mode="constant", constant_values=0)
    right = np.pad(field[:, 1:], ((0, 0), (0, 1)), mode="constant", constant_values=0)
    return cast(NDArray[np.floating[Any]], up + down + left + right - 4.0 * field)


if njit is not None:

    @njit(cache=True)
    def _laplacian_periodic_jit(field):
        rows, cols = field.shape
        out = np.empty_like(field)
        for i in range(rows):
            up = (i - 1) % rows
            down = (i + 1) % rows
            for j in range(cols):
                left = (j - 1) % cols
                right = (j + 1) % cols
                out[i, j] = (
                    field[up, j]
                    + field[down, j]
                    + field[i, left]
                    + field[i, right]
                    - 4.0 * field[i, j]
                )
        return out

    @njit(cache=True)
    def _laplacian_neumann_jit(field):
        rows, cols = field.shape
        out = np.empty_like(field)
        for i in range(rows):
            up = i if i == 0 else i - 1
            down = i if i == rows - 1 else i + 1
            for j in range(cols):
                left = j if j == 0 else j - 1
                right = j if j == cols - 1 else j + 1
                out[i, j] = (
                    field[up, j]
                    + field[down, j]
                    + field[i, left]
                    + field[i, right]
                    - 4.0 * field[i, j]
                )
        return out

    @njit(cache=True)
    def _laplacian_dirichlet_jit(field):
        rows, cols = field.shape
        out = np.empty_like(field)
        for i in range(rows):
            for j in range(cols):
                up = field[i - 1, j] if i > 0 else 0.0
                down = field[i + 1, j] if i < rows - 1 else 0.0
                left = field[i, j - 1] if j > 0 else 0.0
                right = field[i, j + 1] if j < cols - 1 else 0.0
                out[i, j] = up + down + left + right - 4.0 * field[i, j]
        return out

else:  # pragma: no cover
    _laplacian_periodic_jit = None
    _laplacian_neumann_jit = None
    _laplacian_dirichlet_jit = None


def laplacian_backend(use_accel: bool | None = None) -> str:
    if _use_accel(use_accel) and njit is not None:
        return "numba-jit"
    return "numpy-reference"


def compute_laplacian(
    field: NDArray[np.floating],
    boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
    check_stability: bool = True,
    use_accel: bool | None = None,
) -> NDArray[np.floating]:
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError(f"field must be 2D, got ndim={field.ndim}")

    accel = _use_accel(use_accel) and njit is not None
    if boundary == BoundaryCondition.PERIODIC:
        laplacian = _laplacian_periodic_jit(field) if accel else _laplacian_numpy_periodic(field)
    elif boundary == BoundaryCondition.NEUMANN:
        laplacian = _laplacian_neumann_jit(field) if accel else _laplacian_numpy_neumann(field)
    else:
        laplacian = _laplacian_dirichlet_jit(field) if accel else _laplacian_numpy_dirichlet(field)

    if check_stability:
        validate_field_stability(laplacian, field_name="laplacian")
    return cast(NDArray[np.floating[Any]], laplacian)


def compute_gradient(
    field: NDArray[np.floating],
    boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    if boundary == BoundaryCondition.PERIODIC:
        grad_x = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / 2.0
        grad_y = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / 2.0
    else:
        grad_x = np.zeros_like(field)
        grad_y = np.zeros_like(field)
        grad_x[1:-1, :] = (field[2:, :] - field[:-2, :]) / 2.0
        grad_y[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / 2.0
        if boundary == BoundaryCondition.NEUMANN:
            grad_x[0, :] = 0.0
            grad_x[-1, :] = 0.0
            grad_y[:, 0] = 0.0
            grad_y[:, -1] = 0.0
        else:
            grad_x[0, :] = field[1, :] - field[0, :]
            grad_x[-1, :] = field[-1, :] - field[-2, :]
            grad_y[:, 0] = field[:, 1] - field[:, 0]
            grad_y[:, -1] = field[:, -1] - field[:, -2]
    return cast(NDArray[np.floating[Any]], grad_x), cast(NDArray[np.floating[Any]], grad_y)


def compute_field_statistics(field: NDArray[np.floating]) -> dict[str, float]:
    return {
        "min": float(np.min(field)),
        "max": float(np.max(field)),
        "mean": float(np.mean(field)),
        "std": float(np.std(field)),
        "nan_count": int(np.sum(np.isnan(field))),
        "inf_count": int(np.sum(np.isinf(field))),
        "finite_fraction": float(np.mean(np.isfinite(field))),
    }


def validate_field_stability(
    field: NDArray[np.floating],
    field_name: str = "field",
    step: int | None = None,
) -> bool:
    nan_count = int(np.sum(np.isnan(field)))
    inf_count = int(np.sum(np.isinf(field)))
    if nan_count > 0:
        raise NumericalInstabilityError(
            f"NaN values detected in {field_name}",
            step=step,
            field_name=field_name,
            nan_count=nan_count,
        )
    if inf_count > 0:
        raise NumericalInstabilityError(
            f"Inf values detected in {field_name}",
            step=step,
            field_name=field_name,
            inf_count=inf_count,
        )
    return True


def validate_field_bounds(
    field: NDArray[np.floating],
    min_value: float,
    max_value: float,
) -> bool:
    return bool(np.all((field >= min_value) & (field <= max_value)))


def clamp_field(
    field: NDArray[np.floating],
    min_value: float,
    max_value: float,
) -> tuple[NDArray[np.floating], int]:
    needs_clamping = (field < min_value) | (field > max_value)
    clamp_count = int(np.sum(needs_clamping))
    clamped = np.clip(field, min_value, max_value)
    return cast(NDArray[np.floating[Any]], clamped), clamp_count


__all__ = [
    "BoundaryCondition",
    "compute_laplacian",
    "laplacian_backend",
    "compute_gradient",
    "compute_field_statistics",
    "validate_field_stability",
    "validate_field_bounds",
    "clamp_field",
]
