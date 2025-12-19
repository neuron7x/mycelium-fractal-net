"""
Grid Operations for Reaction-Diffusion Simulations.

Implements discretized spatial operators for 2D field simulations:
- Discrete Laplacian (5-point stencil) with multiple boundary conditions
- Gradient operators
- Field statistics and validation

Reference: MFN_MATH_MODEL.md Section 2.4 (Discrete Laplacian)

Mathematical Formulation:
    The continuous Laplacian ∇²u is discretized on a uniform 2D grid:
    
    ∇²u ≈ (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4·u[i,j]) / dx²
    
    With dx=1 (unit grid spacing), the stencil simplifies to:
    
    ∇²u ≈ u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4·u[i,j]

Discretization Details:
    - Spatial: Uniform grid, unit spacing (dx=dy=1)
    - Stencil: 5-point (cross) stencil
    - Order of accuracy: O(dx²) - second-order
    
Boundary Conditions:
    - Periodic: Uses np.roll (toroidal topology)
    - Neumann: Zero-flux (mirror at boundary, dV/dn=0)
    - Dirichlet: Fixed value at boundary (typically zero)

Stability Constraint (CFL):
    For explicit Euler time stepping:
    dt · D · 4/dx² ≤ 1
    With dx=1, dt=1: D_max = 0.25
"""

from __future__ import annotations

from enum import Enum
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]

from mycelium_fractal_net.core.exceptions import NumericalInstabilityError


class BoundaryCondition(Enum):
    """
    Available boundary conditions for spatial discretization.
    
    Reference: MFN_MATH_MODEL.md Section 2.4
    """

    PERIODIC = "periodic"
    """Periodic (wrap-around) boundaries using np.roll."""
    
    NEUMANN = "neumann"
    """Zero-flux (Neumann) boundaries - gradient is zero at boundary."""
    
    DIRICHLET = "dirichlet"
    """Fixed value (Dirichlet) boundaries - typically zero."""


def compute_laplacian(
    field: FloatArray,
    boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
    check_stability: bool = True,
) -> FloatArray:
    """
    Compute discrete 2D Laplacian using 5-point stencil.
    
    Implements discretization from MFN_MATH_MODEL.md Section 2.4:
    
        ∇²u_{i,j} ≈ u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4·u_{i,j}
    
    Discretization scheme:
        - Explicit finite difference
        - 5-point cross stencil
        - Second-order accuracy O(dx²)
        - Unit grid spacing (dx=1)
    
    Parameters
    ----------
    field : FloatArray
        Input field of shape (N, M), typically (N, N) square grid.
    boundary : BoundaryCondition
        Boundary condition type (periodic, neumann, dirichlet).
    check_stability : bool
        Whether to check for NaN/Inf after computation.
    
    Returns
    -------
    FloatArray
        Laplacian of shape (N, M), same as input.
    
    Raises
    ------
    NumericalInstabilityError
        If NaN or Inf values are detected (when check_stability=True).
    
    Examples
    --------
    >>> import numpy as np
    >>> field = np.zeros((10, 10))
    >>> field[5, 5] = 1.0  # Point source
    >>> lap = compute_laplacian(field)
    >>> print(f"Laplacian at center: {lap[5, 5]:.2f}")  # -4.0
    
    Notes
    -----
    - For diffusion equation ∂u/∂t = D·∇²u, use:
      u_new = u + dt * D * compute_laplacian(u)
    - Stability requires dt * D * 4 ≤ 1 (CFL condition)
    """
    if boundary == BoundaryCondition.PERIODIC:
        # Periodic boundaries via np.roll (most efficient)
        up = np.roll(field, 1, axis=0)
        down = np.roll(field, -1, axis=0)
        left = np.roll(field, 1, axis=1)
        right = np.roll(field, -1, axis=1)
        
    elif boundary == BoundaryCondition.NEUMANN:
        # Zero-flux (Neumann): boundary value equals interior neighbor
        # This enforces dV/dn = 0 at all boundaries
        up = np.empty_like(field)
        up[1:, :] = field[:-1, :]
        up[0, :] = field[0, :]  # First row: zero gradient
        
        down = np.empty_like(field)
        down[:-1, :] = field[1:, :]
        down[-1, :] = field[-1, :]  # Last row: zero gradient
        
        left = np.empty_like(field)
        left[:, 1:] = field[:, :-1]
        left[:, 0] = field[:, 0]  # First column: zero gradient
        
        right = np.empty_like(field)
        right[:, :-1] = field[:, 1:]
        right[:, -1] = field[:, -1]  # Last column: zero gradient
        
    else:  # DIRICHLET
        # Fixed value at boundaries (zero by default)
        up = np.pad(field[:-1, :], ((1, 0), (0, 0)), mode="constant", constant_values=0)
        down = np.pad(field[1:, :], ((0, 1), (0, 0)), mode="constant", constant_values=0)
        left = np.pad(field[:, :-1], ((0, 0), (1, 0)), mode="constant", constant_values=0)
        right = np.pad(field[:, 1:], ((0, 0), (0, 1)), mode="constant", constant_values=0)
    
    laplacian = up + down + left + right - 4.0 * field
    
    # Stability check
    if check_stability:
        nan_count = int(np.sum(np.isnan(laplacian)))
        inf_count = int(np.sum(np.isinf(laplacian)))
        
        if nan_count > 0:
            raise NumericalInstabilityError(
                "NaN values in Laplacian computation",
                field_name="laplacian",
                nan_count=nan_count,
            )
        if inf_count > 0:
            raise NumericalInstabilityError(
                "Inf values in Laplacian computation",
                field_name="laplacian",
                inf_count=inf_count,
            )
    
    return cast(FloatArray, laplacian)


def compute_gradient(
    field: FloatArray,
    boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
) -> tuple[FloatArray, FloatArray]:
    """
    Compute spatial gradient using central differences.
    
    Implements central difference approximation:
        ∂u/∂x ≈ (u[i+1,j] - u[i-1,j]) / 2
        ∂u/∂y ≈ (u[i,j+1] - u[i,j-1]) / 2
    
    Discretization scheme:
        - Central difference
        - Second-order accuracy O(dx²)
        - Unit grid spacing
    
    Parameters
    ----------
    field : FloatArray
        Input field of shape (N, M).
    boundary : BoundaryCondition
        Boundary condition type.
    
    Returns
    -------
    tuple[NDArray, NDArray]
        (grad_x, grad_y) gradient components.
    """
    if boundary == BoundaryCondition.PERIODIC:
        grad_x: FloatArray = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / 2.0
        grad_y: FloatArray = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / 2.0
    else:
        # Use forward/backward difference at boundaries
        grad_x = np.zeros_like(field)
        grad_y = np.zeros_like(field)
        
        # Interior: central difference
        grad_x[1:-1, :] = (field[2:, :] - field[:-2, :]) / 2.0
        grad_y[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / 2.0
        
        # Boundaries: forward/backward difference
        grad_x[0, :] = field[1, :] - field[0, :]
        grad_x[-1, :] = field[-1, :] - field[-2, :]
        grad_y[:, 0] = field[:, 1] - field[:, 0]
        grad_y[:, -1] = field[:, -1] - field[:, -2]
    
    return grad_x, grad_y


def compute_field_statistics(
    field: FloatArray,
) -> dict[str, float]:
    """
    Compute statistical summary of field values.
    
    Useful for monitoring stability and validating simulation state.
    Reference: MFN_MATH_MODEL.md Section 4.3 (Clamping and Bounds)
    
    Parameters
    ----------
    field : FloatArray
        Input field of shape (N, M).
    
    Returns
    -------
    dict[str, float]
        Statistics including min, max, mean, std, and NaN/Inf counts.
    """
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
    field: FloatArray,
    field_name: str = "field",
    step: int | None = None,
) -> bool:
    """
    Validate field contains no NaN or Inf values.
    
    Reference: MFN_MATH_MODEL.md Section 2.9 (Validation Invariants)
    
    Parameters
    ----------
    field : FloatArray
        Field to validate.
    field_name : str
        Name for error reporting.
    step : int | None
        Current simulation step for error reporting.
    
    Returns
    -------
    bool
        True if field is stable (no NaN/Inf).
    
    Raises
    ------
    NumericalInstabilityError
        If NaN or Inf values are detected.
    """
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
    field: FloatArray,
    min_value: float,
    max_value: float,
) -> bool:
    """
    Validate all field values are within specified bounds.
    
    Reference: MFN_MATH_MODEL.md Section 4.3 (Clamping and Bounds)
    
    Parameters
    ----------
    field : FloatArray
        Field to validate.
    min_value : float
        Minimum allowed value.
    max_value : float
        Maximum allowed value.
    
    Returns
    -------
    bool
        True if all values are within bounds.
    """
    return bool(np.all((field >= min_value) & (field <= max_value)))


def clamp_field(
    field: FloatArray,
    min_value: float,
    max_value: float,
) -> tuple[FloatArray, int]:
    """
    Clamp field values to specified range and count clamping events.
    
    Reference: MFN_MATH_MODEL.md Section 4.3 (Clamping and Bounds)
    
    Parameters
    ----------
    field : FloatArray
        Field to clamp (modified in place).
    min_value : float
        Minimum allowed value.
    max_value : float
        Maximum allowed value.
    
    Returns
    -------
    tuple[NDArray, int]
        Clamped field and count of values that were clamped.
    """
    needs_clamping = (field < min_value) | (field > max_value)
    clamp_count = int(np.sum(needs_clamping))
    
    clamped: FloatArray = np.clip(field, min_value, max_value)
    
    return clamped, clamp_count
