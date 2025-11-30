"""
Field state types for MyceliumFractalNet.

Defines canonical types for representing the state and history of
simulated 2D potential fields. These types encapsulate numpy arrays
with validated invariants.

Reference:
    - docs/MFN_MATH_MODEL.md — Field evolution equations
    - docs/ARCHITECTURE.md — Turing morphogenesis section
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray


class BoundaryCondition(str, Enum):
    """
    Boundary condition for field simulation.

    See docs/MFN_MATH_MODEL.md Section 2.7 for implementation details.

    Attributes:
        PERIODIC: Wrap-around boundaries (default for Turing patterns)
        NEUMANN: Zero-flux boundaries (∂V/∂n = 0)
        DIRICHLET: Fixed-value boundaries (V = V_boundary)
    """

    PERIODIC = "periodic"
    NEUMANN = "neumann"
    DIRICHLET = "dirichlet"


@dataclass(frozen=True)
class GridShape:
    """
    Immutable grid shape specification.

    Attributes:
        rows: Number of rows (N dimension)
        cols: Number of columns (M dimension)

    Note:
        MFN typically uses square grids (rows == cols), but this type
        supports rectangular grids for future extensibility.

    Reference:
        docs/MFN_MATH_MODEL.md Section 2 — Grid parameters
    """

    rows: int
    cols: int

    def __post_init__(self) -> None:
        """Validate grid dimensions."""
        if self.rows < 2:
            raise ValueError(f"rows must be >= 2, got {self.rows}")
        if self.cols < 2:
            raise ValueError(f"cols must be >= 2, got {self.cols}")

    @property
    def is_square(self) -> bool:
        """Check if grid is square."""
        return self.rows == self.cols

    @property
    def size(self) -> int:
        """Return size for square grids."""
        if not self.is_square:
            raise ValueError("size only defined for square grids")
        return self.rows

    @property
    def total_cells(self) -> int:
        """Total number of cells in the grid."""
        return self.rows * self.cols

    def to_tuple(self) -> Tuple[int, int]:
        """Convert to (rows, cols) tuple."""
        return (self.rows, self.cols)

    @classmethod
    def square(cls, size: int) -> "GridShape":
        """Create a square grid shape."""
        return cls(rows=size, cols=size)


@dataclass
class FieldState:
    """
    Represents the state of a 2D potential field at a single time point.

    The field stores membrane potentials in Volts. Values are typically
    in the range [-0.095, 0.040] V (-95 mV to +40 mV) per MFN_MATH_MODEL.md.

    Attributes:
        data: 2D numpy array of potential values in Volts
        boundary: Boundary condition type

    Invariants:
        - data must be 2D with shape (N, M) where N, M >= 2
        - All values should be finite (no NaN/Inf)

    Reference:
        docs/MFN_MATH_MODEL.md Section 2.6 — Membrane Potential Field Evolution
    """

    data: NDArray[np.float64]
    boundary: BoundaryCondition = BoundaryCondition.PERIODIC

    def __post_init__(self) -> None:
        """Validate field state."""
        if self.data.ndim != 2:
            raise ValueError(f"data must be 2D, got {self.data.ndim}D")
        if self.data.shape[0] < 2 or self.data.shape[1] < 2:
            raise ValueError(f"grid dimensions must be >= 2, got {self.data.shape}")
        if not np.isfinite(self.data).all():
            raise ValueError("data contains NaN or Inf values")

    @property
    def shape(self) -> GridShape:
        """Return grid shape."""
        return GridShape(rows=self.data.shape[0], cols=self.data.shape[1])

    @property
    def grid_size(self) -> int:
        """Return grid size for square fields."""
        if self.data.shape[0] != self.data.shape[1]:
            raise ValueError("grid_size only defined for square fields")
        return int(self.data.shape[0])

    @property
    def min_mV(self) -> float:
        """Minimum potential in millivolts."""
        return float(np.min(self.data)) * 1000.0

    @property
    def max_mV(self) -> float:
        """Maximum potential in millivolts."""
        return float(np.max(self.data)) * 1000.0

    @property
    def mean_mV(self) -> float:
        """Mean potential in millivolts."""
        return float(np.mean(self.data)) * 1000.0

    @property
    def std_mV(self) -> float:
        """Standard deviation of potential in millivolts."""
        return float(np.std(self.data)) * 1000.0

    def to_binary(self, threshold_v: float = -0.060) -> NDArray[np.bool_]:
        """
        Convert to binary mask using threshold.

        Args:
            threshold_v: Threshold in Volts (default: -60 mV)

        Returns:
            Boolean array where True indicates V > threshold
        """
        return self.data > threshold_v

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "shape": self.shape.to_tuple(),
            "boundary": self.boundary.value,
            "min_mV": self.min_mV,
            "max_mV": self.max_mV,
            "mean_mV": self.mean_mV,
            "std_mV": self.std_mV,
        }


@dataclass
class FieldHistory:
    """
    Represents the time evolution of a 2D potential field.

    Stores snapshots of the field at each time step for temporal analysis
    and feature extraction.

    Attributes:
        data: 3D numpy array of shape (T, N, M) where T is time steps
        boundary: Boundary condition type

    Invariants:
        - data must be 3D with shape (T, N, M) where T >= 1, N, M >= 2
        - All values should be finite (no NaN/Inf)

    Reference:
        docs/MFN_FEATURE_SCHEMA.md Section 2.3 — Temporal Features
    """

    data: NDArray[np.float64]
    boundary: BoundaryCondition = BoundaryCondition.PERIODIC

    def __post_init__(self) -> None:
        """Validate field history."""
        if self.data.ndim != 3:
            raise ValueError(f"data must be 3D (T, N, M), got {self.data.ndim}D")
        if self.data.shape[0] < 1:
            raise ValueError(f"time steps must be >= 1, got {self.data.shape[0]}")
        if self.data.shape[1] < 2 or self.data.shape[2] < 2:
            raise ValueError(
                f"spatial dimensions must be >= 2, got {self.data.shape[1:]}"
            )
        if not np.isfinite(self.data).all():
            raise ValueError("data contains NaN or Inf values")

    @property
    def num_steps(self) -> int:
        """Number of time steps."""
        return int(self.data.shape[0])

    @property
    def spatial_shape(self) -> GridShape:
        """Return spatial grid shape."""
        return GridShape(rows=self.data.shape[1], cols=self.data.shape[2])

    @property
    def grid_size(self) -> int:
        """Return grid size for square fields."""
        if self.data.shape[1] != self.data.shape[2]:
            raise ValueError("grid_size only defined for square fields")
        return int(self.data.shape[1])

    def get_frame(self, t: int) -> FieldState:
        """Get field state at time step t."""
        if t < 0 or t >= self.num_steps:
            raise IndexError(f"time index {t} out of range [0, {self.num_steps})")
        return FieldState(data=self.data[t].copy(), boundary=self.boundary)

    @property
    def initial_state(self) -> FieldState:
        """Get initial field state (t=0)."""
        return self.get_frame(0)

    @property
    def final_state(self) -> FieldState:
        """Get final field state (t=T-1)."""
        return self.get_frame(self.num_steps - 1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation (without raw data)."""
        return {
            "num_steps": self.num_steps,
            "spatial_shape": self.spatial_shape.to_tuple(),
            "boundary": self.boundary.value,
            "initial_min_mV": self.initial_state.min_mV,
            "initial_max_mV": self.initial_state.max_mV,
            "final_min_mV": self.final_state.min_mV,
            "final_max_mV": self.final_state.max_mV,
        }
