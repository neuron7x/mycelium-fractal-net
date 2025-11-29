"""
Simulation configuration and result types.

Provides dataclass-based types for simulation input/output.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SimulationConfig:
    """
    Configuration parameters for mycelium field simulation.

    Attributes
    ----------
    grid_size : int
        Size of the 2D grid (N × N). Default 64.
    steps : int
        Number of simulation steps. Default 64.
    alpha : float
        Diffusion coefficient. Must satisfy CFL condition (α ≤ 0.25). Default 0.18.
    spike_probability : float
        Probability of spike events per step. Default 0.25.
    turing_enabled : bool
        Enable Turing morphogenesis patterns. Default True.
    turing_threshold : float
        Activation threshold for Turing patterns. Default 0.75.
    quantum_jitter : bool
        Enable quantum noise jitter. Default False.
    jitter_var : float
        Variance of quantum jitter. Default 0.0005.
    seed : int | None
        Random seed for reproducibility. None for random seed.
    """

    grid_size: int = 64
    steps: int = 64
    alpha: float = 0.18
    spike_probability: float = 0.25
    turing_enabled: bool = True
    turing_threshold: float = 0.75
    quantum_jitter: bool = False
    jitter_var: float = 0.0005
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if not (0.0 < self.alpha <= 0.25):
            raise ValueError("alpha must be in (0, 0.25] for CFL stability")
        if not (0.0 <= self.spike_probability <= 1.0):
            raise ValueError("spike_probability must be in [0, 1]")
        if not (0.0 <= self.turing_threshold <= 1.0):
            raise ValueError("turing_threshold must be in [0, 1]")
        if self.jitter_var < 0.0:
            raise ValueError("jitter_var must be non-negative")


@dataclass
class SimulationResult:
    """
    Container for simulation output data.

    Attributes
    ----------
    field : NDArray[np.float64]
        Final 2D potential field in Volts. Shape (N, N).
    history : NDArray[np.float64] | None
        Time series of field snapshots. Shape (T, N, N). None if not stored.
    growth_events : int
        Total number of growth events during simulation.
    metadata : dict[str, Any]
        Additional simulation metadata (timing, parameters, etc.).
    """

    field: NDArray[np.float64]
    history: NDArray[np.float64] | None = None
    growth_events: int = 0
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result data."""
        if self.field.ndim != 2:
            raise ValueError("field must be 2D array")
        if self.field.shape[0] != self.field.shape[1]:
            raise ValueError("field must be square")
        if self.history is not None:
            if self.history.ndim != 3:
                raise ValueError("history must be 3D array (T, N, N)")
            if self.history.shape[1:] != self.field.shape:
                raise ValueError("history spatial dimensions must match field")

    @property
    def grid_size(self) -> int:
        """Return the grid size N."""
        return int(self.field.shape[0])

    @property
    def has_history(self) -> bool:
        """Check if time history is available."""
        return self.history is not None
