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

    Reference:
        docs/MFN_DATA_MODEL.md — Canonical data model
        docs/MFN_MATH_MODEL.md — Parameter bounds and physical constraints
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
        if not (4 <= self.grid_size <= 512):
            raise ValueError("grid_size must be between 4 and 512")
        if not (1 <= self.steps <= 10000):
            raise ValueError("steps must be between 1 and 10000")
        if not (0.0 < self.alpha <= 0.25):
            raise ValueError("alpha must be in (0, 0.25] for CFL stability")
        if not (0.0 <= self.spike_probability <= 1.0):
            raise ValueError("spike_probability must be in [0, 1]")
        if not (0.0 <= self.turing_threshold <= 1.0):
            raise ValueError("turing_threshold must be in [0, 1]")
        if not (0.0 <= self.jitter_var <= 0.01):
            raise ValueError("jitter_var must be between 0.0 and 0.01")

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "grid_size": self.grid_size,
            "steps": self.steps,
            "alpha": self.alpha,
            "spike_probability": self.spike_probability,
            "turing_enabled": self.turing_enabled,
            "turing_threshold": self.turing_threshold,
            "quantum_jitter": self.quantum_jitter,
            "jitter_var": self.jitter_var,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationConfig":
        """
        Create configuration from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            New SimulationConfig instance.
        """
        def _parse_bool(value: Any, default: bool) -> bool:
            """Parse booleans from common serialized formats.

            Supports native bools, string representations ("true"/"false"),
            and numeric 0/1 flags. Falls back to the provided default when the
            value is None or unrecognized.
            """

            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "1", "yes", "y", "on"}:
                    return True
                if normalized in {"false", "0", "no", "n", "off"}:
                    return False
                return default
            if isinstance(value, (int, float)):
                return bool(value)
            return default

        seed_value = data.get("seed")
        return cls(
            grid_size=int(data.get("grid_size", 64)),
            steps=int(data.get("steps", 64)),
            alpha=float(data.get("alpha", 0.18)),
            spike_probability=float(data.get("spike_probability", 0.25)),
            turing_enabled=_parse_bool(data.get("turing_enabled"), True),
            turing_threshold=float(data.get("turing_threshold", 0.75)),
            quantum_jitter=_parse_bool(data.get("quantum_jitter"), False),
            jitter_var=float(data.get("jitter_var", 0.0005)),
            seed=int(seed_value) if seed_value is not None else None,
        )


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
    turing_activations : int
        Number of Turing pattern activation events during simulation.
    clamping_events : int
        Number of field clamping events during simulation.
    metadata : dict[str, Any]
        Additional simulation metadata (timing, parameters, etc.).

    Reference:
        docs/MFN_DATA_MODEL.md — Canonical data model
        docs/MFN_DATA_PIPELINES.md — Dataset schema
    """

    field: NDArray[np.float64]
    history: NDArray[np.float64] | None = None
    growth_events: int = 0
    turing_activations: int = 0
    clamping_events: int = 0
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
            if not np.isfinite(self.history).all():
                raise ValueError("history contains NaN or Inf values")
        if not np.isfinite(self.field).all():
            raise ValueError("field contains NaN or Inf values")

    @property
    def grid_size(self) -> int:
        """Return the grid size N."""
        return int(self.field.shape[0])

    @property
    def has_history(self) -> bool:
        """Check if time history is available."""
        return self.history is not None

    @property
    def num_steps(self) -> int:
        """Return the number of simulated time steps represented by the result.

        Preference order:
        1. Explicit history length when available.
        2. ``steps_computed`` in metadata produced by the simulation engine.
        3. ``config['steps']`` when present in metadata.
        4. Fallback to ``0`` when no information is available.
        """
        if self.history is not None:
            return int(self.history.shape[0])

        steps_computed = self.metadata.get("steps_computed")
        if isinstance(steps_computed, (int, float)):
            return int(steps_computed)

        config_metadata = self.metadata.get("config")
        if isinstance(config_metadata, dict):
            steps_value = config_metadata.get("steps")
            if isinstance(steps_value, (int, float)):
                return int(steps_value)

        return 0

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        """
        Serialize result to a dictionary.

        Args:
            include_arrays: If True, include field and history arrays as lists.
                           If False, only include metadata and statistics.

        Returns:
            Dictionary representation of the result.
        """
        result: dict[str, Any] = {
            "grid_size": self.grid_size,
            "num_steps": self.num_steps,
            "has_history": self.has_history,
            "growth_events": self.growth_events,
            "turing_activations": self.turing_activations,
            "clamping_events": self.clamping_events,
            "metadata": self.metadata.copy(),
            # Field statistics
            "field_min_mV": float(np.min(self.field)) * 1000.0,
            "field_max_mV": float(np.max(self.field)) * 1000.0,
            "field_mean_mV": float(np.mean(self.field)) * 1000.0,
            "field_std_mV": float(np.std(self.field)) * 1000.0,
        }
        if include_arrays:
            result["field"] = self.field.tolist()
            if self.history is not None:
                result["history"] = self.history.tolist()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationResult":
        """
        Create SimulationResult from a dictionary.

        Args:
            data: Dictionary with result data including 'field' array.

        Returns:
            New SimulationResult instance.

        Raises:
            KeyError: If 'field' key is missing.
            ValueError: If field data is invalid.
        """
        if "field" not in data:
            raise KeyError("'field' key is required in data dictionary")

        field = np.array(data["field"], dtype=np.float64)
        history = None
        if "history" in data and data["history"] is not None:
            history = np.array(data["history"], dtype=np.float64)

        return cls(
            field=field,
            history=history,
            growth_events=int(data.get("growth_events", 0)),
            turing_activations=int(data.get("turing_activations", 0)),
            clamping_events=int(data.get("clamping_events", 0)),
            metadata=dict(data.get("metadata", {})),
        )
