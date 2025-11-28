"""
Reaction-Diffusion Engine for Turing Morphogenesis.

Implements stable numerical discretization of the activator-inhibitor
reaction-diffusion system for pattern formation.

Mathematical Model (from docs/ARCHITECTURE.md Section 2):
    ∂a/∂t = D_a ∇²a + r_a·a(1-a) - i
    ∂i/∂t = D_i ∇²i + r_i·(a - i)

Spatial discretization:
    ∇²u ≈ (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4·u[i,j]) / dx²

Stability (CFL condition for explicit diffusion):
    dt ≤ dx² / (4·max(D_a, D_i))

Reference:
    - Turing, A.M. (1952). The chemical basis of morphogenesis.
    - Cross & Hohenberg (1993). Pattern formation outside of equilibrium.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

from mycelium_fractal_net.core.config import BoundaryCondition, ReactionDiffusionConfig
from mycelium_fractal_net.core.exceptions import (
    NumericalInstabilityError,
)


@dataclass
class ReactionDiffusionMetrics:
    """
    Metrics collected during reaction-diffusion simulation.

    Attributes:
        steps_completed: Number of integration steps.
        activator_min: Minimum activator concentration.
        activator_max: Maximum activator concentration.
        activator_mean: Mean activator concentration.
        inhibitor_min: Minimum inhibitor concentration.
        inhibitor_max: Maximum inhibitor concentration.
        inhibitor_mean: Mean inhibitor concentration.
        pattern_fraction: Fraction of cells above Turing threshold.
        nan_detected: Whether NaN was detected.
        inf_detected: Whether Inf was detected.
        values_clamped: Number of clamping events.
        cfl_number: Current CFL number.
        execution_time_s: Execution time in seconds.
    """

    steps_completed: int = 0
    activator_min: float = 0.0
    activator_max: float = 0.0
    activator_mean: float = 0.0
    inhibitor_min: float = 0.0
    inhibitor_max: float = 0.0
    inhibitor_mean: float = 0.0
    pattern_fraction: float = 0.0
    nan_detected: bool = False
    inf_detected: bool = False
    values_clamped: int = 0
    cfl_number: float = 0.0
    execution_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "steps_completed": self.steps_completed,
            "activator_min": self.activator_min,
            "activator_max": self.activator_max,
            "activator_mean": self.activator_mean,
            "inhibitor_min": self.inhibitor_min,
            "inhibitor_max": self.inhibitor_max,
            "inhibitor_mean": self.inhibitor_mean,
            "pattern_fraction": self.pattern_fraction,
            "nan_detected": self.nan_detected,
            "inf_detected": self.inf_detected,
            "values_clamped": self.values_clamped,
            "cfl_number": self.cfl_number,
            "execution_time_s": self.execution_time_s,
        }


class ReactionDiffusionEngine:
    """
    Numerical engine for Turing reaction-diffusion morphogenesis.

    Implements explicit finite-difference scheme for activator-inhibitor
    dynamics with configurable boundary conditions.

    Features:
    - Periodic, Neumann, or Dirichlet boundary conditions
    - CFL stability validation
    - NaN/Inf detection with early termination
    - Concentration clamping to [0, 1]
    - Pattern detection via Turing threshold

    Example:
        >>> config = ReactionDiffusionConfig(grid_size=64, steps=200, random_seed=42)
        >>> engine = ReactionDiffusionEngine(config)
        >>> activator, inhibitor, metrics = engine.simulate()
        >>> print(f"Pattern fraction: {metrics.pattern_fraction:.2%}")

    Reference: docs/ARCHITECTURE.md Section 2
    """

    def __init__(self, config: ReactionDiffusionConfig | None = None) -> None:
        """
        Initialize reaction-diffusion engine.

        Args:
            config: Configuration parameters. Uses defaults if None.

        Raises:
            ValueError: If configuration is invalid (e.g., CFL violation).
        """
        self.config = config or ReactionDiffusionConfig()
        self.config.validate()
        self._rng = np.random.default_rng(self.config.random_seed)
        self._metrics = ReactionDiffusionMetrics()
        self._metrics.cfl_number = self.config.cfl_condition

    def _apply_boundary(
        self, field: NDArray[Any]
    ) -> NDArray[Any]:
        """
        Apply boundary conditions to field.

        For periodic: uses np.roll (already implicit in Laplacian)
        For Neumann: zero-gradient at edges
        For Dirichlet: fixed values at edges (zeros by default)
        """
        if self.config.boundary == BoundaryCondition.PERIODIC:
            return field  # Periodic handled in Laplacian
        elif self.config.boundary == BoundaryCondition.NEUMANN:
            # Zero-flux: copy edge values
            field[0, :] = field[1, :]
            field[-1, :] = field[-2, :]
            field[:, 0] = field[:, 1]
            field[:, -1] = field[:, -2]
            return field
        else:  # Dirichlet
            # Fixed boundaries (zeros)
            field[0, :] = 0.0
            field[-1, :] = 0.0
            field[:, 0] = 0.0
            field[:, -1] = 0.0
            return field

    def _compute_laplacian(self, field: NDArray[Any]) -> NDArray[Any]:
        """
        Compute discrete Laplacian using 5-point stencil.

        ∇²u ≈ (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4·u[i,j]) / dx²

        Uses periodic boundary conditions via np.roll.
        dx = 1.0 (unit grid spacing).
        """
        # For periodic boundary, use np.roll
        up = np.roll(field, 1, axis=0)
        down = np.roll(field, -1, axis=0)
        left = np.roll(field, 1, axis=1)
        right = np.roll(field, -1, axis=1)

        laplacian = up + down + left + right - 4.0 * field

        return laplacian  # dx² = 1.0

    def _check_stability(
        self,
        activator: NDArray[Any],
        inhibitor: NDArray[Any],
        step: int,
    ) -> int:
        """
        Check for NaN/Inf and count out-of-range values.

        Returns:
            Number of values clamped.

        Raises:
            NumericalInstabilityError: If NaN/Inf detected.
        """
        nan_a = int(np.isnan(activator).sum())
        nan_i = int(np.isnan(inhibitor).sum())
        inf_a = int(np.isinf(activator).sum())
        inf_i = int(np.isinf(inhibitor).sum())

        total_nan = nan_a + nan_i
        total_inf = inf_a + inf_i

        if total_nan > 0 or total_inf > 0:
            self._metrics.nan_detected = total_nan > 0
            self._metrics.inf_detected = total_inf > 0
            if self.config.check_stability:
                raise NumericalInstabilityError(
                    "Reaction-diffusion simulation produced NaN/Inf",
                    field_name="activator/inhibitor",
                    nan_count=total_nan,
                    inf_count=total_inf,
                    step=step,
                )

        # Count out-of-range values
        clamped_a = ((activator < 0) | (activator > 1)).sum()
        clamped_i = ((inhibitor < 0) | (inhibitor > 1)).sum()

        return int(clamped_a + clamped_i)

    def simulate(
        self,
        activator_init: NDArray[Any] | None = None,
        inhibitor_init: NDArray[Any] | None = None,
    ) -> Tuple[NDArray[Any], NDArray[Any], ReactionDiffusionMetrics]:
        """
        Simulate reaction-diffusion dynamics.

        Args:
            activator_init: Initial activator field (N, N). Default: random [0, 0.1].
            inhibitor_init: Initial inhibitor field (N, N). Default: random [0, 0.1].

        Returns:
            activator: Final activator field (N, N).
            inhibitor: Final inhibitor field (N, N).
            metrics: Simulation metrics.

        Raises:
            NumericalInstabilityError: If simulation becomes unstable.
        """
        start_time = time.perf_counter()
        N = self.config.grid_size
        dt = self.config.dt

        # Initialize fields
        if activator_init is not None:
            activator = activator_init.copy()
        else:
            activator = self._rng.uniform(0, 0.1, size=(N, N))

        if inhibitor_init is not None:
            inhibitor = inhibitor_init.copy()
        else:
            inhibitor = self._rng.uniform(0, 0.1, size=(N, N))

        # Parameters
        D_a = self.config.d_activator
        D_i = self.config.d_inhibitor
        r_a = self.config.r_activator
        r_i = self.config.r_inhibitor

        total_clamped = 0

        # Integration loop (explicit Euler for PDEs)
        for step in range(self.config.steps):
            # Apply boundary conditions
            if self.config.boundary != BoundaryCondition.PERIODIC:
                activator = self._apply_boundary(activator)
                inhibitor = self._apply_boundary(inhibitor)

            # Compute Laplacians
            lap_a = self._compute_laplacian(activator)
            lap_i = self._compute_laplacian(inhibitor)

            # Reaction-diffusion update
            # ∂a/∂t = D_a ∇²a + r_a·a(1-a) - i
            # ∂i/∂t = D_i ∇²i + r_i·(a - i)
            da_dt = D_a * lap_a + r_a * (activator * (1.0 - activator)) - inhibitor
            di_dt = D_i * lap_i + r_i * (activator - inhibitor)

            activator = activator + dt * da_dt
            inhibitor = inhibitor + dt * di_dt

            # Stability check
            clamped = self._check_stability(activator, inhibitor, step)
            total_clamped += clamped

            # Clamp to [0, 1]
            activator = np.clip(activator, 0.0, 1.0)
            inhibitor = np.clip(inhibitor, 0.0, 1.0)

        # Compute pattern fraction
        pattern_mask = activator > self.config.turing_threshold
        pattern_fraction = float(pattern_mask.sum()) / (N * N)

        # Collect metrics
        self._metrics.steps_completed = self.config.steps
        self._metrics.activator_min = float(activator.min())
        self._metrics.activator_max = float(activator.max())
        self._metrics.activator_mean = float(activator.mean())
        self._metrics.inhibitor_min = float(inhibitor.min())
        self._metrics.inhibitor_max = float(inhibitor.max())
        self._metrics.inhibitor_mean = float(inhibitor.mean())
        self._metrics.pattern_fraction = pattern_fraction
        self._metrics.values_clamped = total_clamped
        self._metrics.execution_time_s = time.perf_counter() - start_time

        return activator, inhibitor, self._metrics

    def simulate_with_field(
        self,
        field: NDArray[Any],
        field_coupling: float = 0.005,
    ) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], ReactionDiffusionMetrics]:
        """
        Simulate reaction-diffusion with coupling to external field.

        This is the "mycelium field" mode where the membrane potential field
        is modulated by the Turing pattern.

        Args:
            field: Membrane potential field (N, N) in volts.
            field_coupling: Coupling strength from activator to field.

        Returns:
            activator: Final activator field.
            inhibitor: Final inhibitor field.
            field_out: Modified membrane field.
            metrics: Simulation metrics.
        """
        start_time = time.perf_counter()
        N = self.config.grid_size

        if field.shape != (N, N):
            raise ValueError(f"Field shape {field.shape} does not match grid_size {N}")

        field_out = field.copy()

        # Initialize Turing fields
        activator = self._rng.uniform(0, 0.1, size=(N, N))
        inhibitor = self._rng.uniform(0, 0.1, size=(N, N))

        D_a = self.config.d_activator
        D_i = self.config.d_inhibitor
        r_a = self.config.r_activator
        r_i = self.config.r_inhibitor
        dt = self.config.dt
        total_clamped = 0

        for step in range(self.config.steps):
            # Laplacians
            lap_a = self._compute_laplacian(activator)
            lap_i = self._compute_laplacian(inhibitor)

            # Reaction-diffusion
            da_dt = D_a * lap_a + r_a * (activator * (1.0 - activator)) - inhibitor
            di_dt = D_i * lap_i + r_i * (activator - inhibitor)

            activator = activator + dt * da_dt
            inhibitor = inhibitor + dt * di_dt

            # Stability check
            clamped = self._check_stability(activator, inhibitor, step)
            total_clamped += clamped

            # Clamp
            activator = np.clip(activator, 0.0, 1.0)
            inhibitor = np.clip(inhibitor, 0.0, 1.0)

            # Apply Turing pattern to field where activator exceeds threshold
            turing_mask = activator > self.config.turing_threshold
            field_out[turing_mask] += field_coupling

        # Clamp field to physiological range [-95, 40] mV
        field_out = np.clip(field_out, -0.095, 0.040)

        # Collect metrics
        pattern_mask = activator > self.config.turing_threshold
        self._metrics.steps_completed = self.config.steps
        self._metrics.activator_min = float(activator.min())
        self._metrics.activator_max = float(activator.max())
        self._metrics.activator_mean = float(activator.mean())
        self._metrics.inhibitor_min = float(inhibitor.min())
        self._metrics.inhibitor_max = float(inhibitor.max())
        self._metrics.inhibitor_mean = float(inhibitor.mean())
        self._metrics.pattern_fraction = float(pattern_mask.sum()) / (N * N)
        self._metrics.values_clamped = total_clamped
        self._metrics.execution_time_s = time.perf_counter() - start_time

        return activator, inhibitor, field_out, self._metrics

    @property
    def metrics(self) -> ReactionDiffusionMetrics:
        """Get current metrics."""
        return self._metrics

    def reset(self) -> None:
        """Reset engine state."""
        self._rng = np.random.default_rng(self.config.random_seed)
        self._metrics = ReactionDiffusionMetrics()
        self._metrics.cfl_number = self.config.cfl_condition
