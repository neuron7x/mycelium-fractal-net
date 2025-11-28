"""
Reaction-Diffusion Engine — Turing Morphogenesis.

Implements stable numerical schemes for Turing pattern formation:
- Activator-inhibitor reaction-diffusion PDEs
- Discrete Laplacian with periodic boundaries
- CFL stability condition enforcement

Reference: MATH_MODEL.md Section 2 (Reaction-Diffusion Processes)

Equations Implemented:
    ∂a/∂t = D_a ∇²a + r_a * a(1-a) - i     # Activator
    ∂i/∂t = D_i ∇²i + r_i * (a - i)         # Inhibitor

    ∇²u ≈ u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]

Parameters (from MATH_MODEL.md Section 2.3):
    D_a = 0.1         - Activator diffusion (grid²/step)
    D_i = 0.05        - Inhibitor diffusion (grid²/step)
    r_a = 0.01        - Activator reaction rate (1/step)
    r_i = 0.02        - Inhibitor reaction rate (1/step)
    θ = 0.75          - Turing activation threshold

Stability Constraint (MATH_MODEL.md Section 2.5):
    dt * D * 4/dx² ≤ 1
    With dx=1, dt=1: D_max = 0.25
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from .exceptions import NumericalInstabilityError, StabilityError, ValueOutOfRangeError

# === Default Parameters (from MATH_MODEL.md Section 2.3) ===
DEFAULT_D_ACTIVATOR: float = 0.1
DEFAULT_D_INHIBITOR: float = 0.05
DEFAULT_R_ACTIVATOR: float = 0.01
DEFAULT_R_INHIBITOR: float = 0.02
DEFAULT_TURING_THRESHOLD: float = 0.75
DEFAULT_FIELD_ALPHA: float = 0.18
DEFAULT_QUANTUM_JITTER_VAR: float = 0.0005

# === Stability Limits ===
# CFL condition: D * dt * 4/dx² ≤ 1 → D ≤ 0.25 for dt=dx=1
MAX_STABLE_DIFFUSION: float = 0.25

# === Field Bounds (MATH_MODEL.md Section 4.3) ===
FIELD_V_MIN: float = -0.095  # -95 mV
FIELD_V_MAX: float = 0.040  # +40 mV
INITIAL_POTENTIAL_MEAN: float = -0.070  # -70 mV (resting potential)
INITIAL_POTENTIAL_STD: float = 0.005  # 5 mV


class BoundaryCondition(Enum):
    """Available boundary conditions for the spatial grid."""

    PERIODIC = "periodic"  # Wrap around (np.roll)
    NEUMANN = "neumann"  # Zero-flux (mirror at boundary)
    DIRICHLET = "dirichlet"  # Fixed value at boundary


@dataclass
class ReactionDiffusionConfig:
    """
    Configuration for reaction-diffusion engine.

    All parameters have physically meaningful defaults from MATH_MODEL.md.

    Attributes
    ----------
    grid_size : int
        Size of the square grid (N x N). Default 64.
    d_activator : float
        Activator diffusion coefficient. Default 0.1.
        Valid range: 0.01-0.5, must be < 0.25 for stability.
    d_inhibitor : float
        Inhibitor diffusion coefficient. Default 0.05.
        Valid range: 0.01-0.3, must be < 0.25 for stability.
    r_activator : float
        Activator reaction rate. Default 0.01.
        Valid range: 0.001-0.1.
    r_inhibitor : float
        Inhibitor reaction rate. Default 0.02.
        Valid range: 0.001-0.1.
    turing_threshold : float
        Threshold for pattern activation. Default 0.75.
        Valid range: 0.5-0.95.
    alpha : float
        Field diffusion coefficient. Default 0.18.
        Valid range: 0.05-0.24 (must be < 0.25 for stability).
    boundary_condition : BoundaryCondition
        Boundary condition type. Default PERIODIC.
    quantum_jitter : bool
        Enable stochastic noise term. Default False.
    jitter_var : float
        Variance of quantum jitter. Default 0.0005.
    spike_probability : float
        Probability of growth event per step. Default 0.25.
    check_stability : bool
        Check for NaN/Inf after each step. Default True.
    random_seed : int | None
        Seed for reproducibility. Default None.
    """

    grid_size: int = 64
    d_activator: float = DEFAULT_D_ACTIVATOR
    d_inhibitor: float = DEFAULT_D_INHIBITOR
    r_activator: float = DEFAULT_R_ACTIVATOR
    r_inhibitor: float = DEFAULT_R_INHIBITOR
    turing_threshold: float = DEFAULT_TURING_THRESHOLD
    alpha: float = DEFAULT_FIELD_ALPHA
    boundary_condition: BoundaryCondition = BoundaryCondition.PERIODIC
    quantum_jitter: bool = False
    jitter_var: float = DEFAULT_QUANTUM_JITTER_VAR
    spike_probability: float = 0.25
    check_stability: bool = True
    random_seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters against stability constraints."""
        # Grid size validation
        if self.grid_size < 4:
            raise ValueOutOfRangeError(
                "Grid size must be at least 4",
                value=float(self.grid_size),
                min_bound=4.0,
                parameter_name="grid_size",
            )

        # CFL stability condition check
        for name, value in [
            ("d_activator", self.d_activator),
            ("d_inhibitor", self.d_inhibitor),
            ("alpha", self.alpha),
        ]:
            if value >= MAX_STABLE_DIFFUSION:
                raise StabilityError(
                    f"Diffusion coefficient {name}={value} exceeds CFL stability "
                    f"limit of {MAX_STABLE_DIFFUSION}. Reduce to maintain numerical stability."
                )
            if value < 0:
                raise ValueOutOfRangeError(
                    "Diffusion coefficient must be non-negative",
                    value=value,
                    min_bound=0.0,
                    parameter_name=name,
                )

        # Reaction rate validation
        for name, value in [
            ("r_activator", self.r_activator),
            ("r_inhibitor", self.r_inhibitor),
        ]:
            if value < 0:
                raise ValueOutOfRangeError(
                    "Reaction rate must be non-negative",
                    value=value,
                    min_bound=0.0,
                    parameter_name=name,
                )

        # Threshold validation
        if not (0 <= self.turing_threshold <= 1):
            raise ValueOutOfRangeError(
                "Turing threshold must be in [0, 1]",
                value=self.turing_threshold,
                min_bound=0.0,
                max_bound=1.0,
                parameter_name="turing_threshold",
            )

        # Probability validation
        if not (0 <= self.spike_probability <= 1):
            raise ValueOutOfRangeError(
                "Spike probability must be in [0, 1]",
                value=self.spike_probability,
                min_bound=0.0,
                max_bound=1.0,
                parameter_name="spike_probability",
            )


@dataclass
class ReactionDiffusionMetrics:
    """
    Metrics collected during reaction-diffusion simulation.

    Attributes
    ----------
    field_min_v : float
        Minimum field value (V).
    field_max_v : float
        Maximum field value (V).
    field_mean_v : float
        Mean field value (V).
    field_std_v : float
        Standard deviation of field (V).
    activator_mean : float
        Mean activator concentration.
    inhibitor_mean : float
        Mean inhibitor concentration.
    steps_computed : int
        Number of simulation steps.
    growth_events : int
        Number of growth/spike events.
    turing_activations : int
        Number of Turing threshold crossings.
    nan_detected : bool
        Whether NaN was detected.
    inf_detected : bool
        Whether Inf was detected.
    clamping_events : int
        Number of clamping operations.
    steps_to_instability : int | None
        Step count where instability was first detected (if any).
    """

    field_min_v: float = 0.0
    field_max_v: float = 0.0
    field_mean_v: float = 0.0
    field_std_v: float = 0.0
    activator_mean: float = 0.0
    inhibitor_mean: float = 0.0
    steps_computed: int = 0
    growth_events: int = 0
    turing_activations: int = 0
    nan_detected: bool = False
    inf_detected: bool = False
    clamping_events: int = 0
    steps_to_instability: int | None = None


class ReactionDiffusionEngine:
    """
    Engine for Turing reaction-diffusion pattern formation.

    Implements activator-inhibitor dynamics with spatial diffusion
    on a 2D lattice. Uses explicit Euler integration with CFL-stable
    parameters.

    Reference: MATH_MODEL.md Section 2

    Example
    -------
    >>> config = ReactionDiffusionConfig(grid_size=64, random_seed=42)
    >>> engine = ReactionDiffusionEngine(config)
    >>> field, metrics = engine.simulate(steps=100)
    >>> print(f"Field range: [{field.min()*1000:.1f}, {field.max()*1000:.1f}] mV")
    """

    def __init__(self, config: ReactionDiffusionConfig | None = None) -> None:
        """
        Initialize reaction-diffusion engine.

        Parameters
        ----------
        config : ReactionDiffusionConfig | None
            Engine configuration. If None, uses defaults.
        """
        self.config = config or ReactionDiffusionConfig()
        self._metrics = ReactionDiffusionMetrics()
        self._rng = np.random.default_rng(self.config.random_seed)

        # Initialize fields
        self._field: NDArray[np.floating] | None = None
        self._activator: NDArray[np.floating] | None = None
        self._inhibitor: NDArray[np.floating] | None = None

    @property
    def metrics(self) -> ReactionDiffusionMetrics:
        """Get current metrics."""
        return self._metrics

    @property
    def field(self) -> NDArray[np.floating] | None:
        """Get current field state."""
        return self._field

    @property
    def activator(self) -> NDArray[np.floating] | None:
        """Get current activator state."""
        return self._activator

    @property
    def inhibitor(self) -> NDArray[np.floating] | None:
        """Get current inhibitor state."""
        return self._inhibitor

    def reset(self) -> None:
        """Reset engine state and metrics."""
        self._metrics = ReactionDiffusionMetrics()
        self._field = None
        self._activator = None
        self._inhibitor = None
        self._rng = np.random.default_rng(self.config.random_seed)

    def initialize_field(
        self,
        initial_potential_v: float = INITIAL_POTENTIAL_MEAN,
        initial_std_v: float = INITIAL_POTENTIAL_STD,
    ) -> NDArray[np.floating]:
        """
        Initialize potential field with Gaussian distribution.

        Reference: MATH_MODEL.md Section 2.6
        - Initial condition: V ~ N(-70 mV, 5 mV)

        Parameters
        ----------
        initial_potential_v : float
            Mean initial potential (V). Default -70 mV.
        initial_std_v : float
            Standard deviation (V). Default 5 mV.

        Returns
        -------
        NDArray
            Initialized field of shape (N, N).
        """
        n = self.config.grid_size
        self._field = self._rng.normal(
            loc=initial_potential_v,
            scale=initial_std_v,
            size=(n, n),
        ).astype(np.float64)

        # Initialize activator-inhibitor system
        self._activator = self._rng.uniform(0, 0.1, size=(n, n)).astype(np.float64)
        self._inhibitor = self._rng.uniform(0, 0.1, size=(n, n)).astype(np.float64)

        return self._field

    def simulate(
        self,
        steps: int,
        turing_enabled: bool = True,
        return_history: bool = False,
    ) -> tuple[NDArray[np.floating], ReactionDiffusionMetrics]:
        """
        Run reaction-diffusion simulation.

        Parameters
        ----------
        steps : int
            Number of simulation steps.
        turing_enabled : bool
            Enable Turing morphogenesis. Default True.
        return_history : bool
            If True, returns field history instead of final state.

        Returns
        -------
        tuple[NDArray, ReactionDiffusionMetrics]
            Final field (or history if return_history=True) and metrics.

        Raises
        ------
        NumericalInstabilityError
            If NaN or Inf values are detected during simulation.
        """
        self.reset()

        if self._field is None:
            self.initialize_field()

        assert self._field is not None
        assert self._activator is not None
        assert self._inhibitor is not None

        history = []

        for step in range(steps):
            self._simulation_step(step, turing_enabled)

            if return_history:
                history.append(self._field.copy())

            self._metrics.steps_computed += 1

        # Final metrics
        self._update_field_metrics()

        if return_history:
            return np.stack(history), self._metrics

        return self._field.copy(), self._metrics

    def _simulation_step(self, step: int, turing_enabled: bool) -> None:
        """
        Perform one simulation step.

        Includes:
        1. Growth events (spikes)
        2. Field diffusion
        3. Turing morphogenesis (if enabled)
        4. Quantum jitter (if enabled)
        5. Clamping
        6. Stability check
        """
        assert self._field is not None
        assert self._activator is not None
        assert self._inhibitor is not None

        # Growth events (spikes)
        if self._rng.random() < self.config.spike_probability:
            i = self._rng.integers(0, self.config.grid_size)
            j = self._rng.integers(0, self.config.grid_size)
            spike_magnitude = float(self._rng.normal(loc=0.02, scale=0.005))
            self._field[i, j] += spike_magnitude
            self._metrics.growth_events += 1

        # Field diffusion with Laplacian
        laplacian = self._compute_laplacian(self._field)
        self._field = self._field + self.config.alpha * laplacian

        # Turing morphogenesis
        if turing_enabled:
            self._turing_step()

        # Quantum jitter
        if self.config.quantum_jitter:
            jitter = self._rng.normal(
                0,
                np.sqrt(self.config.jitter_var),
                size=self._field.shape,
            )
            self._field = self._field + jitter

        # Clamping to physiological bounds
        field_before = self._field.copy()
        self._field = np.clip(self._field, FIELD_V_MIN, FIELD_V_MAX)
        clamped = np.sum(self._field != field_before)
        self._metrics.clamping_events += int(clamped)

        # Stability check
        if self.config.check_stability:
            self._check_stability(step)

    def _turing_step(self) -> None:
        """
        Perform Turing reaction-diffusion update.

        Reference: MATH_MODEL.md Section 2.2

        ∂a/∂t = D_a ∇²a + r_a * a(1-a) - i
        ∂i/∂t = D_i ∇²i + r_i * (a - i)
        """
        assert self._activator is not None
        assert self._inhibitor is not None
        assert self._field is not None

        # Compute Laplacians
        a_lap = self._compute_laplacian(self._activator)
        i_lap = self._compute_laplacian(self._inhibitor)

        # Reaction-diffusion update
        da = self.config.d_activator * a_lap
        da += self.config.r_activator * (self._activator * (1 - self._activator) - self._inhibitor)

        di = self.config.d_inhibitor * i_lap
        di += self.config.r_inhibitor * (self._activator - self._inhibitor)

        self._activator = self._activator + da
        self._inhibitor = self._inhibitor + di

        # Apply Turing pattern to field where activator exceeds threshold
        turing_mask = self._activator > self.config.turing_threshold
        activation_count = int(np.sum(turing_mask))
        if activation_count > 0:
            self._field[turing_mask] += 0.005
            self._metrics.turing_activations += activation_count

        # Clamp activator/inhibitor to [0, 1]
        self._activator = np.clip(self._activator, 0.0, 1.0)
        self._inhibitor = np.clip(self._inhibitor, 0.0, 1.0)

    def _compute_laplacian(
        self,
        field: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Compute discrete Laplacian using 5-point stencil.

        Reference: MATH_MODEL.md Section 2.4

        ∇²u ≈ u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]

        Parameters
        ----------
        field : NDArray
            Input field of shape (N, N).

        Returns
        -------
        NDArray
            Laplacian of shape (N, N).
        """
        if self.config.boundary_condition == BoundaryCondition.PERIODIC:
            up = np.roll(field, 1, axis=0)
            down = np.roll(field, -1, axis=0)
            left = np.roll(field, 1, axis=1)
            right = np.roll(field, -1, axis=1)
        elif self.config.boundary_condition == BoundaryCondition.NEUMANN:
            # Zero-flux (Neumann): copy boundary values from interior neighbors
            # This ensures dV/dn = 0 at boundaries, where n is the normal direction
            up = np.empty_like(field)
            up[1:, :] = field[:-1, :]
            up[0, :] = field[0, :]  # First row copies itself (zero gradient)
            
            down = np.empty_like(field)
            down[:-1, :] = field[1:, :]
            down[-1, :] = field[-1, :]  # Last row copies itself
            
            left = np.empty_like(field)
            left[:, 1:] = field[:, :-1]
            left[:, 0] = field[:, 0]  # First column copies itself
            
            right = np.empty_like(field)
            right[:, :-1] = field[:, 1:]
            right[:, -1] = field[:, -1]  # Last column copies itself
        else:  # DIRICHLET
            # Zero value at boundaries
            up = np.pad(field[1:, :], ((0, 1), (0, 0)), mode="constant")
            down = np.pad(field[:-1, :], ((1, 0), (0, 0)), mode="constant")
            left = np.pad(field[:, 1:], ((0, 0), (0, 1)), mode="constant")
            right = np.pad(field[:, :-1], ((0, 0), (1, 0)), mode="constant")

        return up + down + left + right - 4.0 * field

    def _check_stability(self, step: int) -> None:
        """Check for NaN/Inf values and raise error if found."""
        for name, arr in [
            ("field", self._field),
            ("activator", self._activator),
            ("inhibitor", self._inhibitor),
        ]:
            if arr is None:
                continue

            nan_count = int(np.sum(np.isnan(arr)))
            inf_count = int(np.sum(np.isinf(arr)))

            if nan_count > 0:
                self._metrics.nan_detected = True
                if self._metrics.steps_to_instability is None:
                    self._metrics.steps_to_instability = step
                raise NumericalInstabilityError(
                    f"NaN values detected in {name}",
                    step=step,
                    field_name=name,
                    nan_count=nan_count,
                )

            if inf_count > 0:
                self._metrics.inf_detected = True
                if self._metrics.steps_to_instability is None:
                    self._metrics.steps_to_instability = step
                raise NumericalInstabilityError(
                    f"Inf values detected in {name}",
                    step=step,
                    field_name=name,
                    inf_count=inf_count,
                )

    def _update_field_metrics(self) -> None:
        """Update field statistics in metrics."""
        if self._field is not None:
            self._metrics.field_min_v = float(np.min(self._field))
            self._metrics.field_max_v = float(np.max(self._field))
            self._metrics.field_mean_v = float(np.mean(self._field))
            self._metrics.field_std_v = float(np.std(self._field))

        if self._activator is not None:
            self._metrics.activator_mean = float(np.mean(self._activator))

        if self._inhibitor is not None:
            self._metrics.inhibitor_mean = float(np.mean(self._inhibitor))

    def validate_cfl_condition(self) -> bool:
        """
        Validate CFL stability condition for current parameters.

        Reference: MATH_MODEL.md Section 2.5
        dt * D * 4/dx² ≤ 1, with dt=dx=1 → D ≤ 0.25

        Returns
        -------
        bool
            True if parameters satisfy CFL condition.
        """
        max_d = max(
            self.config.d_activator,
            self.config.d_inhibitor,
            self.config.alpha,
        )
        return max_d < MAX_STABLE_DIFFUSION
