"""
Reaction-Diffusion Engine — Turing Morphogenesis.

Implements stable numerical schemes for Turing pattern formation:
- Activator-inhibitor reaction-diffusion PDEs
- Discrete Laplacian with periodic boundaries
- CFL stability condition enforcement

Reference: MFN_MATH_MODEL.md Section 2 (Reaction-Diffusion Processes)

Equations Implemented:
    ∂a/∂t = D_a ∇²a + r_a * a(1-a) - i     # Activator
    ∂i/∂t = D_i ∇²i + r_i * (a - i)         # Inhibitor

    ∇²u ≈ u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]

Parameters (from MFN_MATH_MODEL.md Section 2.3):
    D_a = 0.1         - Activator diffusion (grid²/step)
    D_i = 0.05        - Inhibitor diffusion (grid²/step)
    r_a = 0.01        - Activator reaction rate (1/step)
    r_i = 0.02        - Inhibitor reaction rate (1/step)
    θ = 0.75          - Turing activation threshold

Stability Constraint (MFN_MATH_MODEL.md Section 2.5):
    dt * D * 4/dx² ≤ 1
    With dx=1, dt=1: D_max = 0.25
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .exceptions import NumericalInstabilityError, StabilityError, ValueOutOfRangeError

# === Default Parameters (from MFN_MATH_MODEL.md Section 2.3) ===
DEFAULT_D_ACTIVATOR: float = 0.1
DEFAULT_D_INHIBITOR: float = 0.05
DEFAULT_R_ACTIVATOR: float = 0.01
DEFAULT_R_INHIBITOR: float = 0.02
DEFAULT_TURING_THRESHOLD: float = 0.75
DEFAULT_FIELD_ALPHA: float = 0.18
DEFAULT_QUANTUM_JITTER_VAR: float = 0.0005

# === Biophysical Parameter Bounds (from MFN_MATH_MODEL.md Section 2.3) ===
# Diffusion coefficient bounds
D_ACTIVATOR_MIN: float = 0.01   # grid²/step - short-range diffusion lower bound
D_ACTIVATOR_MAX: float = 0.5    # grid²/step - should not exceed 2x CFL limit
D_INHIBITOR_MIN: float = 0.01   # grid²/step - long-range diffusion lower bound
D_INHIBITOR_MAX: float = 0.3    # grid²/step - should be bounded below activator typical max

# Reaction rate bounds
R_ACTIVATOR_MIN: float = 0.001  # 1/step - minimum meaningful growth rate
R_ACTIVATOR_MAX: float = 0.1    # 1/step - upper bound for stability
R_INHIBITOR_MIN: float = 0.001  # 1/step - minimum damping rate
R_INHIBITOR_MAX: float = 0.1    # 1/step - upper bound for stability

# Turing threshold bounds
TURING_THRESHOLD_MIN: float = 0.5   # Below this, patterns trigger too easily
TURING_THRESHOLD_MAX: float = 0.95  # Above this, patterns rarely form

# Field diffusion coefficient bounds
ALPHA_MIN: float = 0.05   # Minimum for observable diffusion
ALPHA_MAX: float = 0.25   # CFL limit (dt=dx=1)

# Jitter variance bounds
JITTER_VAR_MIN: float = 0.0       # No jitter
JITTER_VAR_MAX: float = 0.01      # Upper limit for stability

# Grid size bounds
GRID_SIZE_MIN: int = 4    # Minimum for meaningful patterns
GRID_SIZE_MAX: int = 1024 # Upper limit for computational feasibility

# === Stability Limits ===
# CFL condition: D * dt * 4/dx² ≤ 1 → D ≤ 0.25 for dt=dx=1
MAX_STABLE_DIFFUSION: float = 0.25

# === Field Bounds (MFN_MATH_MODEL.md Section 4.3) ===
FIELD_V_MIN: float = -0.095  # -95 mV - hyperpolarization limit
FIELD_V_MAX: float = 0.040   # +40 mV - action potential peak
INITIAL_POTENTIAL_MEAN: float = -0.070  # -70 mV (resting potential)
INITIAL_POTENTIAL_STD: float = 0.005    # 5 mV initial variance


class BoundaryCondition(Enum):
    """Available boundary conditions for the spatial grid."""

    PERIODIC = "periodic"  # Wrap around (np.roll)
    NEUMANN = "neumann"  # Zero-flux (mirror at boundary)
    DIRICHLET = "dirichlet"  # Fixed value at boundary


def _validate_diffusion_coefficient(
    name: str,
    value: float,
    min_bound: float,
    cfl_limit: float = MAX_STABLE_DIFFUSION,
) -> None:
    """Validate diffusion coefficient against biophysical and CFL bounds.
    
    Parameters
    ----------
    name : str
        Parameter name for error messages.
    value : float
        Diffusion coefficient value.
    min_bound : float
        Minimum allowed value.
    cfl_limit : float
        CFL stability limit (default 0.25).
        
    Raises
    ------
    StabilityError
        If value exceeds CFL limit.
    ValueOutOfRangeError
        If value is below minimum.
    """
    if value > cfl_limit:
        raise StabilityError(
            f"{name}={value} exceeds CFL stability limit "
            f"of {cfl_limit}. Reduce to maintain numerical stability."
        )
    if value < min_bound:
        raise ValueOutOfRangeError(
            f"{name} must be in [{min_bound}, {cfl_limit})",
            value=value,
            min_bound=min_bound,
            max_bound=cfl_limit,
            parameter_name=name,
        )


@dataclass
class ReactionDiffusionConfig:
    """
    Configuration for reaction-diffusion engine.

    All parameters have physically meaningful defaults from MFN_MATH_MODEL.md.

    Attributes
    ----------
    grid_size : int
        Size of the square grid (N x N). Default 64.
    d_activator : float
        Activator diffusion coefficient. Default 0.1.
        Valid range: 0.01-0.5, must be ≤ 0.25 for stability.
    d_inhibitor : float
        Inhibitor diffusion coefficient. Default 0.05.
        Valid range: 0.01-0.3, must be ≤ 0.25 for stability.
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
        Valid range: 0.05-0.25 (must be ≤ 0.25 for stability).
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
        """Validate configuration parameters against biophysical and stability constraints.

        Invariants enforced:
        - Grid size: [4, 1024] for computational feasibility
        - Diffusion coefficients: ≤ 0.25 for CFL stability, within biophysical ranges
        - Reaction rates: [0.001, 0.1] for stable pattern formation
        - Turing threshold: [0.5, 0.95] for meaningful pattern activation
        - Jitter variance: [0, 0.01] for stability
        """
        # Grid size validation with biophysical bounds
        if not (GRID_SIZE_MIN <= self.grid_size <= GRID_SIZE_MAX):
            raise ValueOutOfRangeError(
                f"Grid size must be in [{GRID_SIZE_MIN}, {GRID_SIZE_MAX}]",
                value=float(self.grid_size),
                min_bound=float(GRID_SIZE_MIN),
                max_bound=float(GRID_SIZE_MAX),
                parameter_name="grid_size",
            )

        # Diffusion coefficient validation using helper
        _validate_diffusion_coefficient("d_activator", self.d_activator, D_ACTIVATOR_MIN)
        _validate_diffusion_coefficient("d_inhibitor", self.d_inhibitor, D_INHIBITOR_MIN)
        _validate_diffusion_coefficient("alpha", self.alpha, ALPHA_MIN)

        # Reaction rate validation with biophysical bounds
        if not (R_ACTIVATOR_MIN <= self.r_activator <= R_ACTIVATOR_MAX):
            raise ValueOutOfRangeError(
                f"r_activator must be in [{R_ACTIVATOR_MIN}, {R_ACTIVATOR_MAX}] "
                "for stable pattern formation",
                value=self.r_activator,
                min_bound=R_ACTIVATOR_MIN,
                max_bound=R_ACTIVATOR_MAX,
                parameter_name="r_activator",
            )

        if not (R_INHIBITOR_MIN <= self.r_inhibitor <= R_INHIBITOR_MAX):
            raise ValueOutOfRangeError(
                f"r_inhibitor must be in [{R_INHIBITOR_MIN}, {R_INHIBITOR_MAX}] "
                "for stable damping",
                value=self.r_inhibitor,
                min_bound=R_INHIBITOR_MIN,
                max_bound=R_INHIBITOR_MAX,
                parameter_name="r_inhibitor",
            )

        # Turing threshold validation with biophysical bounds
        if not (TURING_THRESHOLD_MIN <= self.turing_threshold <= TURING_THRESHOLD_MAX):
            raise ValueOutOfRangeError(
                f"turing_threshold must be in [{TURING_THRESHOLD_MIN}, {TURING_THRESHOLD_MAX}] "
                "for meaningful pattern activation",
                value=self.turing_threshold,
                min_bound=TURING_THRESHOLD_MIN,
                max_bound=TURING_THRESHOLD_MAX,
                parameter_name="turing_threshold",
            )

        # Spike probability validation
        if not (0 <= self.spike_probability <= 1):
            raise ValueOutOfRangeError(
                "Spike probability must be in [0, 1]",
                value=self.spike_probability,
                min_bound=0.0,
                max_bound=1.0,
                parameter_name="spike_probability",
            )

        # Jitter variance validation
        if not (JITTER_VAR_MIN <= self.jitter_var <= JITTER_VAR_MAX):
            raise ValueOutOfRangeError(
                f"jitter_var must be in [{JITTER_VAR_MIN}, {JITTER_VAR_MAX}] for stability",
                value=self.jitter_var,
                min_bound=JITTER_VAR_MIN,
                max_bound=JITTER_VAR_MAX,
                parameter_name="jitter_var",
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

    Reference: MFN_MATH_MODEL.md Section 2

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

        Reference: MFN_MATH_MODEL.md Section 2.6
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
        if steps < 1:
            raise ValueError("steps must be at least 1")

        self.reset()

        if self._field is None:
            self.initialize_field()

        assert self._field is not None
        assert self._activator is not None
        assert self._inhibitor is not None

        history: NDArray[np.floating] | None = None
        if return_history:
            history = np.empty(
                (steps, self.config.grid_size, self.config.grid_size),
                dtype=self._field.dtype,
            )

        for step in range(steps):
            self._simulation_step(step, turing_enabled)

            if return_history:
                assert history is not None
                history[step] = self._field

            self._metrics.steps_computed += 1

        # Final metrics
        self._update_field_metrics()

        if return_history:
            assert history is not None
            return history, self._metrics

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
        clamped_mask = (self._field > FIELD_V_MAX) | (self._field < FIELD_V_MIN)
        clamped = int(np.count_nonzero(clamped_mask))
        if clamped:
            np.clip(self._field, FIELD_V_MIN, FIELD_V_MAX, out=self._field)
        self._metrics.clamping_events += clamped

        # Stability check
        if self.config.check_stability:
            self._check_stability(step)

    def _turing_step(self) -> None:
        """
        Perform Turing reaction-diffusion update.

        Reference: MFN_MATH_MODEL.md Section 2.2

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

        Reference: MFN_MATH_MODEL.md Section 2.4

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
            up = np.pad(field[:-1, :], ((1, 0), (0, 0)), mode="constant")
            down = np.pad(field[1:, :], ((0, 1), (0, 0)), mode="constant")
            left = np.pad(field[:, :-1], ((0, 0), (1, 0)), mode="constant")
            right = np.pad(field[:, 1:], ((0, 0), (0, 1)), mode="constant")

        return cast(NDArray[np.floating[Any]], up + down + left + right - 4.0 * field)

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

        Reference: MFN_MATH_MODEL.md Section 2.5
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
        return max_d <= MAX_STABLE_DIFFUSION
