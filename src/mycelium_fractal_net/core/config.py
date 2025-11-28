"""
Configuration dataclasses for numerical engines.

Each configuration class encapsulates all parameters needed for an engine,
with explicit default values, stability constraints, and documentation.

Reference: docs/ARCHITECTURE.md for parameter derivations and physics context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BoundaryCondition(Enum):
    """Boundary condition types for spatial discretization."""

    PERIODIC = "periodic"
    NEUMANN = "neumann"  # Zero-flux (∂u/∂n = 0)
    DIRICHLET = "dirichlet"  # Fixed value


class IntegrationScheme(Enum):
    """Time integration schemes."""

    EULER = "euler"  # Forward Euler (1st order)
    RK4 = "rk4"  # Runge-Kutta 4th order


@dataclass
class MembraneEngineConfig:
    """
    Configuration for membrane potential dynamics engine.

    Implements Nernst-Planck ion dynamics with configurable integration.

    Reference: docs/ARCHITECTURE.md Section 1 (Nernst Equation)

    Parameters:
        dt: Time step in seconds. Must satisfy dt < tau/2 for stability.
        v_rest: Resting membrane potential in volts. Default: -0.070 V (-70 mV).
        v_min: Minimum potential (hyperpolarization floor). Default: -0.095 V (-95 mV).
        v_max: Maximum potential (action potential ceiling). Default: 0.040 V (+40 mV).
        tau: Membrane time constant in seconds. Default: 0.010 s (10 ms).
        temperature_k: Temperature in Kelvin. Default: 310 K (37°C).
        integration_scheme: Numerical integration method. Default: EULER.
        check_stability: Enable NaN/Inf and range checks. Default: True.
        ion_clamp_min: Minimum ion concentration for Nernst (prevents log(0)).
        random_seed: Seed for reproducibility. Default: None (random).

    Stability Conditions:
        - Explicit Euler: dt < tau (CFL-like condition)
        - RK4: dt < 2*tau (more permissive)
    """

    dt: float = 0.001  # 1 ms time step
    v_rest: float = -0.070  # -70 mV in volts
    v_min: float = -0.095  # -95 mV
    v_max: float = 0.040  # +40 mV
    tau: float = 0.010  # 10 ms membrane time constant
    temperature_k: float = 310.0  # 37°C
    integration_scheme: IntegrationScheme = IntegrationScheme.EULER
    check_stability: bool = True
    ion_clamp_min: float = 1e-6  # Minimum ion concentration (M)
    random_seed: int | None = None

    def validate(self) -> None:
        """Validate configuration parameters for stability."""
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.tau <= 0:
            raise ValueError(f"tau must be positive, got {self.tau}")
        if self.v_min >= self.v_max:
            raise ValueError(f"v_min ({self.v_min}) must be < v_max ({self.v_max})")

        # Check CFL-like stability condition
        if self.integration_scheme == IntegrationScheme.EULER:
            if self.dt >= self.tau:
                raise ValueError(
                    f"For Euler, dt ({self.dt}) must be < tau ({self.tau}) for stability"
                )
        elif self.integration_scheme == IntegrationScheme.RK4:
            if self.dt >= 2 * self.tau:
                raise ValueError(
                    f"For RK4, dt ({self.dt}) must be < 2*tau ({2*self.tau}) for stability"
                )


@dataclass
class ReactionDiffusionConfig:
    """
    Configuration for reaction-diffusion (Turing morphogenesis) engine.

    Implements activator-inhibitor reaction-diffusion system:
        ∂a/∂t = D_a ∇²a + r_a·a(1-a) - i
        ∂i/∂t = D_i ∇²i + r_i·(a - i)

    Reference: docs/ARCHITECTURE.md Section 2 (Turing Morphogenesis)

    Parameters:
        grid_size: Spatial grid size (N×N). Must be >= 4.
        dt: Time step. Default: 0.1 (dimensionless).
        steps: Number of integration steps. Default: 100.
        d_activator: Activator diffusion coefficient. Default: 0.1.
        d_inhibitor: Inhibitor diffusion coefficient. Default: 0.05.
        r_activator: Activator reaction rate. Default: 0.01.
        r_inhibitor: Inhibitor reaction rate. Default: 0.02.
        turing_threshold: Threshold for pattern activation. Default: 0.75.
        boundary: Boundary condition type. Default: PERIODIC.
        check_stability: Enable stability checks. Default: True.
        random_seed: Seed for reproducibility.

    Stability Conditions (CFL):
        For explicit scheme: dt ≤ dx² / (4·max(D_a, D_i))
        where dx = 1.0 (unit grid spacing).
    """

    grid_size: int = 64
    dt: float = 0.1
    steps: int = 100
    d_activator: float = 0.1
    d_inhibitor: float = 0.05
    r_activator: float = 0.01
    r_inhibitor: float = 0.02
    turing_threshold: float = 0.75
    boundary: BoundaryCondition = BoundaryCondition.PERIODIC
    check_stability: bool = True
    random_seed: int | None = None

    def validate(self) -> None:
        """Validate configuration parameters for stability."""
        if self.grid_size < 4:
            raise ValueError(f"grid_size must be >= 4, got {self.grid_size}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if self.d_activator < 0 or self.d_inhibitor < 0:
            raise ValueError("Diffusion coefficients must be non-negative")

        # CFL stability condition for 2D explicit diffusion
        dx = 1.0  # Unit grid spacing
        d_max = max(self.d_activator, self.d_inhibitor)
        if d_max > 0:
            dt_max = dx * dx / (4.0 * d_max)
            if self.dt > dt_max:
                raise ValueError(
                    f"CFL violation: dt ({self.dt}) > dx²/(4·D_max) = {dt_max:.4f}. "
                    f"Reduce dt or diffusion coefficients for stability."
                )

    @property
    def cfl_condition(self) -> float:
        """Compute CFL number for current parameters."""
        dx = 1.0
        d_max = max(self.d_activator, self.d_inhibitor)
        if d_max == 0:
            return 0.0
        return self.dt * d_max / (dx * dx)


@dataclass
class FractalGrowthConfig:
    """
    Configuration for fractal growth engine (IFS/DLA model).

    Implements Iterated Function System (IFS) with stochastic affine maps
    and optional Diffusion-Limited Aggregation (DLA) growth.

    Reference: docs/ARCHITECTURE.md Section 3 (Fractal Analysis)

    Parameters:
        num_points: Number of IFS iteration points. Default: 10000.
        num_transforms: Number of affine transformations. Default: 4.
        contraction_min: Minimum contraction factor. Default: 0.2.
        contraction_max: Maximum contraction factor. Default: 0.5.
        max_iterations: Maximum DLA iterations. Default: 1000.
        grid_size: Grid size for DLA lattice. Default: 64.
        dla_enabled: Enable DLA growth mode. Default: False (IFS only).
        check_stability: Enable stability checks. Default: True.
        random_seed: Seed for reproducibility (REQUIRED for determinism).

    Stability Conditions:
        - contraction_max < 1.0 ensures IFS convergence (λ < 0)
        - Lyapunov exponent should be negative for stable dynamics
    """

    num_points: int = 10000
    num_transforms: int = 4
    contraction_min: float = 0.2
    contraction_max: float = 0.5
    max_iterations: int = 1000
    grid_size: int = 64
    dla_enabled: bool = False
    check_stability: bool = True
    random_seed: int | None = None

    # Box-counting dimension estimation parameters
    box_min_size: int = 2
    box_num_scales: int = 5

    def validate(self) -> None:
        """Validate configuration parameters for stability."""
        if self.num_points < 1:
            raise ValueError(f"num_points must be >= 1, got {self.num_points}")
        if self.num_transforms < 1:
            raise ValueError(f"num_transforms must be >= 1, got {self.num_transforms}")
        if not (0 < self.contraction_min < self.contraction_max < 1.0):
            raise ValueError(
                f"Need 0 < contraction_min < contraction_max < 1.0, "
                f"got [{self.contraction_min}, {self.contraction_max}]"
            )
        if self.grid_size < 4:
            raise ValueError(f"grid_size must be >= 4, got {self.grid_size}")


@dataclass
class EngineMetricsBase:
    """Base class for engine metrics collection."""

    steps_completed: int = 0
    nan_detected: bool = False
    inf_detected: bool = False
    values_clamped: int = 0
    execution_time_s: float = 0.0


@dataclass
class CombinedConfig:
    """
    Combined configuration for all engines.

    Useful for loading from JSON/YAML config files.
    """

    membrane: MembraneEngineConfig = field(default_factory=MembraneEngineConfig)
    reaction_diffusion: ReactionDiffusionConfig = field(default_factory=ReactionDiffusionConfig)
    fractal: FractalGrowthConfig = field(default_factory=FractalGrowthConfig)

    def validate_all(self) -> None:
        """Validate all sub-configurations."""
        self.membrane.validate()
        self.reaction_diffusion.validate()
        self.fractal.validate()
