"""
Discretized Update Rules for MyceliumFractalNet Simulations.

Implements explicit time-stepping schemes for:
- Membrane potential diffusion
- Activator-inhibitor reaction-diffusion (Turing morphogenesis)
- Growth events and stochastic perturbations

Reference: MFN_MATH_MODEL.md Section 2 (Reaction-Diffusion Processes)

Mathematical Models:
    
    1. Membrane Potential Field Evolution (MFN_MATH_MODEL.md Section 2.6):
        V^{n+1}_{i,j} = V^n_{i,j} + α·∇²V^n_{i,j} + growth + Turing
        
    2. Activator-Inhibitor System (MFN_MATH_MODEL.md Section 2.2):
        ∂a/∂t = D_a·∇²a + r_a·a(1-a) - i
        ∂i/∂t = D_i·∇²i + r_i·(a - i)

Discretization Scheme:
    - Temporal: Explicit Euler (first-order)
    - Spatial: 5-point stencil Laplacian (second-order)
    - Unit time step: dt = 1
    - Unit grid spacing: dx = 1

Stability Conditions:
    - CFL: D_max < 0.25 (for dt=dx=1)
    - Reaction rates: r < 0.1 (empirically stable)
    
Parameter Defaults (from MFN_MATH_MODEL.md Section 2.3):
    - D_a = 0.1 (activator diffusion)
    - D_i = 0.05 (inhibitor diffusion)
    - r_a = 0.01 (activator reaction)
    - r_i = 0.02 (inhibitor reaction)
    - α = 0.18 (field diffusion)
    - θ = 0.75 (Turing threshold)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from mycelium_fractal_net.core.exceptions import (
    StabilityError,
    ValueOutOfRangeError,
)

from .grid_ops import BoundaryCondition, compute_laplacian, validate_field_stability

# === Model Parameters (from MFN_MATH_MODEL.md Section 2.3) ===
DEFAULT_D_ACTIVATOR: float = 0.1    # Activator diffusion (grid²/step)
DEFAULT_D_INHIBITOR: float = 0.05   # Inhibitor diffusion (grid²/step)
DEFAULT_R_ACTIVATOR: float = 0.01   # Activator reaction rate (1/step)
DEFAULT_R_INHIBITOR: float = 0.02   # Inhibitor reaction rate (1/step)
DEFAULT_ALPHA: float = 0.18         # Field diffusion coefficient
DEFAULT_TURING_THRESHOLD: float = 0.75  # Pattern activation threshold

# === Stability Limits ===
# CFL condition: dt * D * 4/dx² ≤ 1, with dt=dx=1 → D_max = 0.25
MAX_STABLE_DIFFUSION: float = 0.25

# === Field Bounds (from MFN_MATH_MODEL.md Section 4.3) ===
FIELD_V_MIN: float = -0.095  # -95 mV (clamping minimum)
FIELD_V_MAX: float = 0.040   # +40 mV (clamping maximum)

# === Physiological Constants ===
RESTING_POTENTIAL_V: float = -0.070  # -70 mV typical resting potential
SPIKE_MAGNITUDE_V: float = 0.020     # +20 mV typical spike amplitude


@dataclass
class UpdateParameters:
    """
    Parameters for field update rules.
    
    All values correspond to MFN_MATH_MODEL.md Section 2.3 unless noted.
    
    Attributes
    ----------
    d_activator : float
        Activator diffusion coefficient D_a. Default 0.1.
        Valid range: [0.01, 0.25). CFL constraint: < 0.25.
    d_inhibitor : float
        Inhibitor diffusion coefficient D_i. Default 0.05.
        Valid range: [0.01, 0.25). CFL constraint: < 0.25.
    r_activator : float
        Activator reaction rate r_a. Default 0.01.
        Valid range: [0.001, 0.1].
    r_inhibitor : float
        Inhibitor reaction rate r_i. Default 0.02.
        Valid range: [0.001, 0.1].
    alpha : float
        Field diffusion coefficient α. Default 0.18.
        Valid range: [0.05, 0.25). CFL constraint: < 0.25.
    turing_threshold : float
        Threshold θ for Turing pattern activation. Default 0.75.
        Valid range: [0.5, 0.95].
    turing_contribution_v : float
        Potential added where activator > threshold. Default 0.005 V.
    boundary : BoundaryCondition
        Boundary condition for Laplacian. Default PERIODIC.
    """
    d_activator: float = DEFAULT_D_ACTIVATOR
    d_inhibitor: float = DEFAULT_D_INHIBITOR
    r_activator: float = DEFAULT_R_ACTIVATOR
    r_inhibitor: float = DEFAULT_R_INHIBITOR
    alpha: float = DEFAULT_ALPHA
    turing_threshold: float = DEFAULT_TURING_THRESHOLD
    turing_contribution_v: float = 0.005
    boundary: BoundaryCondition = BoundaryCondition.PERIODIC
    
    def __post_init__(self) -> None:
        """Validate parameters against stability constraints."""
        # CFL stability check
        for name, value in [
            ("d_activator", self.d_activator),
            ("d_inhibitor", self.d_inhibitor),
            ("alpha", self.alpha),
        ]:
            if value >= MAX_STABLE_DIFFUSION:
                raise StabilityError(
                    f"Diffusion coefficient {name}={value} violates CFL stability "
                    f"condition (must be < {MAX_STABLE_DIFFUSION})"
                )
            if value < 0:
                raise ValueOutOfRangeError(
                    f"Diffusion coefficient {name} must be non-negative",
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
                    f"Reaction rate {name} must be non-negative",
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


def diffusion_update(
    field: NDArray[np.floating],
    diffusion_coeff: float,
    boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
    check_stability: bool = True,
) -> NDArray[np.floating]:
    """
    Apply one explicit Euler diffusion step.
    
    Implements discretization from MFN_MATH_MODEL.md Section 2.6:
        V^{n+1} = V^n + D·∇²V^n
    
    Discretization scheme:
        - Explicit Euler (first-order temporal)
        - 5-point stencil Laplacian (second-order spatial)
        - Time step dt = 1 (implicit in coefficient)
    
    Stability condition:
        D < 0.25 for unit grid spacing (MFN_MATH_MODEL.md Section 2.5)
    
    Parameters
    ----------
    field : NDArray[np.floating]
        Current field state V^n of shape (N, M).
    diffusion_coeff : float
        Diffusion coefficient D. Must be < 0.25 for stability.
    boundary : BoundaryCondition
        Boundary condition for Laplacian.
    check_stability : bool
        Whether to check for NaN/Inf.
    
    Returns
    -------
    NDArray[np.floating]
        Updated field V^{n+1} of shape (N, M).
    
    Raises
    ------
    StabilityError
        If diffusion_coeff >= 0.25 (CFL violation).
    NumericalInstabilityError
        If NaN/Inf detected in result.
    
    Examples
    --------
    >>> import numpy as np
    >>> field = np.random.randn(64, 64) * 0.01
    >>> field_new = diffusion_update(field, diffusion_coeff=0.18)
    """
    if diffusion_coeff >= MAX_STABLE_DIFFUSION:
        raise StabilityError(
            f"Diffusion coefficient {diffusion_coeff} violates CFL stability "
            f"condition (must be < {MAX_STABLE_DIFFUSION})"
        )
    
    laplacian = compute_laplacian(field, boundary, check_stability=check_stability)
    updated = field + diffusion_coeff * laplacian
    
    if check_stability:
        validate_field_stability(updated, field_name="diffusion_update")
    
    return cast(NDArray[np.floating[Any]], updated)


def activator_inhibitor_update(
    activator: NDArray[np.floating],
    inhibitor: NDArray[np.floating],
    params: UpdateParameters | None = None,
    check_stability: bool = True,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Apply one Turing reaction-diffusion update step.
    
    Implements discretization from MFN_MATH_MODEL.md Section 2.2:
        a^{n+1} = a^n + D_a·∇²a^n + r_a·a^n(1-a^n) - i^n
        i^{n+1} = i^n + D_i·∇²i^n + r_i·(a^n - i^n)
    
    Discretization scheme:
        - Explicit Euler (first-order temporal)
        - 5-point stencil Laplacian (second-order spatial)
        - Clamping to [0, 1] after update (MFN_MATH_MODEL.md Section 4.3)
    
    Stability conditions:
        - D_a, D_i < 0.25 (CFL condition)
        - D_i > D_a typically required for Turing instability
    
    Parameters
    ----------
    activator : NDArray[np.floating]
        Current activator field a^n of shape (N, M).
    inhibitor : NDArray[np.floating]
        Current inhibitor field i^n of shape (N, M).
    params : UpdateParameters | None
        Update parameters. Uses defaults if None.
    check_stability : bool
        Whether to check for NaN/Inf.
    
    Returns
    -------
    tuple[NDArray, NDArray]
        Updated (activator, inhibitor) fields, clamped to [0, 1].
    
    Raises
    ------
    NumericalInstabilityError
        If NaN/Inf detected in result.
    """
    if params is None:
        params = UpdateParameters()
    
    # Compute Laplacians
    a_lap = compute_laplacian(activator, params.boundary, check_stability=check_stability)
    i_lap = compute_laplacian(inhibitor, params.boundary, check_stability=check_stability)
    
    # Reaction-diffusion update
    # ∂a/∂t = D_a·∇²a + r_a·a(1-a) - i
    da = params.d_activator * a_lap + params.r_activator * (activator * (1 - activator) - inhibitor)
    
    # ∂i/∂t = D_i·∇²i + r_i·(a - i)
    di = params.d_inhibitor * i_lap + params.r_inhibitor * (activator - inhibitor)
    
    # Explicit Euler step
    activator_new = activator + da
    inhibitor_new = inhibitor + di
    
    # Clamp to [0, 1] (MFN_MATH_MODEL.md Section 4.3)
    activator_new = np.clip(activator_new, 0.0, 1.0)
    inhibitor_new = np.clip(inhibitor_new, 0.0, 1.0)
    
    if check_stability:
        validate_field_stability(activator_new, field_name="activator")
        validate_field_stability(inhibitor_new, field_name="inhibitor")
    
    return (
        cast(NDArray[np.floating[Any]], activator_new),
        cast(NDArray[np.floating[Any]], inhibitor_new),
    )


def apply_turing_to_field(
    field: NDArray[np.floating],
    activator: NDArray[np.floating],
    threshold: float = DEFAULT_TURING_THRESHOLD,
    contribution_v: float = 0.005,
) -> tuple[NDArray[np.floating], int]:
    """
    Apply Turing pattern modulation to potential field.
    
    Implements modulation from MFN_MATH_MODEL.md Section 2.6:
        V += contribution where a > threshold
    
    Parameters
    ----------
    field : NDArray[np.floating]
        Current potential field V of shape (N, M).
    activator : NDArray[np.floating]
        Activator field a of shape (N, M).
    threshold : float
        Turing activation threshold θ. Default 0.75.
    contribution_v : float
        Potential contribution in Volts. Default 0.005 V (+5 mV).
    
    Returns
    -------
    tuple[NDArray, int]
        Updated field and count of activation events.
    """
    turing_mask = activator > threshold
    activation_count = int(np.sum(turing_mask))
    
    field_new = field.copy()
    if activation_count > 0:
        field_new[turing_mask] += contribution_v
    
    return cast(NDArray[np.floating[Any]], field_new), activation_count


def apply_growth_event(
    field: NDArray[np.floating],
    rng: np.random.Generator,
    grid_size: int,
    magnitude_mean_v: float = SPIKE_MAGNITUDE_V,
    magnitude_std_v: float = 0.005,
) -> tuple[NDArray[np.floating], bool]:
    """
    Apply single growth event (spike) at random location.
    
    Implements growth events from MFN_MATH_MODEL.md Section 2.6:
        Growth events: Random spikes adding ~20 mV
    
    Parameters
    ----------
    field : NDArray[np.floating]
        Current potential field of shape (N, M).
    rng : np.random.Generator
        Random number generator.
    grid_size : int
        Grid dimension for coordinate sampling.
    magnitude_mean_v : float
        Mean spike magnitude. Default 0.020 V (+20 mV).
    magnitude_std_v : float
        Spike magnitude std. Default 0.005 V (5 mV).
    
    Returns
    -------
    tuple[NDArray, bool]
        Updated field and whether event occurred (always True here).
    """
    i = int(rng.integers(0, grid_size))
    j = int(rng.integers(0, grid_size))
    
    spike_magnitude = float(rng.normal(magnitude_mean_v, magnitude_std_v))
    
    field_new = field.copy()
    field_new[i, j] += spike_magnitude
    
    return cast(NDArray[np.floating[Any]], field_new), True


def apply_quantum_jitter(
    field: NDArray[np.floating],
    rng: np.random.Generator,
    variance: float = 0.0005,
) -> NDArray[np.floating]:
    """
    Apply stochastic noise term (quantum jitter).
    
    Implements optional stochastic term from MFN_MATH_MODEL.md Section 2.8:
        V^{n+1} = V^n + ... + ξ
        where ξ ~ N(0, σ²) with σ² = 0.0005
    
    Note: "Quantum" is a metaphor; this is Gaussian noise, not quantum effects.
    See MFN_MATH_MODEL.md Section 6 (Hypothesis vs. Established Theory).
    
    Parameters
    ----------
    field : NDArray[np.floating]
        Current field of shape (N, M).
    rng : np.random.Generator
        Random number generator.
    variance : float
        Noise variance σ². Default 0.0005.
    
    Returns
    -------
    NDArray[np.floating]
        Field with added noise.
    """
    noise = rng.normal(0, np.sqrt(variance), size=field.shape)
    return cast(NDArray[np.floating[Any]], field + noise)


def clamp_potential_field(
    field: NDArray[np.floating],
    v_min: float = FIELD_V_MIN,
    v_max: float = FIELD_V_MAX,
) -> tuple[NDArray[np.floating], int]:
    """
    Clamp potential field to physiological bounds.
    
    Implements clamping from MFN_MATH_MODEL.md Section 4.3:
        V ∈ [-95, 40] mV (enforced by clamping)
    
    Parameters
    ----------
    field : NDArray[np.floating]
        Potential field in Volts.
    v_min : float
        Minimum potential. Default -0.095 V (-95 mV).
    v_max : float
        Maximum potential. Default 0.040 V (+40 mV).
    
    Returns
    -------
    tuple[NDArray, int]
        Clamped field and count of clamped values.
    """
    needs_clamping = (field < v_min) | (field > v_max)
    clamp_count = int(np.sum(needs_clamping))
    
    clamped = np.clip(field, v_min, v_max)
    
    return cast(NDArray[np.floating[Any]], clamped), clamp_count


def full_simulation_step(
    field: NDArray[np.floating],
    activator: NDArray[np.floating],
    inhibitor: NDArray[np.floating],
    rng: np.random.Generator,
    params: UpdateParameters | None = None,
    turing_enabled: bool = True,
    quantum_jitter: bool = False,
    jitter_var: float = 0.0005,
    spike_probability: float = 0.25,
    check_stability: bool = True,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    dict[str, Any],
]:
    """
    Perform one complete simulation step including all components.
    
    Implements full time step from MFN_MATH_MODEL.md Section 2:
        1. Growth events (spikes) with probability p
        2. Field diffusion: V += α·∇²V
        3. Turing morphogenesis (if enabled)
        4. Quantum jitter (if enabled)
        5. Clamping to bounds
    
    Parameters
    ----------
    field : NDArray[np.floating]
        Current potential field V of shape (N, N).
    activator : NDArray[np.floating]
        Current activator field a of shape (N, N).
    inhibitor : NDArray[np.floating]
        Current inhibitor field i of shape (N, N).
    rng : np.random.Generator
        Random number generator.
    params : UpdateParameters | None
        Update parameters. Uses defaults if None.
    turing_enabled : bool
        Enable Turing morphogenesis. Default True.
    quantum_jitter : bool
        Enable quantum jitter noise. Default False.
    jitter_var : float
        Jitter variance. Default 0.0005.
    spike_probability : float
        Probability of growth event. Default 0.25.
    check_stability : bool
        Check for NaN/Inf. Default True.
    
    Returns
    -------
    tuple[NDArray, NDArray, NDArray, dict]
        Updated (field, activator, inhibitor) and metrics dict.
        Metrics include: growth_event, turing_activations, clamping_events.
    
    Raises
    ------
    NumericalInstabilityError
        If NaN/Inf detected during step.
    """
    if params is None:
        params = UpdateParameters()
    
    metrics: dict[str, Any] = {
        "growth_event": False,
        "turing_activations": 0,
        "clamping_events": 0,
    }
    
    grid_size = field.shape[0]
    
    # 1. Growth events (spikes)
    if rng.random() < spike_probability:
        field, _ = apply_growth_event(field, rng, grid_size)
        metrics["growth_event"] = True
    
    # 2. Field diffusion
    field = diffusion_update(
        field, params.alpha, params.boundary, check_stability=check_stability
    )
    
    # 3. Turing morphogenesis
    if turing_enabled:
        activator, inhibitor = activator_inhibitor_update(
            activator, inhibitor, params, check_stability=check_stability
        )
        
        field, activations = apply_turing_to_field(
            field, activator, params.turing_threshold, params.turing_contribution_v
        )
        metrics["turing_activations"] = activations
    
    # 4. Quantum jitter
    if quantum_jitter:
        field = apply_quantum_jitter(field, rng, jitter_var)
    
    # 5. Clamping
    field, clamp_count = clamp_potential_field(field)
    metrics["clamping_events"] = clamp_count
    
    # Final stability check
    if check_stability:
        validate_field_stability(field, field_name="field")
        validate_field_stability(activator, field_name="activator")
        validate_field_stability(inhibitor, field_name="inhibitor")
    
    return (
        cast(NDArray[np.floating[Any]], field),
        cast(NDArray[np.floating[Any]], activator),
        cast(NDArray[np.floating[Any]], inhibitor),
        metrics,
    )


def validate_cfl_condition(diffusion_coeff: float) -> bool:
    """
    Validate CFL stability condition for diffusion coefficient.
    
    Reference: MFN_MATH_MODEL.md Section 2.5
        dt * D * 4/dx² ≤ 1
        With dt=dx=1: D ≤ 0.25
    
    Parameters
    ----------
    diffusion_coeff : float
        Diffusion coefficient to validate.
    
    Returns
    -------
    bool
        True if coefficient satisfies CFL condition.
    """
    return diffusion_coeff < MAX_STABLE_DIFFUSION
