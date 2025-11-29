"""
Numerics submodule for MyceliumFractalNet.

Contains numerical algorithms and solvers for field simulation:
- grid_ops: Spatial discretization operators (Laplacian, gradient)
- update_rules: Time-stepping schemes for reaction-diffusion dynamics

Reference: MFN_MATH_MODEL.md for mathematical formalization.
"""

from .grid_ops import (
    BoundaryCondition,
    clamp_field,
    compute_field_statistics,
    compute_gradient,
    compute_laplacian,
    validate_field_bounds,
    validate_field_stability,
)
from .update_rules import (
    DEFAULT_ALPHA,
    DEFAULT_D_ACTIVATOR,
    DEFAULT_D_INHIBITOR,
    DEFAULT_R_ACTIVATOR,
    DEFAULT_R_INHIBITOR,
    DEFAULT_TURING_THRESHOLD,
    FIELD_V_MAX,
    FIELD_V_MIN,
    MAX_STABLE_DIFFUSION,
    UpdateParameters,
    activator_inhibitor_update,
    apply_growth_event,
    apply_quantum_jitter,
    apply_turing_to_field,
    clamp_potential_field,
    diffusion_update,
    full_simulation_step,
    validate_cfl_condition,
)

__all__ = [
    # grid_ops
    "BoundaryCondition",
    "compute_laplacian",
    "compute_gradient",
    "compute_field_statistics",
    "validate_field_stability",
    "validate_field_bounds",
    "clamp_field",
    # update_rules
    "UpdateParameters",
    "diffusion_update",
    "activator_inhibitor_update",
    "apply_turing_to_field",
    "apply_growth_event",
    "apply_quantum_jitter",
    "clamp_potential_field",
    "full_simulation_step",
    "validate_cfl_condition",
    # constants
    "DEFAULT_D_ACTIVATOR",
    "DEFAULT_D_INHIBITOR",
    "DEFAULT_R_ACTIVATOR",
    "DEFAULT_R_INHIBITOR",
    "DEFAULT_ALPHA",
    "DEFAULT_TURING_THRESHOLD",
    "MAX_STABLE_DIFFUSION",
    "FIELD_V_MIN",
    "FIELD_V_MAX",
]
