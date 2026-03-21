"""
Numerics submodule for MyceliumFractalNet.

Contains numerical algorithms and solvers for field simulation:
- grid_ops: Spatial discretization operators (Laplacian, gradient)
- update_rules: Time-stepping compatibility layer delegating into the canonical kernel
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .grid_ops import (
    BoundaryCondition,
    clamp_field,
    compute_field_statistics,
    compute_gradient,
    compute_laplacian,
    laplacian_backend,
    validate_field_bounds,
    validate_field_stability,
)

_UPDATE_RULE_EXPORTS = {
    "DEFAULT_ALPHA",
    "DEFAULT_D_ACTIVATOR",
    "DEFAULT_D_INHIBITOR",
    "DEFAULT_R_ACTIVATOR",
    "DEFAULT_R_INHIBITOR",
    "DEFAULT_TURING_THRESHOLD",
    "FIELD_V_MAX",
    "FIELD_V_MIN",
    "MAX_STABLE_DIFFUSION",
    "UpdateParameters",
    "activator_inhibitor_update",
    "apply_growth_event",
    "apply_quantum_jitter",
    "apply_turing_to_field",
    "clamp_potential_field",
    "diffusion_update",
    "full_simulation_step",
    "validate_cfl_condition",
}


def __getattr__(name: str) -> Any:
    if name in _UPDATE_RULE_EXPORTS:
        module = import_module("mycelium_fractal_net.numerics.update_rules")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BoundaryCondition",
    "compute_laplacian",
    "compute_gradient",
    "laplacian_backend",
    "compute_field_statistics",
    "validate_field_stability",
    "validate_field_bounds",
    "clamp_field",
    *_UPDATE_RULE_EXPORTS,
]
