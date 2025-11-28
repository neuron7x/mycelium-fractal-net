
"""
MyceliumFractalNet v4.1 package.
"""

from .model import (
    MyceliumFractalNet,
    compute_heterosynaptic_modulation,
    compute_nernst_potential,
    compute_stdp_weight_change,
    compute_turing_dispersion,
    estimate_fractal_dimension,
    run_validation,
    run_validation_cli,
    simulate_mycelium_field,
    verify_stdp_lipschitz,
    verify_turing_instability,
)

__all__ = [
    "MyceliumFractalNet",
    "compute_heterosynaptic_modulation",
    "compute_nernst_potential",
    "compute_stdp_weight_change",
    "compute_turing_dispersion",
    "estimate_fractal_dimension",
    "run_validation",
    "run_validation_cli",
    "simulate_mycelium_field",
    "verify_stdp_lipschitz",
    "verify_turing_instability",
]

__version__ = "4.1.0"
