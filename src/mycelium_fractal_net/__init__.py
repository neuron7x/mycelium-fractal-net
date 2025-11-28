
"""
MyceliumFractalNet v4.1 package.
"""

from .model import (
    MyceliumFractalNet,
    compute_nernst_potential,
    simulate_mycelium_field,
    estimate_fractal_dimension,
    run_validation,
    run_validation_cli,
)

__all__ = [
    "MyceliumFractalNet",
    "compute_nernst_potential",
    "simulate_mycelium_field",
    "estimate_fractal_dimension",
    "run_validation",
    "run_validation_cli",
]

__version__ = "4.1.0"
