"""
Core numerical engines for MyceliumFractalNet.

This module provides numerically stable implementations of:
1. Membrane potential dynamics (Nernst equation, ODE integration)
2. Reaction-diffusion field evolution (Turing morphogenesis PDEs)
3. Fractal growth models (IFS, DLA with stability guarantees)

All implementations follow MATH_MODEL.md specifications with:
- Explicit stability constraints (CFL conditions, clamping)
- NaN/Inf prevention via validation and clamping
- Reproducible results via seeded RNG
- Configurable parameters via dataclasses
"""

from .exceptions import (
    NumericalInstabilityError,
    StabilityError,
    ValueOutOfRangeError,
)
from .fractal_growth_engine import (
    FractalConfig,
    FractalGrowthEngine,
    FractalMetrics,
)
from .membrane_engine import (
    MembraneConfig,
    MembraneEngine,
    MembraneMetrics,
)
from .reaction_diffusion_engine import (
    ReactionDiffusionConfig,
    ReactionDiffusionEngine,
    ReactionDiffusionMetrics,
)

__all__ = [
    # Exceptions
    "StabilityError",
    "ValueOutOfRangeError",
    "NumericalInstabilityError",
    # Membrane Engine
    "MembraneEngine",
    "MembraneConfig",
    "MembraneMetrics",
    # Reaction-Diffusion Engine
    "ReactionDiffusionEngine",
    "ReactionDiffusionConfig",
    "ReactionDiffusionMetrics",
    # Fractal Growth Engine
    "FractalGrowthEngine",
    "FractalConfig",
    "FractalMetrics",
]
