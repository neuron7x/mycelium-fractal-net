"""
MyceliumFractalNet Core Numerical Engines.

This package provides stable numerical implementations for:
- Membrane potential dynamics (ODE integration)
- Reaction-diffusion field simulation (PDE discretization)
- Fractal growth models (DLA/IFS stochastic)

Each engine follows the mathematical specifications in docs/ARCHITECTURE.md
and includes:
- Explicit stability conditions and parameter validation
- NaN/Inf checking with custom exceptions
- Configurable parameters via dataclasses
- Metrics collection for monitoring
"""

from mycelium_fractal_net.core.config import (
    FractalGrowthConfig,
    MembraneEngineConfig,
    ReactionDiffusionConfig,
)
from mycelium_fractal_net.core.exceptions import (
    NumericalInstabilityError,
    StabilityError,
    ValueOutOfRangeError,
)
from mycelium_fractal_net.core.fractal_growth_engine import (
    FractalGrowthEngine,
    FractalGrowthMetrics,
)
from mycelium_fractal_net.core.membrane_engine import (
    MembraneEngine,
    MembraneMetrics,
)
from mycelium_fractal_net.core.reaction_diffusion_engine import (
    ReactionDiffusionEngine,
    ReactionDiffusionMetrics,
)

__all__ = [
    # Config
    "MembraneEngineConfig",
    "ReactionDiffusionConfig",
    "FractalGrowthConfig",
    # Exceptions
    "StabilityError",
    "ValueOutOfRangeError",
    "NumericalInstabilityError",
    # Engines
    "MembraneEngine",
    "MembraneMetrics",
    "ReactionDiffusionEngine",
    "ReactionDiffusionMetrics",
    "FractalGrowthEngine",
    "FractalGrowthMetrics",
]
