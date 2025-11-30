"""
Core numerical engines for MyceliumFractalNet.

This module provides numerically stable implementations of the six
canonical domain modules:

1. **nernst** — Nernst-Planck electrochemistry (membrane potentials)
2. **turing** — Turing morphogenesis (reaction-diffusion patterns)
3. **fractal** — Fractal analysis (box-counting dimension, IFS)
4. **stdp** — STDP plasticity (spike-timing dependent learning)
5. **federated** — Byzantine-robust aggregation (Hierarchical Krum)
6. **stability** — Lyapunov exponents and stability metrics

All implementations follow MFN_MATH_MODEL.md specifications with:
- Explicit stability constraints (CFL conditions, clamping)
- NaN/Inf prevention via validation and clamping
- Reproducible results via seeded RNG
- Configurable parameters via dataclasses

Layer Boundaries:
    core/ contains pure mathematical/dynamical implementations.
    No FastAPI, uvicorn, or HTTP-level dependencies allowed here.
    Integration with external systems goes through integration/ layer.

Reference:
    - docs/MFN_MATH_MODEL.md — Mathematical formalization
    - docs/ARCHITECTURE.md — System architecture
    - docs/MFN_CODE_STRUCTURE.md — Code structure documentation
"""

from .engine import run_mycelium_simulation, run_mycelium_simulation_with_history
from .exceptions import (
    NumericalInstabilityError,
    StabilityError,
    ValueOutOfRangeError,
)
from .federated import (
    HierarchicalKrumAggregator,
    aggregate_gradients_krum,
)
from .field import MyceliumField
from .fractal import (
    estimate_fractal_dimension,
    generate_fractal_ifs,
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
from .nernst import compute_nernst_potential
from .reaction_diffusion_engine import (
    ReactionDiffusionConfig,
    ReactionDiffusionEngine,
    ReactionDiffusionMetrics,
)
from .stability import (
    compute_lyapunov_exponent,
    compute_stability_metrics,
    is_stable,
)
from .stdp import STDPPlasticity
from .turing import simulate_mycelium_field
from .types import SimulationConfig, SimulationResult

__all__ = [
    # Exceptions
    "StabilityError",
    "ValueOutOfRangeError",
    "NumericalInstabilityError",
    # Simulation Types
    "SimulationConfig",
    "SimulationResult",
    "MyceliumField",
    # Simulation API
    "run_mycelium_simulation",
    "run_mycelium_simulation_with_history",
    # === Domain-Specific Public API ===
    # Nernst (membrane potentials)
    "compute_nernst_potential",
    "MembraneEngine",
    "MembraneConfig",
    "MembraneMetrics",
    # Turing (reaction-diffusion)
    "simulate_mycelium_field",
    "ReactionDiffusionEngine",
    "ReactionDiffusionConfig",
    "ReactionDiffusionMetrics",
    # Fractal (dimension analysis)
    "estimate_fractal_dimension",
    "generate_fractal_ifs",
    "FractalGrowthEngine",
    "FractalConfig",
    "FractalMetrics",
    # STDP (plasticity)
    "STDPPlasticity",
    # Federated (Byzantine-robust aggregation)
    "HierarchicalKrumAggregator",
    "aggregate_gradients_krum",
    # Stability (Lyapunov analysis)
    "compute_lyapunov_exponent",
    "compute_stability_metrics",
    "is_stable",
]
