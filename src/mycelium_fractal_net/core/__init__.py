"""
Core numerical engines for MyceliumFractalNet.

This module provides numerically stable implementations of:
1. Membrane potential dynamics (Nernst equation, ODE integration)
2. Reaction-diffusion field evolution (Turing morphogenesis PDEs)
3. Fractal growth models (IFS, DLA with stability guarantees)
4. STDP synaptic plasticity
5. Byzantine-robust federated aggregation (Krum)
6. Stability analysis (Lyapunov exponents)

All implementations follow MFN_MATH_MODEL.md specifications with:
- Explicit stability constraints (CFL conditions, clamping)
- NaN/Inf prevention via validation and clamping
- Reproducible results via seeded RNG
- Configurable parameters via dataclasses

Domain Modules:
- nernst: Nernst equation, ion electrochemistry
- turing: Turing morphogenesis, reaction-diffusion patterns
- fractal: Fractal dimension, IFS generation
- stdp: Spike-Timing Dependent Plasticity
- federated: Byzantine-robust Krum aggregation
- stability: Lyapunov exponents, stability metrics
"""

from .engine import run_mycelium_simulation, run_mycelium_simulation_with_history
from .exceptions import (
    NumericalInstabilityError,
    StabilityError,
    ValueOutOfRangeError,
)

# Federated learning
from .federated import (
    FEDERATED_BYZANTINE_FRACTION,
    FEDERATED_NUM_CLUSTERS,
    HierarchicalKrumAggregator,
    aggregate_gradients_krum,
)
from .field import MyceliumField

# Fractal analysis
from .fractal import (
    BIOLOGICAL_DIM_MAX,
    BIOLOGICAL_DIM_MIN,
    compute_lyapunov_exponent,
    estimate_fractal_dimension,
    generate_fractal_ifs,
)

# === Engine Imports (original) ===
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

# === Domain Module Imports ===
# Nernst electrochemistry
from .nernst import (
    BODY_TEMPERATURE_K,
    FARADAY_CONSTANT,
    ION_CLAMP_MIN,
    NERNST_RTFZ_MV,
    R_GAS_CONSTANT,
    compute_nernst_potential,
    symbolic_nernst_verification,
)
from .reaction_diffusion_engine import (
    BoundaryCondition,
    ReactionDiffusionConfig,
    ReactionDiffusionEngine,
    ReactionDiffusionMetrics,
)

# Stability analysis
from .stability import (
    LYAPUNOV_STABLE_MAX,
    compute_stability_metrics,
    is_stable,
)

# STDP plasticity
from .stdp import (
    STDP_A_MINUS,
    STDP_A_PLUS,
    STDP_TAU_MINUS,
    STDP_TAU_PLUS,
    STDPPlasticity,
)

# Turing morphogenesis
from .turing import (
    QUANTUM_JITTER_VAR,
    TURING_THRESHOLD,
    simulate_mycelium_field,
)
from .types import SimulationConfig, SimulationResult

__all__ = [
    # === Exceptions ===
    "StabilityError",
    "ValueOutOfRangeError",
    "NumericalInstabilityError",
    # === Simulation Types ===
    "SimulationConfig",
    "SimulationResult",
    "MyceliumField",
    # === Simulation API ===
    "run_mycelium_simulation",
    "run_mycelium_simulation_with_history",
    # === Nernst Module (electrochemistry) ===
    "R_GAS_CONSTANT",
    "FARADAY_CONSTANT",
    "BODY_TEMPERATURE_K",
    "NERNST_RTFZ_MV",
    "ION_CLAMP_MIN",
    "compute_nernst_potential",
    "symbolic_nernst_verification",
    "MembraneEngine",
    "MembraneConfig",
    "MembraneMetrics",
    # === Turing Module (morphogenesis) ===
    "TURING_THRESHOLD",
    "QUANTUM_JITTER_VAR",
    "simulate_mycelium_field",
    "ReactionDiffusionEngine",
    "ReactionDiffusionConfig",
    "ReactionDiffusionMetrics",
    "BoundaryCondition",
    # === Fractal Module (dimension analysis) ===
    "BIOLOGICAL_DIM_MIN",
    "BIOLOGICAL_DIM_MAX",
    "estimate_fractal_dimension",
    "generate_fractal_ifs",
    "compute_lyapunov_exponent",
    "FractalGrowthEngine",
    "FractalConfig",
    "FractalMetrics",
    # === STDP Module (synaptic plasticity) ===
    "STDP_TAU_PLUS",
    "STDP_TAU_MINUS",
    "STDP_A_PLUS",
    "STDP_A_MINUS",
    "STDPPlasticity",
    # === Federated Module (Byzantine aggregation) ===
    "FEDERATED_NUM_CLUSTERS",
    "FEDERATED_BYZANTINE_FRACTION",
    "HierarchicalKrumAggregator",
    "aggregate_gradients_krum",
    # === Stability Module (Lyapunov analysis) ===
    "LYAPUNOV_STABLE_MAX",
    "is_stable",
    "compute_stability_metrics",
]
