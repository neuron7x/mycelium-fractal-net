
"""
MyceliumFractalNet v4.1 package.

Bio-inspired adaptive network with fractal dynamics, STDP plasticity,
sparse attention, and Byzantine-robust federated learning.

New in v4.1:
- core/ module with numerically stable engines for:
  - Membrane potential (Nernst equation, ODE integration)
  - Reaction-diffusion (Turing morphogenesis)
  - Fractal growth (IFS, box-counting)
"""

# Import core engines (new numerical implementations)
from .analytics import FeatureVector, compute_fractal_features
from .core import (
    FractalConfig,
    FractalGrowthEngine,
    FractalMetrics,
    MembraneConfig,
    MembraneEngine,
    MembraneMetrics,
    MyceliumField,
    NumericalInstabilityError,
    ReactionDiffusionConfig,
    ReactionDiffusionEngine,
    ReactionDiffusionMetrics,
    SimulationConfig,
    SimulationResult,
    StabilityError,
    ValueOutOfRangeError,
    run_mycelium_simulation,
    run_mycelium_simulation_with_history,
)
from .model import (
    BODY_TEMPERATURE_K,
    FARADAY_CONSTANT,
    ION_CLAMP_MIN,
    NERNST_RTFZ_MV,
    QUANTUM_JITTER_VAR,
    R_GAS_CONSTANT,
    SPARSE_TOPK,
    STDP_A_MINUS,
    STDP_A_PLUS,
    STDP_TAU_MINUS,
    STDP_TAU_PLUS,
    TURING_THRESHOLD,
    HierarchicalKrumAggregator,
    MyceliumFractalNet,
    SparseAttention,
    STDPPlasticity,
    ValidationConfig,
    compute_lyapunov_exponent,
    compute_nernst_potential,
    estimate_fractal_dimension,
    generate_fractal_ifs,
    run_validation,
    run_validation_cli,
    simulate_mycelium_field,
)

__all__ = [
    # Constants
    "R_GAS_CONSTANT",
    "FARADAY_CONSTANT",
    "BODY_TEMPERATURE_K",
    "NERNST_RTFZ_MV",
    "ION_CLAMP_MIN",
    "TURING_THRESHOLD",
    "STDP_TAU_PLUS",
    "STDP_TAU_MINUS",
    "STDP_A_PLUS",
    "STDP_A_MINUS",
    "SPARSE_TOPK",
    "QUANTUM_JITTER_VAR",
    # Functions
    "compute_nernst_potential",
    "simulate_mycelium_field",
    "estimate_fractal_dimension",
    "generate_fractal_ifs",
    "compute_lyapunov_exponent",
    "run_validation",
    "run_validation_cli",
    # Simulation API (new)
    "run_mycelium_simulation",
    "run_mycelium_simulation_with_history",
    # Classes (legacy)
    "STDPPlasticity",
    "SparseAttention",
    "HierarchicalKrumAggregator",
    "MyceliumFractalNet",
    "ValidationConfig",
    # Simulation Types (new)
    "SimulationConfig",
    "SimulationResult",
    "MyceliumField",
    "FeatureVector",
    # Analytics API (new)
    "compute_fractal_features",
    # Core Engines (new)
    "MembraneEngine",
    "MembraneConfig",
    "MembraneMetrics",
    "ReactionDiffusionEngine",
    "ReactionDiffusionConfig",
    "ReactionDiffusionMetrics",
    "FractalGrowthEngine",
    "FractalConfig",
    "FractalMetrics",
    # Exceptions
    "StabilityError",
    "ValueOutOfRangeError",
    "NumericalInstabilityError",
]

__version__ = "4.1.0"
