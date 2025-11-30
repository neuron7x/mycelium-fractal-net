
"""
MyceliumFractalNet v4.1 package.

Bio-inspired adaptive network with fractal dynamics, STDP plasticity,
sparse attention, and Byzantine-robust federated learning.

Public API:
-----------
Functions (aligned with README examples):
    - compute_nernst_potential(z_valence, c_out, c_in, T) -> float
    - simulate_mycelium_field(rng, grid_size, steps, ...) -> (field, growth_events)
    - estimate_fractal_dimension(binary_field, ...) -> float
    - generate_fractal_ifs(rng, ...) -> (points, lyapunov)
    - compute_lyapunov_exponent(field_history, dt) -> float
    - aggregate_gradients_krum(gradients, ...) -> Tensor

Classes:
    - STDPPlasticity: Spike-Timing Dependent Plasticity module
    - SparseAttention: Top-k sparse attention mechanism
    - HierarchicalKrumAggregator: Byzantine-robust federated aggregation
    - MyceliumFractalNet: Neural network with fractal dynamics

Core Engines:
    - MembraneEngine: Nernst equation, membrane potential dynamics
    - ReactionDiffusionEngine: Turing morphogenesis pattern formation
    - FractalGrowthEngine: IFS fractals, box-counting dimension

Architecture Layers:
    - core/: Pure numerical engines (no infrastructure dependencies)
    - integration/: Schemas, adapters, service context
    - api.py, CLI: External interfaces

Reference: docs/ARCHITECTURE.md, docs/MFN_SYSTEM_ROLE.md
"""

# === Analytics Module ===
from .analytics import FeatureVector, compute_fractal_features

# === Configuration Module ===
from .config import (
    DatasetConfig,
    FeatureConfig,
    make_dataset_config_default,
    make_dataset_config_demo,
    make_feature_config_default,
    make_feature_config_demo,
    make_simulation_config_default,
    make_simulation_config_demo,
    validate_dataset_config,
    validate_feature_config,
    validate_simulation_config,
)

# === Core Domain Modules (canonical imports) ===
# Nernst electrochemistry (core.nernst)
# Turing morphogenesis (core.turing)
# Fractal analysis (core.fractal)
# STDP plasticity (core.stdp)
# Federated learning (core.federated)
# Core engines
from .core import (
    BODY_TEMPERATURE_K,
    FARADAY_CONSTANT,
    ION_CLAMP_MIN,
    NERNST_RTFZ_MV,
    QUANTUM_JITTER_VAR,
    R_GAS_CONSTANT,
    STDP_A_MINUS,
    STDP_A_PLUS,
    STDP_TAU_MINUS,
    STDP_TAU_PLUS,
    TURING_THRESHOLD,
    FractalConfig,
    FractalGrowthEngine,
    FractalMetrics,
    HierarchicalKrumAggregator,
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
    STDPPlasticity,
    ValueOutOfRangeError,
    aggregate_gradients_krum,
    compute_lyapunov_exponent,
    compute_nernst_potential,
    estimate_fractal_dimension,
    generate_fractal_ifs,
    run_mycelium_simulation,
    run_mycelium_simulation_with_history,
    simulate_mycelium_field,
)

# === Model Module (neural network, validation) ===
from .model import (
    SPARSE_TOPK,
    MyceliumFractalNet,
    SparseAttention,
    ValidationConfig,
    run_validation,
    run_validation_cli,
)

__all__ = [
    # === Physical Constants ===
    "R_GAS_CONSTANT",
    "FARADAY_CONSTANT",
    "BODY_TEMPERATURE_K",
    "NERNST_RTFZ_MV",
    "ION_CLAMP_MIN",
    # === Model Parameters ===
    "TURING_THRESHOLD",
    "STDP_TAU_PLUS",
    "STDP_TAU_MINUS",
    "STDP_A_PLUS",
    "STDP_A_MINUS",
    "SPARSE_TOPK",
    "QUANTUM_JITTER_VAR",
    # === Public Functions (README API) ===
    "compute_nernst_potential",
    "simulate_mycelium_field",
    "estimate_fractal_dimension",
    "generate_fractal_ifs",
    "compute_lyapunov_exponent",
    "aggregate_gradients_krum",
    "run_validation",
    "run_validation_cli",
    # === Simulation API ===
    "run_mycelium_simulation",
    "run_mycelium_simulation_with_history",
    # === Classes ===
    "STDPPlasticity",
    "SparseAttention",
    "HierarchicalKrumAggregator",
    "MyceliumFractalNet",
    "ValidationConfig",
    # === Types ===
    "SimulationConfig",
    "SimulationResult",
    "MyceliumField",
    "FeatureVector",
    # === Analytics ===
    "compute_fractal_features",
    # === Core Engines ===
    "MembraneEngine",
    "MembraneConfig",
    "MembraneMetrics",
    "ReactionDiffusionEngine",
    "ReactionDiffusionConfig",
    "ReactionDiffusionMetrics",
    "FractalGrowthEngine",
    "FractalConfig",
    "FractalMetrics",
    # === Exceptions ===
    "StabilityError",
    "ValueOutOfRangeError",
    "NumericalInstabilityError",
    # === Configuration ===
    "DatasetConfig",
    "FeatureConfig",
    "validate_simulation_config",
    "validate_feature_config",
    "validate_dataset_config",
    "make_simulation_config_demo",
    "make_simulation_config_default",
    "make_feature_config_demo",
    "make_feature_config_default",
    "make_dataset_config_demo",
    "make_dataset_config_default",
]

__version__ = "4.1.0"
