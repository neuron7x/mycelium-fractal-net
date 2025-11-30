
"""
MyceliumFractalNet v4.1 package.

Bio-inspired adaptive network with fractal dynamics, STDP plasticity,
sparse attention, and Byzantine-robust federated learning.

Public API (matching README examples):
--------------------------------------

**Nernst-Planck Electrochemistry:**
    >>> from mycelium_fractal_net import compute_nernst_potential
    >>> E_K = compute_nernst_potential(z_valence=1, concentration_out_molar=5e-3,
    ...                                concentration_in_molar=140e-3, temperature_k=310.0)
    >>> # E_K ≈ -89 mV

**Turing Morphogenesis:**
    >>> from mycelium_fractal_net import simulate_mycelium_field
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> field, growth_events = simulate_mycelium_field(rng, grid_size=64, steps=64)

**Fractal Analysis:**
    >>> from mycelium_fractal_net import estimate_fractal_dimension
    >>> binary = field > -0.060  # threshold -60 mV
    >>> D = estimate_fractal_dimension(binary)
    >>> # D ∈ [1.4, 1.9]

**Federated Learning:**
    >>> from mycelium_fractal_net import aggregate_gradients_krum
    >>> import torch
    >>> gradients = [torch.randn(100) for _ in range(50)]
    >>> aggregated = aggregate_gradients_krum(gradients)

Architecture Layers:
-------------------
- **core/** — Pure mathematical/dynamical engines (no HTTP dependencies)
- **integration/** — Schemas, adapters for API/CLI
- **analytics/** — Feature extraction module

Reference:
    - docs/MFN_SYSTEM_ROLE.md — System capabilities and boundaries
    - docs/ARCHITECTURE.md — System architecture
    - docs/MFN_CODE_STRUCTURE.md — Code structure documentation
"""

# === Analytics API ===
from .analytics import FeatureVector, compute_fractal_features

# === Centralized Configuration ===
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

# === Core Domain Modules (Canonical API) ===
from .core import (
    # Exceptions
    NumericalInstabilityError,
    StabilityError,
    ValueOutOfRangeError,
    # Simulation orchestration
    MyceliumField,
    SimulationConfig,
    SimulationResult,
    run_mycelium_simulation,
    run_mycelium_simulation_with_history,
    # Nernst (membrane potentials)
    compute_nernst_potential,
    MembraneConfig,
    MembraneEngine,
    MembraneMetrics,
    # Turing (reaction-diffusion)
    simulate_mycelium_field,
    ReactionDiffusionConfig,
    ReactionDiffusionEngine,
    ReactionDiffusionMetrics,
    # Fractal (dimension analysis)
    estimate_fractal_dimension,
    generate_fractal_ifs,
    FractalConfig,
    FractalGrowthEngine,
    FractalMetrics,
    # STDP (plasticity)
    STDPPlasticity,
    # Federated (Byzantine-robust aggregation)
    HierarchicalKrumAggregator,
    aggregate_gradients_krum,
    # Stability (Lyapunov analysis)
    compute_lyapunov_exponent,
    compute_stability_metrics,
    is_stable,
)

# === Model Layer (Neural Network, Validation) ===
from .model import (
    # Physical constants
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
    # Neural network
    MyceliumFractalNet,
    SparseAttention,
    # Validation pipeline
    ValidationConfig,
    run_validation,
    run_validation_cli,
)

__all__ = [
    # === PUBLIC API (as shown in README) ===
    # Nernst-Planck (membrane potentials)
    "compute_nernst_potential",
    # Turing (reaction-diffusion)
    "simulate_mycelium_field",
    # Fractal (dimension analysis)
    "estimate_fractal_dimension",
    "generate_fractal_ifs",
    # Federated (Byzantine-robust aggregation)
    "aggregate_gradients_krum",
    "HierarchicalKrumAggregator",
    # Stability (Lyapunov analysis)
    "compute_lyapunov_exponent",
    "compute_stability_metrics",
    "is_stable",
    # === VALIDATION & NEURAL NETWORK ===
    "run_validation",
    "run_validation_cli",
    "MyceliumFractalNet",
    "ValidationConfig",
    # === SIMULATION API ===
    "run_mycelium_simulation",
    "run_mycelium_simulation_with_history",
    "SimulationConfig",
    "SimulationResult",
    "MyceliumField",
    # === ANALYTICS API ===
    "FeatureVector",
    "compute_fractal_features",
    # === CORE ENGINE CLASSES ===
    # Nernst
    "MembraneEngine",
    "MembraneConfig",
    "MembraneMetrics",
    # Turing
    "ReactionDiffusionEngine",
    "ReactionDiffusionConfig",
    "ReactionDiffusionMetrics",
    # Fractal
    "FractalGrowthEngine",
    "FractalConfig",
    "FractalMetrics",
    # STDP
    "STDPPlasticity",
    # Attention
    "SparseAttention",
    # === PHYSICAL CONSTANTS ===
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
    # === EXCEPTIONS ===
    "StabilityError",
    "ValueOutOfRangeError",
    "NumericalInstabilityError",
    # === CONFIGURATION ===
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
