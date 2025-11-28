
"""
MyceliumFractalNet v4.1 package.

Bio-inspired adaptive network with fractal dynamics, STDP plasticity,
sparse attention, and Byzantine-robust federated learning.
"""

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
    # Classes
    "STDPPlasticity",
    "SparseAttention",
    "HierarchicalKrumAggregator",
    "MyceliumFractalNet",
    "ValidationConfig",
]

__version__ = "4.1.0"
