"""
Infrastructure utilities for MyceliumFractalNet.

This module provides:
- RNG control layer for reproducible experiments
- Run registry for tracking experimental runs

Reference:
    - docs/MFN_REPRODUCIBILITY.md — Reproducibility documentation
"""

from .rng import (
    DEFAULT_SEED,
    RNGContext,
    create_rng,
    get_numpy_rng,
    set_global_seed,
)
from .run_registry import (
    RunHandle,
    RunMeta,
    RunRegistry,
    RunStatus,
    get_registry,
    reset_global_registry,
)

__all__ = [
    # RNG control
    "DEFAULT_SEED",
    "RNGContext",
    "create_rng",
    "set_global_seed",
    "get_numpy_rng",
    # Run registry
    "RunRegistry",
    "RunHandle",
    "RunMeta",
    "RunStatus",
    "get_registry",
    "reset_global_registry",
]
