"""
RNG Control Layer for MyceliumFractalNet.

Provides unified random number generator control for reproducible simulations.
Supports NumPy random generators and optional PyTorch seeding.

This module is the single source of truth for RNG state in MFN simulations.

Usage:
    >>> from mycelium_fractal_net.infra.rng import create_rng, set_global_seed
    >>> # Create isolated RNG context
    >>> rng_ctx = create_rng(seed=42)
    >>> rng = rng_ctx.numpy_rng
    >>> # Or set global seed for all RNGs
    >>> set_global_seed(42)

Reference: docs/MFN_REPRODUCIBILITY.md
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Optional torch support - many MFN operations don't require torch
_TORCH_AVAILABLE = False
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    pass


# Global RNG instance for module-level access
_GLOBAL_RNG: np.random.Generator | None = None
_GLOBAL_SEED: int | None = None


@dataclass
class RNGContext:
    """
    Unified RNG context for reproducible simulations.

    Encapsulates NumPy random generator and optionally synchronizes
    Python's random module and PyTorch RNG state.

    Attributes:
        seed: The seed used to initialize this context.
        numpy_rng: NumPy random generator instance.

    Example:
        >>> ctx = RNGContext(seed=42)
        >>> rng = ctx.numpy_rng
        >>> value = rng.random()
        >>> # Fork for child operations
        >>> child_ctx = ctx.fork()
    """

    seed: int | None = None
    _numpy_rng: np.random.Generator = field(default=None, repr=False)  # type: ignore

    def __post_init__(self) -> None:
        """Initialize RNG state from seed."""
        if self._numpy_rng is None:
            self._numpy_rng = np.random.default_rng(self.seed)

    @property
    def numpy_rng(self) -> np.random.Generator:
        """Get NumPy random generator."""
        return self._numpy_rng

    def fork(self, child_seed: int | None = None) -> "RNGContext":
        """
        Create a child RNG context with derived seed.

        The child context has independent RNG state, derived from the
        parent's current state or an explicit child_seed.

        Args:
            child_seed: Optional explicit seed for child. If None,
                       derives seed from parent RNG.

        Returns:
            New RNGContext with independent state.
        """
        if child_seed is None:
            # Derive child seed from current RNG state
            child_seed = int(self._numpy_rng.integers(0, 2**31))
        return RNGContext(seed=child_seed)

    def reset(self) -> None:
        """
        Reset RNG to initial state using original seed.

        Useful for re-running experiments with identical random state.
        """
        self._numpy_rng = np.random.default_rng(self.seed)

    def get_state(self) -> dict[str, Any]:
        """
        Get current RNG state for serialization.

        Returns:
            Dictionary with seed and RNG bit generator state.
        """
        return {
            "seed": self.seed,
            "bit_generator_state": self._numpy_rng.bit_generator.state,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "RNGContext":
        """
        Restore RNG context from serialized state.

        Args:
            state: Dictionary from get_state().

        Returns:
            Restored RNGContext.
        """
        ctx = cls(seed=state.get("seed"))
        if "bit_generator_state" in state:
            ctx._numpy_rng.bit_generator.state = state["bit_generator_state"]
        return ctx


def create_rng(seed: int | None = None) -> RNGContext:
    """
    Create a new RNG context with the given seed.

    Factory function for creating isolated RNG contexts.

    Args:
        seed: Random seed. None for non-deterministic initialization.

    Returns:
        RNGContext with NumPy generator initialized.

    Example:
        >>> ctx = create_rng(42)
        >>> rng = ctx.numpy_rng
        >>> value = rng.random()  # Reproducible
    """
    return RNGContext(seed=seed)


def set_global_seed(seed: int) -> None:
    """
    Set global seed for all random number generators.

    This function sets seeds for:
    - NumPy (np.random.default_rng)
    - Python random module
    - PyTorch (if available)

    Use this at the start of scripts/experiments for full reproducibility.

    Args:
        seed: Integer seed value.

    Example:
        >>> set_global_seed(42)
        >>> # All subsequent random operations are reproducible
    """
    global _GLOBAL_RNG, _GLOBAL_SEED

    _GLOBAL_SEED = seed
    _GLOBAL_RNG = np.random.default_rng(seed)

    # Python random module
    random.seed(seed)

    # NumPy legacy API (for compatibility)
    np.random.seed(seed)

    # PyTorch (if available)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_global_rng() -> np.random.Generator:
    """
    Get the global RNG instance.

    Returns the global NumPy generator set by set_global_seed().
    If no global seed was set, creates a new unseeded generator.

    Returns:
        NumPy random generator.
    """
    global _GLOBAL_RNG
    if _GLOBAL_RNG is None:
        _GLOBAL_RNG = np.random.default_rng()
    return _GLOBAL_RNG


def get_global_seed() -> int | None:
    """
    Get the current global seed value.

    Returns:
        The seed set by set_global_seed(), or None if not set.
    """
    return _GLOBAL_SEED


__all__ = [
    "RNGContext",
    "create_rng",
    "set_global_seed",
    "get_global_rng",
    "get_global_seed",
]
