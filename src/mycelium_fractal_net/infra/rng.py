"""
Unified Random Number Generator (RNG) control layer for MyceliumFractalNet.

This module provides centralized RNG management for reproducible simulations,
experiments, and tests. It supports deterministic seeding across numpy, random,
and torch libraries.

Usage:
    >>> from mycelium_fractal_net.infra.rng import create_rng, set_global_seed, RNGContext
    >>> # Create a seeded RNG context
    >>> rng_ctx = create_rng(seed=42)
    >>> rng = rng_ctx.numpy_rng  # numpy Generator
    >>> # Set global seed for all libraries
    >>> set_global_seed(42)

Reference:
    - docs/MFN_REPRODUCIBILITY.md — Reproducibility documentation
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.random import Generator

# Default seed for reproducibility when not specified
DEFAULT_SEED: int = 42


@dataclass
class RNGContext:
    """
    Container for synchronized random number generators.

    Holds RNG instances for numpy, Python's random module, and optionally torch.
    Ensures consistent seeding across all sources of randomness.

    Attributes
    ----------
    seed : int
        The seed used to initialize the RNG context.
    numpy_rng : Generator
        NumPy random generator (numpy.random.Generator).
    _original_random_state : tuple | None
        Original state of Python's random module (for restoration).

    Example
    -------
    >>> ctx = RNGContext.create(seed=42)
    >>> arr = ctx.numpy_rng.random(10)  # reproducible array
    >>> ctx_fork = ctx.fork()  # independent copy
    """

    seed: int
    numpy_rng: Generator = field(repr=False)
    _original_random_state: Optional[tuple[object, ...]] = field(default=None, repr=False)

    @classmethod
    def create(cls, seed: int) -> "RNGContext":
        """
        Create a new RNG context with the given seed.

        Parameters
        ----------
        seed : int
            The seed for reproducibility.

        Returns
        -------
        RNGContext
            A new RNG context with synchronized generators.
        """
        # Store original state for potential restoration
        original_state = random.getstate()

        # Seed Python's random module
        random.seed(seed)

        # Create NumPy generator
        numpy_rng = np.random.default_rng(seed)

        # Optionally seed torch if available
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass  # torch not installed

        return cls(
            seed=seed,
            numpy_rng=numpy_rng,
            _original_random_state=original_state,
        )

    def fork(self, offset: int = 1) -> "RNGContext":
        """
        Create a new independent RNG context derived from this one.

        Useful for spawning independent random streams for parallel operations
        or sub-simulations.

        Parameters
        ----------
        offset : int
            Offset to add to the seed for the new context. Default is 1.

        Returns
        -------
        RNGContext
            A new RNG context with seed = self.seed + offset.
        """
        return RNGContext.create(self.seed + offset)

    def reset(self) -> None:
        """
        Reset the RNG to the original seed state.

        This allows re-running the same sequence of random numbers.
        """
        random.seed(self.seed)
        self.numpy_rng = np.random.default_rng(self.seed)

        try:
            import torch

            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        except ImportError:
            pass

    def restore_original_state(self) -> None:
        """
        Restore the original random state from before this context was created.

        Useful for tests that should not affect global random state.
        """
        if self._original_random_state is not None:
            random.setstate(self._original_random_state)


def create_rng(seed: int | None = None) -> RNGContext:
    """
    Create a new RNG context.

    Parameters
    ----------
    seed : int | None
        The seed for reproducibility. If None, uses DEFAULT_SEED (42).

    Returns
    -------
    RNGContext
        A new RNG context with synchronized generators.

    Example
    -------
    >>> rng_ctx = create_rng(seed=42)
    >>> values = rng_ctx.numpy_rng.random(5)
    """
    if seed is None:
        seed = DEFAULT_SEED
    return RNGContext.create(seed)


def set_global_seed(seed: int) -> None:
    """
    Set the global seed for all random number generators.

    This function seeds:
    - Python's random module
    - NumPy's default random state (legacy API)
    - PyTorch (if available)

    Parameters
    ----------
    seed : int
        The seed value for reproducibility.

    Example
    -------
    >>> set_global_seed(42)
    >>> # Now all random operations will be deterministic
    """
    # Python's random
    random.seed(seed)

    # NumPy legacy global state (for code using np.random.* directly)
    np.random.seed(seed)

    # PyTorch
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # For full reproducibility on CUDA (may impact performance)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_numpy_rng(seed: int | None = None) -> Generator:
    """
    Get a NumPy random generator with the given seed.

    Convenience function that returns just the NumPy Generator
    without creating a full RNGContext.

    Parameters
    ----------
    seed : int | None
        The seed for the generator. If None, uses 42 as default.

    Returns
    -------
    Generator
        A NumPy random Generator instance.

    Example
    -------
    >>> rng = get_numpy_rng(42)
    >>> values = rng.random(10)
    """
    if seed is None:
        seed = DEFAULT_SEED
    return np.random.default_rng(seed)


__all__ = [
    "DEFAULT_SEED",
    "RNGContext",
    "create_rng",
    "set_global_seed",
    "get_numpy_rng",
]
