"""
Turing Morphogenesis Module — Reaction-Diffusion Pattern Formation.

This module provides Turing reaction-diffusion pattern formation
capabilities. Re-exports the stable implementation from reaction_diffusion_engine.py
and adds the legacy simulate_mycelium_field function from model.py.

Reference: MFN_MATH_MODEL.md Section 2 (Reaction-Diffusion Processes)

Equations:
    ∂a/∂t = D_a ∇²a + r_a * a(1-a) - i     # Activator
    ∂i/∂t = D_i ∇²i + r_i * (a - i)         # Inhibitor

Parameters (from MFN_MATH_MODEL.md):
    D_a = 0.1         - Activator diffusion
    D_i = 0.05        - Inhibitor diffusion
    r_a = 0.01        - Activator reaction rate
    r_i = 0.02        - Inhibitor reaction rate
    θ = 0.75          - Turing activation threshold
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

# Re-export from reaction_diffusion_engine
from .reaction_diffusion_engine import (
    DEFAULT_D_ACTIVATOR,
    DEFAULT_D_INHIBITOR,
    DEFAULT_FIELD_ALPHA,
    DEFAULT_QUANTUM_JITTER_VAR,
    DEFAULT_R_ACTIVATOR,
    DEFAULT_R_INHIBITOR,
    DEFAULT_TURING_THRESHOLD,
    FIELD_V_MAX,
    FIELD_V_MIN,
    BoundaryCondition,
    ReactionDiffusionConfig,
    ReactionDiffusionEngine,
    ReactionDiffusionMetrics,
)

# Default Turing threshold (exported constant)
TURING_THRESHOLD: float = DEFAULT_TURING_THRESHOLD

# Default quantum jitter variance
QUANTUM_JITTER_VAR: float = DEFAULT_QUANTUM_JITTER_VAR


def simulate_mycelium_field(
    rng: np.random.Generator,
    grid_size: int = 64,
    steps: int = 64,
    alpha: float = DEFAULT_FIELD_ALPHA,
    spike_probability: float = 0.25,
    turing_enabled: bool = True,
    turing_threshold: float = DEFAULT_TURING_THRESHOLD,
    quantum_jitter: bool = False,
    jitter_var: float = DEFAULT_QUANTUM_JITTER_VAR,
) -> Tuple[NDArray[Any], int]:
    """
    Simulate mycelium-like potential field on 2D lattice with Turing morphogenesis.

    Model features:
    - Field V initialized around -70 mV
    - Discrete Laplacian diffusion
    - Turing reaction-diffusion morphogenesis (activator-inhibitor)
    - Optional quantum jitter for stochastic dynamics
    - Ion clamping for numerical stability

    Physics:
    - Turing threshold = 0.75 for pattern formation
    - Quantum jitter variance = 0.0005 (stable at 0.067 normalized)

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    grid_size : int
        Grid size N x N.
    steps : int
        Simulation steps.
    alpha : float
        Diffusion coefficient.
    spike_probability : float
        Probability of growth event per step.
    turing_enabled : bool
        Enable Turing morphogenesis.
    turing_threshold : float
        Threshold for Turing pattern activation.
    quantum_jitter : bool
        Enable quantum jitter noise.
    jitter_var : float
        Variance of quantum jitter.

    Returns
    -------
    field : NDArray[Any]
        Array of shape (N, N) in volts.
    growth_events : int
        Number of growth events.
    """
    # Initialize around -70 mV
    field = rng.normal(loc=-0.07, scale=0.005, size=(grid_size, grid_size))
    growth_events = 0

    # Turing activator-inhibitor system
    if turing_enabled:
        activator = rng.uniform(0, 0.1, size=(grid_size, grid_size))
        inhibitor = rng.uniform(0, 0.1, size=(grid_size, grid_size))
        da, di = DEFAULT_D_ACTIVATOR, DEFAULT_D_INHIBITOR  # diffusion rates
        ra, ri = DEFAULT_R_ACTIVATOR, DEFAULT_R_INHIBITOR  # reaction rates

    for step in range(steps):
        # Growth events (spikes)
        if rng.random() < spike_probability:
            i = int(rng.integers(0, grid_size))
            j = int(rng.integers(0, grid_size))
            field[i, j] += float(rng.normal(loc=0.02, scale=0.005))
            growth_events += 1

        # Laplacian diffusion
        up = np.roll(field, 1, axis=0)
        down = np.roll(field, -1, axis=0)
        left = np.roll(field, 1, axis=1)
        right = np.roll(field, -1, axis=1)
        laplacian = up + down + left + right - 4.0 * field
        field = field + alpha * laplacian

        # Turing morphogenesis
        if turing_enabled:
            # Laplacian for activator/inhibitor
            a_lap = (
                np.roll(activator, 1, axis=0)
                + np.roll(activator, -1, axis=0)
                + np.roll(activator, 1, axis=1)
                + np.roll(activator, -1, axis=1)
                - 4.0 * activator
            )
            i_lap = (
                np.roll(inhibitor, 1, axis=0)
                + np.roll(inhibitor, -1, axis=0)
                + np.roll(inhibitor, 1, axis=1)
                + np.roll(inhibitor, -1, axis=1)
                - 4.0 * inhibitor
            )

            # Reaction-diffusion update
            activator += da * a_lap + ra * (activator * (1 - activator) - inhibitor)
            inhibitor += di * i_lap + ri * (activator - inhibitor)

            # Apply Turing pattern to field where activator exceeds threshold
            turing_mask = activator > turing_threshold
            field[turing_mask] += 0.005

            # Clamp activator/inhibitor
            activator = np.clip(activator, 0, 1)
            inhibitor = np.clip(inhibitor, 0, 1)

        # Quantum jitter
        if quantum_jitter:
            jitter = rng.normal(0, np.sqrt(jitter_var), size=field.shape)
            field += jitter

        # Ion clamping (≈ [-95, 40] mV)
        field = np.clip(field, FIELD_V_MIN, FIELD_V_MAX)

    return field, growth_events


__all__ = [
    # Constants
    "TURING_THRESHOLD",
    "QUANTUM_JITTER_VAR",
    # Main functions
    "simulate_mycelium_field",
    # Engine classes (re-exported)
    "ReactionDiffusionEngine",
    "ReactionDiffusionConfig",
    "ReactionDiffusionMetrics",
    "BoundaryCondition",
]
