"""
Turing Morphogenesis Module.

This module provides the public API for Turing reaction-diffusion pattern
formation, re-exporting validated implementations from model.py for
backward compatibility, plus the new ReactionDiffusionEngine for advanced use.

Conceptual domain: Reaction-diffusion dynamics, pattern formation

Reference:
    - docs/MFN_MATH_MODEL.md Section 2 (Reaction-Diffusion Processes)
    - docs/ARCHITECTURE.md Section 2 (Mycelium Field Simulation)

Mathematical Model:
    ∂a/∂t = D_a ∇²a + r_a·a(1-a) - i    (Activator)
    ∂i/∂t = D_i ∇²i + r_i·(a - i)        (Inhibitor)

Parameters:
    D_a = 0.1 grid²/step    - Activator diffusion
    D_i = 0.05 grid²/step   - Inhibitor diffusion
    r_a = 0.01              - Activator reaction rate
    r_i = 0.02              - Inhibitor reaction rate
    θ = 0.75                - Turing activation threshold

Example:
    >>> from mycelium_fractal_net.core.turing import simulate_mycelium_field
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> field, growth_events = simulate_mycelium_field(
    ...     rng=rng,
    ...     grid_size=64,
    ...     steps=64,
    ...     turing_enabled=True
    ... )
    >>> # field: [-95, 40] mV range
"""

from __future__ import annotations

# Re-export the original simulate_mycelium_field from model.py
# to maintain exact backward compatibility
from ..model import simulate_mycelium_field

# Re-export validated reaction-diffusion engine for advanced use
from .reaction_diffusion_engine import (
    DEFAULT_D_ACTIVATOR,
    DEFAULT_D_INHIBITOR,
    DEFAULT_FIELD_ALPHA,
    DEFAULT_QUANTUM_JITTER_VAR,
    DEFAULT_R_ACTIVATOR,
    DEFAULT_R_INHIBITOR,
    DEFAULT_TURING_THRESHOLD,
    ReactionDiffusionConfig,
    ReactionDiffusionEngine,
    ReactionDiffusionMetrics,
)

# Re-exported constants
TURING_THRESHOLD = DEFAULT_TURING_THRESHOLD
D_ACTIVATOR = DEFAULT_D_ACTIVATOR
D_INHIBITOR = DEFAULT_D_INHIBITOR
R_ACTIVATOR = DEFAULT_R_ACTIVATOR
R_INHIBITOR = DEFAULT_R_INHIBITOR
FIELD_ALPHA = DEFAULT_FIELD_ALPHA
QUANTUM_JITTER_VAR = DEFAULT_QUANTUM_JITTER_VAR

__all__ = [
    # Constants
    "TURING_THRESHOLD",
    "D_ACTIVATOR",
    "D_INHIBITOR",
    "R_ACTIVATOR",
    "R_INHIBITOR",
    "FIELD_ALPHA",
    "QUANTUM_JITTER_VAR",
    # Classes
    "ReactionDiffusionConfig",
    "ReactionDiffusionEngine",
    "ReactionDiffusionMetrics",
    # Functions
    "simulate_mycelium_field",
]
