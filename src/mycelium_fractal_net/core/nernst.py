"""
Nernst-Planck Electrochemistry Module.

This module provides the public API for Nernst potential computation,
re-exporting validated implementations for backward compatibility.

Conceptual domain: Ion electrochemistry, membrane potentials

Reference:
    - docs/MFN_MATH_MODEL.md Section 1 (Membrane Potentials)
    - docs/ARCHITECTURE.md Section 1 (Nernst Equation)

Mathematical Model:
    E = (RT/zF) * ln([ion]_out / [ion]_in)
    
    At 37°C (310K): RT/zF = 26.73 mV for z=1
    For K⁺: [K]_in = 140 mM, [K]_out = 5 mM → E_K ≈ -89 mV

Example:
    >>> from mycelium_fractal_net.core.nernst import compute_nernst_potential
    >>> E_K = compute_nernst_potential(
    ...     z_valence=1,
    ...     concentration_out_molar=5e-3,   # [K⁺]out = 5 mM
    ...     concentration_in_molar=140e-3,  # [K⁺]in = 140 mM
    ...     temperature_k=310.0             # 37°C
    ... )
    >>> # E_K ≈ -0.08901 V ≈ -89 mV
"""

from __future__ import annotations

# Re-export the original implementation from model.py for backward compatibility
from ..model import compute_nernst_potential

# Re-export validated engine for advanced use with strict validation
from .membrane_engine import (
    BODY_TEMPERATURE_K,
    FARADAY_CONSTANT,
    ION_CLAMP_MIN,
    R_GAS_CONSTANT,
    TEMPERATURE_MAX_K,
    TEMPERATURE_MIN_K,
    MembraneConfig,
    MembraneEngine,
    MembraneMetrics,
)

__all__ = [
    # Physical constants
    "R_GAS_CONSTANT",
    "FARADAY_CONSTANT",
    "BODY_TEMPERATURE_K",
    "ION_CLAMP_MIN",
    "TEMPERATURE_MIN_K",
    "TEMPERATURE_MAX_K",
    # Classes (advanced use with strict validation)
    "MembraneConfig",
    "MembraneEngine",
    "MembraneMetrics",
    # Functions
    "compute_nernst_potential",
]
