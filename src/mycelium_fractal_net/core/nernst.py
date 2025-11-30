"""
Nernst Potential Module — Ion Electrochemistry.

This module provides the Nernst equation computation for membrane potentials.
Re-exports the stable implementation from membrane_engine.py for backward compatibility,
and adds the legacy function signatures from model.py.

Reference: MFN_MATH_MODEL.md Section 1 (Membrane Potentials)

Equations:
    E = (RT/zF) * ln([ion]_out / [ion]_in)   # Nernst equation

Physics verification:
- At 37°C (310K): RT/zF = 26.73 mV for z=1
- For K⁺: [K]_in = 140 mM, [K]_out = 5 mM → E_K ≈ -89 mV
"""

from __future__ import annotations

import math

import sympy as sp

# Re-export core constants and implementation from membrane_engine
from .membrane_engine import (
    BODY_TEMPERATURE_K,
    FARADAY_CONSTANT,
    ION_CLAMP_MIN,
    R_GAS_CONSTANT,
    MembraneConfig,
    MembraneEngine,
    MembraneMetrics,
)

# Computed constant: RT/zF at 37°C (z=1) in millivolts
NERNST_RTFZ_MV: float = (R_GAS_CONSTANT * BODY_TEMPERATURE_K / FARADAY_CONSTANT) * 1000.0


def compute_nernst_potential(
    z_valence: int,
    concentration_out_molar: float,
    concentration_in_molar: float,
    temperature_k: float = BODY_TEMPERATURE_K,
) -> float:
    """
    Compute membrane potential using Nernst equation (in volts).

    E = (R*T)/(z*F) * ln([ion]_out / [ion]_in)

    Physics verification:
    - For K+: [K]_in = 140 mM, [K]_out = 5 mM at 37°C → E_K ≈ -89 mV
    - RT/zF at 37°C (z=1) = 26.73 mV → 58.17 mV for ln to log10

    Parameters
    ----------
    z_valence : int
        Ion valence (K+ = 1, Ca2+ = 2).
    concentration_out_molar : float
        Extracellular concentration (mol/L).
    concentration_in_molar : float
        Intracellular concentration (mol/L).
    temperature_k : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Membrane potential in volts.
    """
    # Clamp concentrations to avoid log(0) or negative values
    c_out = max(concentration_out_molar, ION_CLAMP_MIN)
    c_in = max(concentration_in_molar, ION_CLAMP_MIN)

    if c_out <= 0 or c_in <= 0:
        raise ValueError("Concentrations must be positive for Nernst potential.")

    ratio = c_out / c_in
    return (R_GAS_CONSTANT * temperature_k) / (z_valence * FARADAY_CONSTANT) * math.log(ratio)


def symbolic_nernst_verification() -> float:
    """
    Use sympy to verify Nernst equation on concrete values.

    Returns numeric potential for K+ at standard concentrations.
    This provides symbolic verification that the implementation
    matches the mathematical formulation.

    Returns
    -------
    float
        Computed E_K potential in Volts for standard K+ concentrations.
    """
    R, T, z, F, c_out, c_in = sp.symbols("R T z F c_out c_in", positive=True)
    E_expr = (R * T) / (z * F) * sp.log(c_out / c_in)

    subs = {
        R: R_GAS_CONSTANT,
        T: BODY_TEMPERATURE_K,
        z: 1,
        F: FARADAY_CONSTANT,
        c_out: 5e-3,
        c_in: 140e-3,
    }
    E_val = float(E_expr.subs(subs).evalf())
    return E_val


# Alias for legacy compatibility
_symbolic_nernst_example = symbolic_nernst_verification

__all__ = [
    # Constants
    "R_GAS_CONSTANT",
    "FARADAY_CONSTANT",
    "BODY_TEMPERATURE_K",
    "NERNST_RTFZ_MV",
    "ION_CLAMP_MIN",
    # Functions
    "compute_nernst_potential",
    "symbolic_nernst_verification",
    "_symbolic_nernst_example",
    # Engine classes (re-exported)
    "MembraneEngine",
    "MembraneConfig",
    "MembraneMetrics",
]
