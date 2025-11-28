import math

from mycelium_fractal_net import compute_nernst_potential, symbolic_nernst_verify


def test_nernst_potassium_physiological_range() -> None:
    e_v = compute_nernst_potential(
        z_valence=1,
        concentration_out_molar=5e-3,
        concentration_in_molar=140e-3,
    )
    e_mv = e_v * 1000.0
    assert -95.0 < e_mv < -80.0
    assert math.isfinite(e_mv)


def test_nernst_potassium_target_89mv() -> None:
    """Verify E_K = -89mV as specified in problem statement."""
    e_v = compute_nernst_potential(
        z_valence=1,
        concentration_out_molar=5e-3,
        concentration_in_molar=140e-3,
    )
    e_mv = e_v * 1000.0
    # Target: E_K ≈ -89mV with tolerance ±1mV
    assert -90.0 < e_mv < -88.0, f"E_K = {e_mv:.2f} mV, expected ~-89mV"


def test_nernst_symbolic_numeric_match() -> None:
    """Verify symbolic and numeric Nernst calculations match."""
    e_symbolic = symbolic_nernst_verify() * 1000.0
    e_numeric = compute_nernst_potential(1, 5e-3, 140e-3) * 1000.0

    # Should match to floating-point precision
    assert abs(e_symbolic - e_numeric) < 1e-6


def test_nernst_rt_zf_ln_constant() -> None:
    """Verify RT/zF * ln factor is ~25.9mV at body temperature."""
    from mycelium_fractal_net.model import (
        BODY_TEMPERATURE_K,
        FARADAY_CONSTANT,
        R_GAS_CONSTANT,
    )

    # RT/zF for z=1 at 310K
    rt_zf = (R_GAS_CONSTANT * BODY_TEMPERATURE_K) / (1 * FARADAY_CONSTANT) * 1000  # mV
    # Should be ~26.7mV at 310K, or ~25.9mV at 300K
    # At 310K: 8.314 * 310 / 96485 * 1000 = 26.7mV
    assert 25.0 < rt_zf < 28.0


def test_nernst_sodium_positive() -> None:
    """Sodium potential should be positive (higher outside)."""
    # Typical Na+ concentrations: [Na]out = 145mM, [Na]in = 12mM
    e_na_v = compute_nernst_potential(
        z_valence=1,
        concentration_out_molar=145e-3,
        concentration_in_molar=12e-3,
    )
    e_na_mv = e_na_v * 1000.0
    # E_Na should be positive (~+60mV)
    assert 50.0 < e_na_mv < 70.0


def test_nernst_invalid_concentrations() -> None:
    """Should raise error for non-positive concentrations."""
    import pytest

    with pytest.raises(ValueError):
        compute_nernst_potential(1, 0.0, 140e-3)

    with pytest.raises(ValueError):
        compute_nernst_potential(1, 5e-3, -1.0)

