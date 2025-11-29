"""
Tests for MembraneEngine — Nernst potential and ODE integration.

Validates:
- Nernst equation correctness (MFN_MATH_MODEL.md Section 1)
- Numerical stability (no NaN/Inf)
- Ion clamping behavior
- ODE integration schemes (Euler, RK4)
"""

import math

import numpy as np
import pytest

from mycelium_fractal_net.core import (
    MembraneConfig,
    MembraneEngine,
    ValueOutOfRangeError,
)
from mycelium_fractal_net.core.membrane_engine import (
    BODY_TEMPERATURE_K,
    ION_CLAMP_MIN,
    IntegrationScheme,
)


class TestMembraneConfig:
    """Test MembraneConfig validation."""

    def test_default_config_valid(self) -> None:
        """Default configuration should be valid."""
        config = MembraneConfig()
        assert config.temperature_k == BODY_TEMPERATURE_K
        assert config.ion_clamp_min == ION_CLAMP_MIN

    def test_negative_temperature_raises(self) -> None:
        """Negative temperature should raise ValueOutOfRangeError."""
        with pytest.raises(ValueOutOfRangeError, match="Temperature"):
            MembraneConfig(temperature_k=-10.0)

    def test_negative_ion_clamp_raises(self) -> None:
        """Negative ion clamp should raise ValueOutOfRangeError."""
        with pytest.raises(ValueOutOfRangeError, match="clamp"):
            MembraneConfig(ion_clamp_min=-1e-6)

    def test_negative_dt_raises(self) -> None:
        """Negative time step should raise ValueOutOfRangeError."""
        with pytest.raises(ValueOutOfRangeError, match="Time step"):
            MembraneConfig(dt=-0.001)

    def test_invalid_potential_range_raises(self) -> None:
        """Invalid potential range should raise."""
        with pytest.raises(ValueOutOfRangeError, match="potential"):
            MembraneConfig(potential_min_v=0.1, potential_max_v=-0.1)


class TestNernstPotential:
    """Test Nernst equation calculations."""

    def test_potassium_standard(self) -> None:
        """Test K+ potential at standard conditions: E_K ≈ -89 mV.
        
        Reference: MFN_MATH_MODEL.md Section 1.4
        [K]_in = 140 mM, [K]_out = 5 mM → E_K ≈ -89 mV
        """
        engine = MembraneEngine()
        e_k = engine.compute_nernst_potential(
            z_valence=1,
            concentration_out_molar=5e-3,
            concentration_in_molar=140e-3,
        )
        e_k_mv = e_k * 1000.0
        assert -95.0 < e_k_mv < -80.0, f"E_K = {e_k_mv:.2f} mV, expected ~-89 mV"

    def test_sodium_standard(self) -> None:
        """Test Na+ potential at standard conditions: E_Na ≈ +65 mV.
        
        Reference: MFN_MATH_MODEL.md Section 1.4
        [Na]_in = 12 mM, [Na]_out = 145 mM → E_Na ≈ +65 mV
        """
        engine = MembraneEngine()
        e_na = engine.compute_nernst_potential(
            z_valence=1,
            concentration_out_molar=145e-3,
            concentration_in_molar=12e-3,
        )
        e_na_mv = e_na * 1000.0
        assert 55.0 < e_na_mv < 75.0, f"E_Na = {e_na_mv:.2f} mV, expected ~+65 mV"

    def test_calcium_standard(self) -> None:
        """Test Ca2+ potential at standard conditions: E_Ca ≈ +129 mV.
        
        Reference: MFN_MATH_MODEL.md Section 1.4
        [Ca]_in = 0.0001 mM, [Ca]_out = 2 mM, z=2 → E_Ca ≈ +129 mV
        """
        engine = MembraneEngine()
        e_ca = engine.compute_nernst_potential(
            z_valence=2,
            concentration_out_molar=2e-3,
            concentration_in_molar=0.0001e-3,
        )
        e_ca_mv = e_ca * 1000.0
        assert e_ca_mv > 100.0, f"E_Ca = {e_ca_mv:.2f} mV, expected >100 mV"

    def test_chloride_standard(self) -> None:
        """Test Cl- potential at standard conditions.
        
        Reference: MFN_MATH_MODEL.md Section 1.4
        [Cl]_in = 4 mM, [Cl]_out = 120 mM, z=-1 → E_Cl ≈ -89 mV
        """
        engine = MembraneEngine()
        e_cl = engine.compute_nernst_potential(
            z_valence=-1,
            concentration_out_molar=120e-3,
            concentration_in_molar=4e-3,
        )
        e_cl_mv = e_cl * 1000.0
        assert -100.0 < e_cl_mv < -80.0, f"E_Cl = {e_cl_mv:.2f} mV"

    def test_zero_valence_raises(self) -> None:
        """Zero valence should raise ValueError."""
        engine = MembraneEngine()
        with pytest.raises(ValueOutOfRangeError, match="valence"):
            engine.compute_nernst_potential(
                z_valence=0,
                concentration_out_molar=5e-3,
                concentration_in_molar=140e-3,
            )

    def test_ion_clamping_prevents_nan(self) -> None:
        """Ion clamping should prevent NaN for very small concentrations."""
        engine = MembraneEngine()
        e = engine.compute_nernst_potential(
            z_valence=1,
            concentration_out_molar=1e-15,  # Very small
            concentration_in_molar=140e-3,
        )
        assert math.isfinite(e), "Clamping should prevent NaN"
        assert engine.metrics.clamping_events > 0

    def test_equal_concentrations_zero(self) -> None:
        """Equal concentrations should give zero potential."""
        engine = MembraneEngine()
        e = engine.compute_nernst_potential(
            z_valence=1,
            concentration_out_molar=100e-3,
            concentration_in_molar=100e-3,
        )
        assert abs(e) < 1e-10, f"E = {e*1000:.6f} mV, expected 0"

    def test_sign_consistency(self) -> None:
        """Verify sign consistency: [X]_out > [X]_in and z > 0 → E > 0."""
        engine = MembraneEngine()
        
        # Higher outside → positive potential (for cations)
        e_pos = engine.compute_nernst_potential(
            z_valence=1,
            concentration_out_molar=100e-3,
            concentration_in_molar=10e-3,
        )
        assert e_pos > 0
        
        # Lower outside → negative potential (for cations)
        e_neg = engine.compute_nernst_potential(
            z_valence=1,
            concentration_out_molar=10e-3,
            concentration_in_molar=100e-3,
        )
        assert e_neg < 0


class TestNernstPotentialArray:
    """Test vectorized Nernst calculations."""

    def test_array_computation(self) -> None:
        """Test batch computation produces correct results."""
        engine = MembraneEngine()
        c_out = np.array([5e-3, 145e-3, 2e-3])
        c_in = np.array([140e-3, 12e-3, 0.1e-3])
        z = 1
        
        e = engine.compute_nernst_potential_array(z, c_out, c_in)
        
        assert e.shape == (3,)
        assert np.all(np.isfinite(e))
        
        # Check metrics updated
        assert engine.metrics.potential_min_v == pytest.approx(float(np.min(e)), abs=1e-10)
        assert engine.metrics.potential_max_v == pytest.approx(float(np.max(e)), abs=1e-10)


class TestODEIntegration:
    """Test ODE integration schemes."""

    def test_euler_stability(self) -> None:
        """Test Euler integration maintains stability."""
        config = MembraneConfig(
            integration_scheme=IntegrationScheme.EULER,
            dt=1e-4,
        )
        engine = MembraneEngine(config)
        
        # Simple decay ODE: dV/dt = -V (stable)
        def decay(v: np.ndarray) -> np.ndarray:
            return -v
        
        v0 = np.array([0.01])  # 10 mV
        v_final, metrics = engine.integrate_ode(v0, decay, steps=100)
        
        assert np.isfinite(v_final).all()
        assert metrics.nan_detected is False
        assert metrics.inf_detected is False
        assert metrics.steps_computed == 100

    def test_rk4_stability(self) -> None:
        """Test RK4 integration maintains stability."""
        config = MembraneConfig(
            integration_scheme=IntegrationScheme.RK4,
            dt=1e-4,
        )
        engine = MembraneEngine(config)
        
        def decay(v: np.ndarray) -> np.ndarray:
            return -v
        
        v0 = np.array([0.01])
        v_final, metrics = engine.integrate_ode(v0, decay, steps=100)
        
        assert np.isfinite(v_final).all()
        assert metrics.steps_computed == 100

    def test_clamping_during_integration(self) -> None:
        """Test potential clamping during integration."""
        config = MembraneConfig(dt=1e-3)
        engine = MembraneEngine(config)
        
        # Growth ODE that would exceed bounds
        def growth(v: np.ndarray) -> np.ndarray:
            return np.ones_like(v) * 100  # Strong positive growth
        
        v0 = np.array([0.0])
        v_final, metrics = engine.integrate_ode(v0, growth, steps=100, clamp=True)
        
        # Should be clamped to max
        assert float(v_final[0]) <= config.potential_max_v
        assert metrics.clamping_events > 0


class TestDeterminism:
    """Test reproducibility with fixed seeds."""

    def test_same_seed_same_result(self) -> None:
        """Same seed should produce identical results."""
        config1 = MembraneConfig(random_seed=42)
        config2 = MembraneConfig(random_seed=42)
        
        engine1 = MembraneEngine(config1)
        engine2 = MembraneEngine(config2)
        
        # Both should compute identical potentials
        e1 = engine1.compute_nernst_potential(1, 5e-3, 140e-3)
        e2 = engine2.compute_nernst_potential(1, 5e-3, 140e-3)
        
        assert e1 == e2


class TestStabilitySmoke:
    """Stability smoke tests — run N steps and verify no NaN/Inf."""

    def test_smoke_1000_nernst_calculations(self) -> None:
        """Run 1000 Nernst calculations without NaN/Inf."""
        engine = MembraneEngine()
        rng = np.random.default_rng(42)
        
        for _ in range(1000):
            z = int(rng.choice([1, 2, -1, -2]))
            c_out = float(rng.uniform(1e-5, 1.0))
            c_in = float(rng.uniform(1e-5, 1.0))
            
            e = engine.compute_nernst_potential(z, c_out, c_in)
            assert math.isfinite(e), f"NaN/Inf for z={z}, c_out={c_out}, c_in={c_in}"

    def test_smoke_ode_integration_1000_steps(self) -> None:
        """Run ODE integration for 1000 steps without NaN/Inf."""
        config = MembraneConfig(dt=1e-4)
        engine = MembraneEngine(config)
        
        def oscillator(v: np.ndarray) -> np.ndarray:
            # Simple damped oscillator
            return -0.1 * v + 0.01 * np.sin(v * 100)
        
        v0 = np.array([-0.07])  # -70 mV
        v_final, metrics = engine.integrate_ode(v0, oscillator, steps=1000)
        
        assert np.isfinite(v_final).all()
        assert metrics.nan_detected is False
        assert metrics.inf_detected is False


class TestValidation:
    """Test validation methods."""

    def test_validate_potential_range_physiological(self) -> None:
        """Test physiological range validation."""
        engine = MembraneEngine()
        
        # Within physiological range
        assert engine.validate_potential_range(-0.070, strict_physiological=True)
        assert engine.validate_potential_range(-0.090, strict_physiological=True)
        assert engine.validate_potential_range(0.030, strict_physiological=True)
        
        # Outside physiological range
        assert not engine.validate_potential_range(-0.120, strict_physiological=True)
        assert not engine.validate_potential_range(0.100, strict_physiological=True)

    def test_validate_potential_range_physical(self) -> None:
        """Test physical (wider) range validation."""
        engine = MembraneEngine()
        
        # Within physical range
        assert engine.validate_potential_range(-0.120, strict_physiological=False)
        assert engine.validate_potential_range(0.100, strict_physiological=False)
        
        # Outside physical range
        assert not engine.validate_potential_range(-0.200, strict_physiological=False)
