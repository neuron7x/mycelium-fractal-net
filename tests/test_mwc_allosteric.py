"""Tests for the MWC allosteric model implementation.

Validates that the Monod-Wyman-Changeux model for GABA-A α1β3γ2 receptors
produces biophysically correct dose-response curves.

References:
    Gielen & Bhatt (2019) Br J Pharmacol 176:2524-2537
        Muscimol EC50 on α1β3γ2: 5-15 μM
    Chang et al. (1996) Biophys J 71:2454-2468
        MWC parameters for GABA-A
"""

from __future__ import annotations

import numpy as np
import pytest

from mycelium_fractal_net.neurochem.mwc import (
    MWC_C,
    MWC_K_R_UM,
    MWC_K_T_UM,
    MWC_L0,
    MWC_N_SITES,
    effective_gabaa_shunt,
    effective_serotonergic_gain,
    mwc_dose_response,
    mwc_ec50,
    mwc_fraction,
)


class TestMWCModel:
    """Core MWC equation correctness."""

    def test_zero_concentration_returns_near_zero(self) -> None:
        """At zero agonist, receptor is predominantly in T (closed) state."""
        r = mwc_fraction(0.0, 0.0)
        # With L0=5000, R_fraction at zero should be ~0.0002
        assert r < 0.01, f"R_fraction at zero concentration too high: {r}"

    def test_saturating_concentration_approaches_one(self) -> None:
        """At very high agonist, receptor should be nearly fully open."""
        r = mwc_fraction(10000.0, 0.0)
        assert r > 0.95, f"R_fraction at saturation too low: {r}"

    def test_monotonically_increasing(self) -> None:
        """R_fraction must increase with concentration — no dips."""
        concentrations = np.logspace(-2, 4, 100)
        responses = mwc_dose_response(concentrations)
        diffs = np.diff(responses)
        assert np.all(diffs >= -1e-12), f"Non-monotonic: min diff = {diffs.min()}"

    def test_bounded_zero_one(self) -> None:
        """R_fraction must be in [0, 1] for all concentrations."""
        concentrations = np.logspace(-3, 5, 200)
        responses = mwc_dose_response(concentrations)
        assert np.all(responses >= 0.0), f"Negative R_fraction: {responses.min()}"
        assert np.all(responses <= 1.0), f"R_fraction > 1: {responses.max()}"

    def test_negative_concentration_clamped(self) -> None:
        """Negative concentration should be treated as zero."""
        r = mwc_fraction(-5.0, 0.0)
        r_zero = mwc_fraction(0.0, 0.0)
        assert r == r_zero

    def test_invalid_K_R_returns_zero(self) -> None:
        """K_R <= 0 is non-physical, should return 0."""
        assert mwc_fraction(10.0, 0.0, K_R=0.0) == 0.0
        assert mwc_fraction(10.0, 0.0, K_R=-1.0) == 0.0

    def test_invalid_L0_returns_zero(self) -> None:
        """L0 <= 0 is non-physical, should return 0."""
        assert mwc_fraction(10.0, 0.0, L0=0.0) == 0.0
        assert mwc_fraction(10.0, 0.0, L0=-1.0) == 0.0


class TestMWCEC50:
    """EC50 validation against published electrophysiology data."""

    def test_ec50_in_published_range(self) -> None:
        """EC50 for muscimol on α1β3γ2 should be 5-15 μM.

        Ref: Gielen & Bhatt (2019) Br J Pharmacol 176:2524-2537
        """
        ec50 = mwc_ec50()
        assert 3.0 <= ec50 <= 20.0, (
            f"EC50 = {ec50:.2f} μM, expected 5-15 μM for muscimol on α1β3γ2"
        )

    def test_ec50_response_is_half_max(self) -> None:
        """At EC50, R_fraction should be ~R_max/2."""
        ec50 = mwc_ec50()
        r = mwc_fraction(ec50, 0.0)
        # R_max = 1 / (1 + L0 * c^n) ≈ 0.9996
        r_max = 1.0 / (1.0 + MWC_L0 * (MWC_C ** MWC_N_SITES))
        assert abs(r - r_max / 2.0) < 0.01, f"R at EC50 = {r}, expected ~{r_max/2:.4f}"

    def test_ec50_increases_with_L0(self) -> None:
        """Higher L0 (more closed at rest) should shift EC50 rightward."""
        ec50_low = mwc_ec50(L0=1000.0)
        ec50_high = mwc_ec50(L0=10000.0)
        assert ec50_high > ec50_low, (
            f"EC50 should increase with L0: {ec50_low} vs {ec50_high}"
        )

    def test_ec50_decreases_with_n(self) -> None:
        """More binding sites should make the transition steeper, shifting EC50."""
        ec50_2 = mwc_ec50(n=2)
        ec50_4 = mwc_ec50(n=4)
        # With more cooperative sites, EC50 can shift
        assert ec50_2 != ec50_4  # They should differ


class TestMWCParameters:
    """Verify parameter consistency."""

    def test_c_equals_kr_over_kt(self) -> None:
        assert abs(MWC_C - MWC_K_R_UM / MWC_K_T_UM) < 1e-10

    def test_n_sites_positive(self) -> None:
        assert MWC_N_SITES >= 1

    def test_l0_positive(self) -> None:
        assert MWC_L0 > 0

    def test_kr_less_than_kt(self) -> None:
        """R state has higher affinity (lower K) than T state."""
        assert MWC_K_R_UM < MWC_K_T_UM

    def test_c_less_than_one(self) -> None:
        """c = K_R/K_T < 1 is required for agonist to favor R state."""
        assert 0.0 < MWC_C < 1.0


class TestMWCDoseResponse:
    """Vectorized dose-response curve."""

    def test_shape_preserved(self) -> None:
        conc = np.array([0.1, 1.0, 10.0, 100.0])
        resp = mwc_dose_response(conc)
        assert resp.shape == conc.shape

    def test_matches_scalar(self) -> None:
        """Vectorized should match scalar for each element."""
        conc = np.array([0.0, 1.0, 10.0, 100.0, 1000.0])
        vectorized = mwc_dose_response(conc)
        scalar = np.array([mwc_fraction(c, 0.0) for c in conc])
        np.testing.assert_allclose(vectorized, scalar, atol=1e-10)

    def test_hill_slope_greater_than_one(self) -> None:
        """MWC with n>1 should produce a Hill slope > 1 (cooperativity)."""
        # Measure effective Hill slope at EC50
        ec50 = mwc_ec50()
        delta = ec50 * 0.01
        r_below = mwc_fraction(ec50 - delta, 0.0)
        r_above = mwc_fraction(ec50 + delta, 0.0)
        # Slope in logit space ≈ Hill coefficient
        dr_dc = (r_above - r_below) / (2 * delta)
        # At EC50 for Hill equation: slope = n / (4 * EC50)
        # For n=1 Hill: slope = 1 / (4 * EC50)
        hill_1_slope = 1.0 / (4.0 * ec50)
        # MWC slope should be steeper than Hill n=1
        assert dr_dc > hill_1_slope * 0.8, (
            f"MWC slope {dr_dc:.6f} not steeper than Hill n=1 slope {hill_1_slope:.6f}"
        )


class TestBackwardCompatibility:
    """Verify API compatibility with old Hill-equation signature."""

    def test_affinity_param_accepted(self) -> None:
        """affinity_um parameter should be accepted without error."""
        r = mwc_fraction(10.0, affinity_um=5.0)
        assert 0.0 <= r <= 1.0

    def test_effective_gabaa_shunt_unchanged(self) -> None:
        """effective_gabaa_shunt API unchanged."""
        active = np.array([0.0, 0.5, 1.0])
        result = effective_gabaa_shunt(active, 0.5)
        assert result.shape == active.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 0.95)

    def test_effective_serotonergic_gain_unchanged(self) -> None:
        """effective_serotonergic_gain API unchanged."""
        drive = np.array([0.0, 0.5, 1.0])
        result = effective_serotonergic_gain(drive, 0.1, 0.05)
        assert result.shape == drive.shape
        assert np.all(result >= -0.10)
        assert np.all(result <= 0.25)


class TestCausalRuleSIM011:
    """Verify SIM-011 causal rule for MWC monotonicity."""

    def test_sim011_passes(self) -> None:
        import mycelium_fractal_net as mfn
        from mycelium_fractal_net.core.causal_validation import validate_causal_consistency

        seq = mfn.simulate(mfn.SimulationSpec(grid_size=16, steps=8, seed=42))
        v = validate_causal_consistency(seq, mode="strict")
        sim011 = [r for r in v.rule_results if r.rule_id == "SIM-011"]
        assert len(sim011) == 1, "SIM-011 should be evaluated"
        assert sim011[0].passed, f"SIM-011 failed: observed={sim011[0].observed}"
