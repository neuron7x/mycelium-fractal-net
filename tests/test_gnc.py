"""Tests for General Neuromodulatory Control (GNC+) v2.0."""

import numpy as np
import pytest

from mycelium_fractal_net.neurochem.gnc import (
    GNCBridge,
    GNCState,
    MODULATORS,
    SIGMA,
    THETA,
    compute_gnc_state,
    gnc_diagnose,
    step,
)


class TestGNCConstants:
    def test_theta_dimension(self):
        assert len(THETA) == 9

    def test_modulators_dimension(self):
        assert len(MODULATORS) == 7

    def test_sigma_shape(self):
        assert len(SIGMA) == 7
        for m in MODULATORS:
            assert len(SIGMA[m]) == 9


class TestGNCState:
    def test_default_state(self):
        state = GNCState.default()
        assert all(state.modulators[m] == 0.5 for m in MODULATORS)
        assert all(state.theta[t] == 0.5 for t in THETA)

    def test_from_levels(self):
        levels = {"Glutamate": 0.7, "GABA": 0.3, "Dopamine": 0.8}
        state = GNCState.from_levels(levels)
        assert state.modulators["Glutamate"] == 0.7
        assert state.modulators["GABA"] == 0.3
        assert state.modulators["Dopamine"] == 0.8
        # Missing modulators default to 0.5
        assert state.modulators["Opioid"] == 0.5
        # Theta computed from levels
        assert all(0.1 <= state.theta[t] <= 0.9 for t in THETA)

    def test_to_dict(self):
        state = GNCState.default()
        d = state.to_dict()
        assert "modulators" in d
        assert "theta" in d

    def test_summary(self):
        state = compute_gnc_state({"Dopamine": 0.8, "GABA": 0.3})
        text = state.summary()
        assert "GNC+" in text
        assert "Dopamine" in text

    def test_clipping(self):
        state = GNCState.from_levels({"Glutamate": 1.5, "GABA": -0.5})
        assert state.modulators["Glutamate"] == 1.0
        assert state.modulators["GABA"] == 0.0


class TestSigma:
    def test_dopamine_signs(self):
        s = SIGMA["Dopamine"]
        assert s["nu"] == +1   # reward valuation up
        assert s["tau"] == -1  # inhibitory threshold down
        assert s["beta"] == -1 # policy stability down (explore)

    def test_gaba_signs(self):
        s = SIGMA["GABA"]
        assert s["alpha"] == -1  # learning rate down (stabilize)
        assert s["beta"] == +1   # policy stability up
        assert s["tau"] == +1    # inhibitory threshold up


class TestStep:
    def test_step_returns_new_state(self):
        state = GNCState.default()
        next_state = step(state)
        assert isinstance(next_state, GNCState)
        # Should be different due to noise
        assert next_state.theta != state.theta

    def test_step_bounded(self):
        state = compute_gnc_state({"Dopamine": 0.9, "Noradrenaline": 0.9})
        for _ in range(10):
            state = step(state)
        assert all(0.1 <= state.theta[t] <= 0.9 for t in THETA)


class TestDiagnosis:
    def test_optimal(self):
        state = GNCState.default()
        diag = gnc_diagnose(state)
        assert diag.regime in ("optimal", "dysregulated")
        assert 0 <= diag.coherence <= 1

    def test_hyperactivated(self):
        state = compute_gnc_state({m: 0.85 for m in MODULATORS})
        diag = gnc_diagnose(state)
        assert diag.regime == "hyperactivated"

    def test_hypoactivated(self):
        state = compute_gnc_state({m: 0.15 for m in MODULATORS})
        diag = gnc_diagnose(state)
        assert diag.regime == "hypoactivated"

    def test_summary(self):
        state = compute_gnc_state({"Dopamine": 0.8, "GABA": 0.2})
        diag = gnc_diagnose(state)
        text = diag.summary()
        assert "GNC+" in text
        assert diag.dominant_axis in MODULATORS
        assert diag.suppressed_axis in MODULATORS

    def test_falsification_flags(self):
        state = compute_gnc_state({"Noradrenaline": 0.9, "GABA": 0.1})
        diag = gnc_diagnose(state)
        # Should have some structure (may or may not have flags)
        assert isinstance(diag.falsification_flags, list)


class TestGNCBridge:
    def test_modulate_anomaly(self):
        bridge = GNCBridge()
        raw = 0.5
        mod = bridge.modulate_anomaly_score(raw)
        assert 0 <= mod <= 1

    def test_high_na_boosts(self):
        bridge = GNCBridge(GNCState.from_levels({"Noradrenaline": 0.9}))
        raw = 0.3
        mod = bridge.modulate_anomaly_score(raw)
        assert mod > raw  # NA boosts sensitivity

    def test_high_gaba_dampens(self):
        bridge = GNCBridge(GNCState.from_levels({"GABA": 0.9}))
        raw = 0.3
        mod = bridge.modulate_anomaly_score(raw)
        assert mod < raw  # GABA dampens

    def test_update_from_m_score(self):
        bridge = GNCBridge()
        bridge.update_from_m_score(0.4)
        assert bridge.state.modulators["Dopamine"] == pytest.approx(0.8, abs=0.01)

    def test_activate_resilience(self):
        bridge = GNCBridge()
        before = bridge.state.modulators["Opioid"]
        bridge.activate_resilience()
        assert bridge.state.modulators["Opioid"] > before

    def test_summary(self):
        bridge = GNCBridge()
        text = bridge.summary()
        assert "GNC+" in text
