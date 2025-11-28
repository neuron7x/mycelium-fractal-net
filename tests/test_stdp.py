"""Tests for STDP (Spike-Timing-Dependent Plasticity) functions."""
import math

import numpy as np

from mycelium_fractal_net import (
    compute_heterosynaptic_modulation,
    compute_stdp_weight_change,
    verify_stdp_lipschitz,
)
from mycelium_fractal_net.model import STDP_A_MINUS, STDP_A_PLUS, STDP_TAU_MS


def test_stdp_ltp_positive_delta() -> None:
    """Pre before post (delta_t > 0) should produce LTP (positive weight change)."""
    delta_w = compute_stdp_weight_change(delta_t_ms=10.0)
    assert delta_w > 0
    assert delta_w <= STDP_A_PLUS  # Bounded by A+


def test_stdp_ltd_negative_delta() -> None:
    """Post before pre (delta_t < 0) should produce LTD (negative weight change)."""
    delta_w = compute_stdp_weight_change(delta_t_ms=-10.0)
    assert delta_w < 0
    assert abs(delta_w) <= STDP_A_MINUS  # Bounded by A-


def test_stdp_zero_delta() -> None:
    """At delta_t = 0, weight change should be zero."""
    delta_w = compute_stdp_weight_change(delta_t_ms=0.0)
    assert delta_w == 0.0


def test_stdp_asymmetric() -> None:
    """STDP should be asymmetric: |LTD| != |LTP| at same |delta_t|."""
    ltp = compute_stdp_weight_change(delta_t_ms=10.0)
    ltd = compute_stdp_weight_change(delta_t_ms=-10.0)
    # A- > A+ for homeostasis, so |LTD| > LTP
    assert abs(ltd) != ltp


def test_stdp_exponential_decay() -> None:
    """STDP should decay exponentially with time constant tau."""
    dw_10 = compute_stdp_weight_change(delta_t_ms=10.0)
    dw_30 = compute_stdp_weight_change(delta_t_ms=30.0)
    # At 30ms vs 10ms with tau=20ms
    expected_ratio = math.exp(-30 / STDP_TAU_MS) / math.exp(-10 / STDP_TAU_MS)
    actual_ratio = dw_30 / dw_10
    assert abs(actual_ratio - expected_ratio) < 0.01


def test_stdp_constants() -> None:
    """Verify STDP constants match specification."""
    assert STDP_TAU_MS == 20.0  # tau=20ms per spec
    assert STDP_A_PLUS == 0.01
    assert STDP_A_MINUS == 0.012


def test_stdp_lipschitz_verification() -> None:
    """Verify STDP satisfies Lipschitz bound < 0.01."""
    satisfies, constant = verify_stdp_lipschitz(epsilon=0.01)
    # Default parameters: max(0.01, 0.012) / 20 = 0.0006 < 0.01
    assert satisfies is True
    assert constant < 0.01
    assert constant == max(STDP_A_PLUS, STDP_A_MINUS) / STDP_TAU_MS


def test_heterosynaptic_modulation_empty() -> None:
    """Empty spike history should return 0 modulation."""
    spike_history = np.array([])
    g_n = compute_heterosynaptic_modulation(spike_history)
    assert g_n == 0.0


def test_heterosynaptic_modulation_range() -> None:
    """Modulation factor should be in [0, 1]."""
    rng = np.random.default_rng(42)
    spike_history = rng.integers(0, 2, size=(10, 100))
    g_n = compute_heterosynaptic_modulation(spike_history)
    assert 0.0 <= g_n <= 1.0


def test_heterosynaptic_modulation_with_spikes() -> None:
    """Modulation increases with more spikes."""
    low_activity = np.zeros((10, 100), dtype=int)
    low_activity[0, 0] = 1  # One spike

    high_activity = np.ones((10, 100), dtype=int)

    g_low = compute_heterosynaptic_modulation(low_activity)
    g_high = compute_heterosynaptic_modulation(high_activity)

    assert g_high > g_low
