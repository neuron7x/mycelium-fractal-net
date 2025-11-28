"""Tests for STDP plasticity module."""
import torch

from mycelium_fractal_net import STDP_A_MINUS, STDP_A_PLUS, STDP_TAU_MINUS, STDP_TAU_PLUS
from mycelium_fractal_net.model import STDPPlasticity


def test_stdp_parameters_match_spec() -> None:
    """Verify STDP parameters match specification: tau±20ms, a±0.01/0.012."""
    stdp = STDPPlasticity()

    assert stdp.tau_plus == STDP_TAU_PLUS
    assert stdp.tau_minus == STDP_TAU_MINUS
    assert stdp.a_plus == STDP_A_PLUS
    assert stdp.a_minus == STDP_A_MINUS

    # Check actual values
    assert abs(stdp.tau_plus - 0.020) < 1e-6  # 20 ms
    assert abs(stdp.tau_minus - 0.020) < 1e-6  # 20 ms
    assert abs(stdp.a_plus - 0.01) < 1e-6
    assert abs(stdp.a_minus - 0.012) < 1e-6


def test_stdp_ltp_when_pre_before_post() -> None:
    """Test Long-Term Potentiation when presynaptic spike precedes postsynaptic."""
    stdp = STDPPlasticity()

    # Pre spike at t=0, post spike at t=0.01 (10ms later)
    pre_times = torch.tensor([[0.0]])
    post_times = torch.tensor([[0.01]])
    weights = torch.ones(1, 1)

    delta_w = stdp.compute_weight_update(pre_times, post_times, weights)

    # LTP should produce positive weight change
    assert delta_w.sum().item() > 0


def test_stdp_ltd_when_post_before_pre() -> None:
    """Test Long-Term Depression when postsynaptic spike precedes presynaptic."""
    stdp = STDPPlasticity()

    # Post spike at t=0, pre spike at t=0.01 (10ms later)
    pre_times = torch.tensor([[0.01]])
    post_times = torch.tensor([[0.0]])
    weights = torch.ones(1, 1)

    delta_w = stdp.compute_weight_update(pre_times, post_times, weights)

    # LTD should produce negative weight change
    assert delta_w.sum().item() < 0


def test_stdp_forward_pass_through() -> None:
    """Test that STDP forward pass is identity."""
    stdp = STDPPlasticity()
    x = torch.randn(4, 10)

    out = stdp(x)

    assert torch.allclose(out, x)


def test_stdp_exponential_decay() -> None:
    """Test that STDP weight update decays exponentially with time difference."""
    stdp = STDPPlasticity()

    # Test at different time delays
    delays = [0.005, 0.010, 0.020, 0.040]
    ltp_values = []

    for delay in delays:
        pre_times = torch.tensor([[0.0]])
        post_times = torch.tensor([[delay]])
        weights = torch.ones(1, 1)
        delta_w = stdp.compute_weight_update(pre_times, post_times, weights)
        ltp_values.append(delta_w.item())

    # Weight changes should decrease with larger time delays
    for i in range(len(ltp_values) - 1):
        assert ltp_values[i] > ltp_values[i + 1]
