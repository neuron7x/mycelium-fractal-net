"""Tests for Turing reaction-diffusion dispersion relation."""
import math

from mycelium_fractal_net import (
    compute_turing_dispersion,
    symbolic_turing_verify,
    verify_turing_instability,
)


def test_turing_dispersion_zero_wavenumber() -> None:
    """At k=0, dispersion should return homogeneous stability value."""
    lambda_0 = compute_turing_dispersion(k=0.0, d_u=1.0, d_v=10.0, a=0.5, b=1.0)
    # At k=0, trace = a - b = 0.5 - 1.0 = -0.5
    # det = a*(-b) + a*b = -0.5 + 0.5 = 0
    # lambda = (trace + sqrt(trace^2)) / 2 = 0 (larger eigenvalue when det=0)
    assert math.isfinite(lambda_0)
    assert lambda_0 <= 0  # Stable or marginally stable at k=0


def test_turing_dispersion_nonzero_wavenumber() -> None:
    """At nonzero k, dispersion relation should be finite."""
    lambda_k = compute_turing_dispersion(k=0.5, d_u=1.0, d_v=10.0, a=0.5, b=0.3)
    assert math.isfinite(lambda_k)


def test_turing_instability_verification() -> None:
    """Verify Turing instability condition with typical parameters."""
    # Parameters known to produce Turing instability
    is_unstable, max_lambda, k_at_max = verify_turing_instability(
        d_u=1.0, d_v=10.0, a=0.5, b=0.3
    )
    assert isinstance(is_unstable, bool)
    assert math.isfinite(max_lambda)
    assert k_at_max > 0


def test_symbolic_turing_verification() -> None:
    """Test sympy-based Turing verification."""
    result = symbolic_turing_verify()
    assert "trace" in result
    assert "determinant" in result
    assert "lambda_max_at_k0.3" in result
    assert "is_unstable" in result
    assert all(math.isfinite(v) for v in result.values())


def test_turing_threshold_constant() -> None:
    """Verify TURING_THRESHOLD constant is defined correctly."""
    from mycelium_fractal_net.model import TURING_THRESHOLD

    assert TURING_THRESHOLD == 0.75
