"""Tests for 1D fractal denoiser."""

from __future__ import annotations

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from mycelium_fractal_net.signal import OptimizedFractalDenoise1D


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def test_denoiser_preserves_shape_cpu() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    model = OptimizedFractalDenoise1D()
    data = torch.randn(1, 1, 1024)

    with torch.no_grad():
        output = model(data)

    assert output.shape == data.shape
    assert output.device.type == "cpu"


def test_denoiser_do_no_harm_random_walk() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)

    steps = rng.normal(0.0, 0.01, size=1024)
    base = np.cumsum(steps)
    noise = rng.normal(0.0, 0.02, size=1024)
    noisy = base + noise

    model = OptimizedFractalDenoise1D()
    tensor = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        denoised = model(tensor).squeeze(0).squeeze(0).cpu().numpy()

    mse_noisy = _mse(noisy, base)
    mse_denoised = _mse(denoised, base)

    assert mse_denoised <= mse_noisy * 0.9


def test_denoiser_improves_spike_noise_simple() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)

    length = 512
    base = np.zeros(length, dtype=np.float64)
    base[length // 3: 2 * length // 3] = 0.5

    noisy = base + rng.normal(0.0, 0.05, size=length)
    spike_indices = rng.choice(length, size=8, replace=False)
    noisy[spike_indices] += rng.choice([-2.0, 2.0], size=8)

    model = OptimizedFractalDenoise1D()
    tensor = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        denoised = model(tensor).squeeze(0).squeeze(0).cpu().numpy()

    mse_noisy = _mse(noisy, base)
    mse_denoised = _mse(denoised, base)

    assert mse_denoised < mse_noisy * 0.6


def test_denoiser_multichannel_shape_and_finite() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    data = torch.randn(2, 3, 512)
    model = OptimizedFractalDenoise1D()
    with torch.no_grad():
        out = model(data)
    assert out.shape == data.shape
    assert torch.isfinite(out).all()


@settings(max_examples=10, deadline=500)
@given(
    batch=st.integers(min_value=1, max_value=2),
    channels=st.integers(min_value=1, max_value=3),
    length=st.integers(min_value=32, max_value=96),
)
def test_denoiser_property_same_shape_and_finite(batch: int, channels: int, length: int) -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    data = torch.randn(batch, channels, length)
    model = OptimizedFractalDenoise1D()
    with torch.no_grad():
        out = model(data)
    assert out.shape == data.shape
    assert torch.isfinite(out).all()
