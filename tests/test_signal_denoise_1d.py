"""Tests for 1D fractal denoiser."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from mycelium_fractal_net.signal import OptimizedFractalDenoise1D

MODE_CONFIGS = [
    {},
    {"mode": "fractal", "population_size": 64, "range_size": 8, "iterations_fractal": 1},
]

SPIKE_IMPROVEMENT_RATIO = 0.98  # require at least modest improvement on spikes
RANDOM_WALK_DRIFT_RATIO = 0.10  # allow at most 10% relative change on random walk


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


@pytest.mark.parametrize("mode_kwargs", MODE_CONFIGS)
def test_denoiser_preserves_shape_cpu(mode_kwargs: dict[str, object]) -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    model = OptimizedFractalDenoise1D(**mode_kwargs)
    data = torch.randn(1, 1, 256)

    with torch.no_grad():
        output = model(data)

    assert output.shape == data.shape
    assert output.device.type == "cpu"


def test_fractal_improves_spikes_mse() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    rng = np.random.default_rng(0)

    length = 256
    base = np.zeros(length, dtype=np.float64)
    base[length // 4 : length // 2] = 0.8
    base[length // 2 : 3 * length // 4] = -0.3

    noisy = base + rng.normal(0.0, 0.05, size=length)
    spike_indices = rng.choice(length, size=10, replace=False)
    noisy[spike_indices] += rng.choice([-1.5, 1.5], size=10)

    model = OptimizedFractalDenoise1D(
        mode="fractal",
        population_size=64,
        range_size=8,
        iterations_fractal=1,
    )
    tensor = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        denoised = model(tensor).squeeze(0).squeeze(0).cpu().numpy()

    mse_noisy = _mse(noisy, base)
    mse_denoised = _mse(denoised, base)

    assert mse_denoised <= mse_noisy * SPIKE_IMPROVEMENT_RATIO


def test_fractal_do_no_harm_random_walk() -> None:
    torch.manual_seed(1)
    np.random.seed(1)
    rng = np.random.default_rng(1)

    steps = rng.normal(0.0, 0.02, size=512)
    base = np.cumsum(steps)
    noise = rng.normal(0.0, 0.03, size=512)
    noisy = base + noise

    model = OptimizedFractalDenoise1D(
        mode="fractal",
        population_size=32,
        range_size=8,
        iterations_fractal=1,
        s_max=0.3,
        s_threshold=10.0,
        overlap=False,
    )
    tensor = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        denoised = model(tensor).squeeze(0).squeeze(0).cpu().numpy()

    mse_noisy = _mse(noisy, base)
    mse_denoised = _mse(denoised, base)
    relative = abs(mse_denoised - mse_noisy) / max(mse_noisy, 1e-12)

    assert relative < RANDOM_WALK_DRIFT_RATIO


@pytest.mark.parametrize("mode_kwargs", MODE_CONFIGS)
def test_multichannel_does_not_crash(mode_kwargs: dict[str, object]) -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    data = torch.randn(2, 3, 128)
    model = OptimizedFractalDenoise1D(**mode_kwargs)
    with torch.no_grad():
        out = model(data)
    assert out.shape == data.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("mode_kwargs", MODE_CONFIGS)
@settings(max_examples=25, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=2),
    channels=st.integers(min_value=1, max_value=3),
    length=st.integers(min_value=32, max_value=96),
)
def test_denoiser_property_same_shape_and_finite(
    mode_kwargs: dict[str, object],
    batch: int,
    channels: int,
    length: int,
) -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    data = torch.randn(batch, channels, length)
    model = OptimizedFractalDenoise1D(**mode_kwargs)
    with torch.no_grad():
        out = model(data)
    assert out.shape == data.shape
    assert torch.isfinite(out).all()
