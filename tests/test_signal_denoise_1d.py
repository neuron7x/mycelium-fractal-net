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


@pytest.mark.parametrize(
    ("shape", "mode_kwargs"),
    [
        ((1024,), {}),
        ((1, 1024), {}),
        ((2, 3, 512), {}),
        ((1024,), {"mode": "fractal", "population_size": 32, "range_size": 8}),
        ((2, 3, 512), {"mode": "fractal", "population_size": 32, "range_size": 8}),
    ],
)
def test_shape_invariants(shape: tuple[int, ...], mode_kwargs: dict[str, object]) -> None:
    torch.manual_seed(123)
    np.random.seed(123)
    data = torch.randn(*shape)
    model = OptimizedFractalDenoise1D(**mode_kwargs)
    with torch.no_grad():
        out = model(data)
    assert out.shape == data.shape


@pytest.mark.parametrize("mode_kwargs", MODE_CONFIGS)
def test_outputs_finite(mode_kwargs: dict[str, object]) -> None:
    torch.manual_seed(7)
    np.random.seed(7)
    data = torch.randn(2, 3, 128)
    model = OptimizedFractalDenoise1D(**mode_kwargs)
    with torch.no_grad():
        out = model(data)
    assert torch.isfinite(out).all()


def test_fractal_do_no_harm_random_walk() -> None:
    torch.manual_seed(1)
    np.random.seed(1)
    rng = np.random.default_rng(1)

    steps = rng.normal(0.0, 0.1, size=256)
    base = np.cumsum(steps)
    noise = rng.normal(0.0, 0.05, size=256)
    noisy = base + noise

    model = OptimizedFractalDenoise1D(
        mode="fractal",
        population_size=32,
        range_size=8,
        iterations_fractal=1,
        overlap=False,
        do_no_harm=True,
        harm_ratio=0.90,
        s_max=0.5,
        s_threshold=0.01,
    )
    tensor = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        denoised = model(tensor).squeeze(0).squeeze(0).cpu().numpy()

    mse_noisy = _mse(noisy, base)
    mse_denoised = _mse(denoised, base)
    assert mse_denoised <= mse_noisy * 1.10


def test_fractal_improves_spikes_mse() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    rng = np.random.default_rng(0)

    length = 256
    x = np.linspace(0, 2 * np.pi, length)
    base = 0.2 * np.sin(x)
    base[length // 3 : 2 * length // 3] += 0.5

    noisy = base + rng.normal(0.0, 0.05, size=length)
    spike_indices = rng.choice(length, size=10, replace=False)
    noisy[spike_indices] += rng.choice([-1.0, 1.0], size=10)

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
    assert np.isfinite(denoised).all()


def test_determinism_with_fixed_seed() -> None:
    params = {
        "mode": "fractal",
        "population_size": 64,
        "range_size": 8,
        "iterations_fractal": 2,
        "overlap": False,
    }
    torch.manual_seed(99)
    np.random.seed(99)
    data = torch.randn(1, 1, 128)
    model = OptimizedFractalDenoise1D(**params)
    with torch.no_grad():
        out1 = model(data).clone()

    torch.manual_seed(99)
    np.random.seed(99)
    data_repeat = torch.randn(1, 1, 128)
    model_repeat = OptimizedFractalDenoise1D(**params)
    with torch.no_grad():
        out2 = model_repeat(data_repeat).clone()

    assert torch.equal(out1, out2)


@pytest.mark.parametrize("mode_kwargs", MODE_CONFIGS)
@settings(max_examples=15, deadline=None)
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
