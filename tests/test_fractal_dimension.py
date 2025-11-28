import numpy as np

from mycelium_fractal_net import estimate_fractal_dimension


def test_fractal_dimension_reasonable_range() -> None:
    rng = np.random.default_rng(42)
    field = rng.random((64, 64)) > 0.7
    D = estimate_fractal_dimension(field)
    assert 0.5 <= D <= 2.0
