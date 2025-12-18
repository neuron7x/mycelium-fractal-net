import numpy as np

from mycelium_fractal_net.integration.ws_adapters import _compute_fractal_dimension


def test_compute_fractal_dimension_handles_zero_activity() -> None:
    """Zero-activity fields should return a stable dimension of 0.0."""
    field = np.zeros((16, 16), dtype=float)

    dimension = _compute_fractal_dimension(field, threshold=0.5)

    assert dimension == 0.0
