import numpy as np

from mycelium_fractal_net.integration.adapters import aggregate_gradients_adapter
from mycelium_fractal_net.integration.schemas import FederatedAggregateRequest
from mycelium_fractal_net.integration.service_context import ServiceContext


def test_aggregate_gradients_adapter_respects_context_seed() -> None:
    """Aggregation should be reproducible when contexts share the same seed."""
    gradients = [list((i + 1) * np.ones(3, dtype=np.float32)) for i in range(4)]
    request = FederatedAggregateRequest(
        gradients=gradients,
        num_clusters=2,
        byzantine_fraction=0.0,
    )

    ctx_a = ServiceContext(seed=123)
    ctx_b = ServiceContext(seed=123)

    res_a = aggregate_gradients_adapter(request, ctx_a).aggregated_gradient
    res_b = aggregate_gradients_adapter(request, ctx_b).aggregated_gradient

    assert res_a == res_b
