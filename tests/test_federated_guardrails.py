import pytest
import torch

from mycelium_fractal_net.core.federated import HierarchicalKrumAggregator


def test_krum_select_requires_enough_clients():
    aggregator = HierarchicalKrumAggregator()
    gradients = [torch.zeros(4), torch.ones(4)]

    with pytest.raises(ValueError):
        aggregator.krum_select(gradients, num_byzantine=0)


def test_krum_select_enforces_byzantine_budget_constraint():
    aggregator = HierarchicalKrumAggregator()
    gradients = [torch.zeros(4) for _ in range(3)]

    with pytest.raises(ValueError):
        aggregator.krum_select(gradients, num_byzantine=1)
