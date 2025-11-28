"""Tests for hierarchical Krum aggregation for federated learning."""
import numpy as np
import torch

from mycelium_fractal_net.model import HierarchicalKrumAggregator


def test_krum_aggregator_default_parameters() -> None:
    """Verify default parameters match spec: 100 clusters, 20% Byzantine, 10% sample."""
    agg = HierarchicalKrumAggregator()

    assert agg.num_clusters == 100
    assert abs(agg.byzantine_fraction - 0.2) < 1e-6
    assert abs(agg.sample_fraction - 0.1) < 1e-6


def test_krum_select_single_gradient() -> None:
    """Test Krum returns the single gradient when only one is provided."""
    agg = HierarchicalKrumAggregator()
    grad = torch.randn(10)

    result = agg.krum_select([grad], num_byzantine=0)

    assert torch.allclose(result, grad)


def test_krum_select_multiple_gradients() -> None:
    """Test Krum selects one gradient from multiple."""
    agg = HierarchicalKrumAggregator()

    # Create similar gradients (honest) and one outlier (Byzantine)
    honest_grads = [torch.randn(10) for _ in range(5)]
    byzantine_grad = torch.randn(10) * 100  # Outlier

    all_grads = honest_grads + [byzantine_grad]

    result = agg.krum_select(all_grads, num_byzantine=1)

    # Result should be closer to honest gradients than Byzantine
    honest_mean = torch.stack(honest_grads).mean(dim=0)
    dist_to_honest = torch.norm(result - honest_mean)
    dist_to_byzantine = torch.norm(result - byzantine_grad)

    assert dist_to_honest < dist_to_byzantine


def test_aggregate_simple_case() -> None:
    """Test hierarchical aggregation with small number of clients."""
    agg = HierarchicalKrumAggregator(num_clusters=2)
    rng = np.random.default_rng(42)

    # Create 10 similar gradients
    grads = [torch.randn(5) for _ in range(10)]

    result = agg.aggregate(grads, rng=rng)

    assert result.shape == (5,)
    assert not torch.isnan(result).any()


def test_aggregate_robustness_to_byzantine() -> None:
    """Test aggregation robustness to Byzantine (malicious) gradients."""
    agg = HierarchicalKrumAggregator(num_clusters=3)
    rng = np.random.default_rng(42)

    # Create honest gradients pointing in positive direction
    honest_grads = [torch.ones(10) + torch.randn(10) * 0.1 for _ in range(8)]

    # Create Byzantine gradients pointing in opposite direction
    byzantine_grads = [-torch.ones(10) * 10 for _ in range(2)]  # 20% Byzantine

    all_grads = honest_grads + byzantine_grads

    result = agg.aggregate(all_grads, rng=rng)

    # Result should be closer to honest direction (positive)
    # Mean of honest gradients is approximately ones(10)
    assert result.mean() > 0  # Should be positive on average


def test_aggregate_empty_raises() -> None:
    """Test that aggregating empty list raises ValueError."""
    agg = HierarchicalKrumAggregator()

    try:
        agg.aggregate([])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No gradients" in str(e)


def test_krum_select_empty_raises() -> None:
    """Test that Krum with empty list raises ValueError."""
    agg = HierarchicalKrumAggregator()

    try:
        agg.krum_select([], num_byzantine=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No gradients" in str(e)
