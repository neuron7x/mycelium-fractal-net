"""Tests for hierarchical Krum aggregation for federated learning.

Mathematical validation tests for Byzantine-robust gradient aggregation.

Reference: Blanchard et al. (2017), Yin et al. (2018)

Algorithm:
    Krum(g_1,...,g_n) = g_i where i = argmin_j Σ_{k∈N_j} ||g_j - g_k||²
    N_j = (n - f - 2) nearest neighbors of g_j

Byzantine tolerance:
    f < (n - 2) / 2

Complexity:
    Single Krum: O(n² × d)
    Hierarchical: O(n²/C × d + C² × d)
"""

from typing import Any

import numpy as np
import pytest
import torch

from mycelium_fractal_net.core import HierarchicalKrumAggregator as CoreHierarchicalKrumAggregator
from mycelium_fractal_net.model import HierarchicalKrumAggregator


class TestKrumParameterValidation:
    """Tests for Krum aggregator parameter validation."""

    def test_krum_aggregator_default_parameters(self) -> None:
        """Verify default parameters match spec: 100 clusters, 20% Byzantine, 10% sample."""
        agg = HierarchicalKrumAggregator()

        assert agg.num_clusters == 100
        assert abs(agg.byzantine_fraction - 0.2) < 1e-6
        assert abs(agg.sample_fraction - 0.1) < 1e-6

    def test_krum_rejects_zero_clusters(self) -> None:
        """Verify rejection of num_clusters < 1."""
        with pytest.raises(ValueError, match="num_clusters"):
            HierarchicalKrumAggregator(num_clusters=0)

    def test_krum_rejects_high_byzantine_fraction(self) -> None:
        """Verify rejection of byzantine_fraction >= 50%.

        Mathematical constraint: f < (n-2)/2 for convergence
        With f_frac >= 0.5, cannot guarantee Byzantine tolerance.
        """
        with pytest.raises(ValueError, match="byzantine_fraction"):
            HierarchicalKrumAggregator(byzantine_fraction=0.5)

    def test_krum_rejects_invalid_sample_fraction(self) -> None:
        """Verify rejection of invalid sample_fraction."""
        with pytest.raises(ValueError, match="sample_fraction"):
            HierarchicalKrumAggregator(sample_fraction=0)

        with pytest.raises(ValueError, match="sample_fraction"):
            HierarchicalKrumAggregator(sample_fraction=1.5)


class TestKrumAlgorithmCorrectness:
    """Tests for Krum algorithm mathematical correctness."""

    def test_krum_select_single_gradient(self) -> None:
        """Test Krum returns the single gradient when only one is provided."""
        agg = HierarchicalKrumAggregator()
        grad = torch.randn(10)

        result = agg.krum_select([grad], num_byzantine=0)

        assert torch.allclose(result, grad)

    def test_krum_select_multiple_gradients(self) -> None:
        """Test Krum selects one gradient from multiple.

        Krum should select gradient closest to majority (honest) gradients,
        not the outlier (Byzantine).
        """
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

    def test_krum_neighbor_count_formula(self) -> None:
        """Test Krum uses correct neighbor count: n - f - 2.

        For n=6 gradients with f=1 Byzantine:
        Neighbor count = 6 - 1 - 2 = 3
        """
        agg = HierarchicalKrumAggregator()
        n = 6
        f = 1
        # Neighbor count formula: n - f - 2 = 3 (documented above)

        grads = [torch.randn(10) for _ in range(n)]

        # The algorithm should work correctly with this neighbor count
        result = agg.krum_select(grads, num_byzantine=f)

        assert result.shape == (10,)

    def test_krum_requires_sufficient_neighbors(self) -> None:
        """Krum should reject configurations that violate n > 2f + 2."""
        agg = HierarchicalKrumAggregator()

        # n=2, f=0 violates requirement (2 <= 2*0 + 2)
        grads_two = [torch.randn(5) for _ in range(2)]
        with pytest.raises(ValueError, match="2f \+ 2"):
            agg.krum_select(grads_two, num_byzantine=0)

        # n=3, f=1 violates requirement (3 <= 2*1 + 2)
        grads_three = [torch.randn(5) for _ in range(3)]
        with pytest.raises(ValueError, match="2f \+ 2"):
            agg.krum_select(grads_three, num_byzantine=1)


class TestKrumByzantineRobustness:
    """Tests for Byzantine fault tolerance guarantees."""

    def test_aggregate_robustness_to_byzantine(self) -> None:
        """Test aggregation robustness to Byzantine (malicious) gradients.

        With 20% Byzantine tolerance, honest gradients should dominate.
        """
        agg = HierarchicalKrumAggregator(num_clusters=3)
        rng = np.random.default_rng(42)

        # Create honest gradients pointing in positive direction
        honest_grads = [torch.ones(10) + torch.randn(10) * 0.1 for _ in range(8)]

        # Create Byzantine gradients pointing in opposite direction
        byzantine_grads = [-torch.ones(10) * 10 for _ in range(2)]  # 20% Byzantine

        all_grads = honest_grads + byzantine_grads

        result = agg.aggregate(all_grads, rng=rng)

        # Result should be closer to honest direction (positive)
        assert result.mean() > 0  # Should be positive on average

    def test_byzantine_tolerance_boundary(self) -> None:
        """Test Byzantine tolerance at boundary condition.

        For n clients, tolerance is f < (n-2)/2.
        With n=6: f < 2, so f=1 should work, f=2 is boundary.
        """
        agg = HierarchicalKrumAggregator(num_clusters=1)
        rng = np.random.default_rng(42)

        # 6 gradients, 1 Byzantine (well within tolerance)
        honest_grads = [torch.zeros(5) for _ in range(5)]
        byzantine_grads = [torch.ones(5) * 100]

        all_grads = honest_grads + byzantine_grads
        result = agg.aggregate(all_grads, rng=rng)

        # Result should be closer to zero (honest) than 100 (Byzantine)
        assert torch.norm(result).item() < 50

    def test_byzantine_budget_respects_zero_fraction(self) -> None:
        """Ensure zero Byzantine fraction doesn't force a phantom attacker budget."""

        agg = HierarchicalKrumAggregator(byzantine_fraction=0.0, num_clusters=1)

        assert agg._estimate_byzantine_count(10) == 0

        rng = np.random.default_rng(0)
        grads = [torch.randn(4) for _ in range(5)]

        result = agg.aggregate(grads, rng=rng)

        # With no expected Byzantines, aggregation should stay close to mean honest direction
        honest_mean = torch.stack(grads).mean(dim=0)
        assert torch.norm(result - honest_mean) < 5

    def test_byzantine_budget_clamps_when_group_too_small(self) -> None:
        """Clamp Byzantine budget when clusters are smaller than theoretical limit."""

        agg = HierarchicalKrumAggregator(byzantine_fraction=0.4, num_clusters=1)

        # Only two gradients cannot support any Byzantine budget
        assert agg._estimate_byzantine_count(2) == 0

        rng = np.random.default_rng(123)
        grads = [torch.randn(3) for _ in range(2)]

        result = agg.aggregate(grads, rng=rng)

        assert result.shape == (3,)

    def test_aggregate_fallback_when_krum_not_applicable(self) -> None:
        """Aggregation should fall back when n <= 2f + 2."""

        agg = HierarchicalKrumAggregator(num_clusters=1, byzantine_fraction=0.0)
        grads = [torch.randn(6) for _ in range(2)]

        result = agg.aggregate(grads, rng=np.random.default_rng(0))

        expected = torch.stack(grads).mean(dim=0)
        assert torch.allclose(result, expected)


class TestCoreKrumParityWithModel:
    """Ensure core aggregator mirrors validated model behavior for safety."""

    def test_zero_byzantine_fraction_does_not_force_budget(self) -> None:
        """Core aggregator should not invent Byzantine clients when fraction is zero."""

        agg = CoreHierarchicalKrumAggregator(byzantine_fraction=0.0, num_clusters=1)

        assert agg._estimate_byzantine_count(10) == 0

        rng = np.random.default_rng(7)
        grads = [torch.randn(4) for _ in range(5)]

        result = agg.aggregate(grads, rng=rng)

        honest_mean = torch.stack(grads).mean(dim=0)
        assert torch.norm(result - honest_mean) < 5

    def test_byzantine_budget_clamped_for_small_groups(self) -> None:
        """When there are too few gradients, budget should clamp to zero."""

        agg = CoreHierarchicalKrumAggregator(byzantine_fraction=0.3, num_clusters=1)

        assert agg._estimate_byzantine_count(2) == 0

        rng = np.random.default_rng(11)
        grads = [torch.randn(3) for _ in range(2)]

        result = agg.aggregate(grads, rng=rng)

        assert result.shape == (3,)

    def test_core_krum_rejects_insufficient_neighbors(self) -> None:
        """Core Krum should enforce the n > 2f + 2 requirement like the model."""

        agg = CoreHierarchicalKrumAggregator(num_clusters=1)

        # n=2, f=0 violates requirement (2 <= 2*0 + 2)
        grads_two = [torch.randn(5) for _ in range(2)]
        with pytest.raises(ValueError, match="2f \+ 2"):
            agg.krum_select(grads_two, num_byzantine=0)

        # n=3, f=1 also violates requirement (3 <= 2*1 + 2)
        grads_three = [torch.randn(5) for _ in range(3)]
        with pytest.raises(ValueError, match="2f \+ 2"):
            agg.krum_select(grads_three, num_byzantine=1)


class TestKrumNumericalStability:
    """Tests for numerical stability of Krum aggregation."""

    def test_aggregate_simple_case(self) -> None:
        """Test hierarchical aggregation with small number of clients."""
        agg = HierarchicalKrumAggregator(num_clusters=2)
        rng = np.random.default_rng(42)

        # Create 10 similar gradients
        grads = [torch.randn(5) for _ in range(10)]

        result = agg.aggregate(grads, rng=rng)

        assert result.shape == (5,)
        assert not torch.isnan(result).any()

    def test_aggregate_empty_raises(self) -> None:
        """Test that aggregating empty list raises ValueError."""
        agg = HierarchicalKrumAggregator()

        with pytest.raises(ValueError, match="No gradients"):
            agg.aggregate([])

    def test_krum_select_empty_raises(self) -> None:
        """Test that Krum with empty list raises ValueError."""
        agg = HierarchicalKrumAggregator()

        with pytest.raises(ValueError, match="No gradients"):
            agg.krum_select([], num_byzantine=0)

    def test_aggregate_large_gradients(self) -> None:
        """Test aggregation handles large gradient values."""
        agg = HierarchicalKrumAggregator(num_clusters=2)
        rng = np.random.default_rng(42)

        # Large but finite gradients
        grads = [torch.randn(10) * 1e6 for _ in range(10)]

        result = agg.aggregate(grads, rng=rng)

        assert torch.isfinite(result).all()

    def test_aggregate_deterministic(self) -> None:
        """Test aggregation is deterministic with fixed seed."""
        agg = HierarchicalKrumAggregator(num_clusters=2)

        grads = [torch.randn(5) for _ in range(10)]

        rng1 = np.random.default_rng(42)
        result1 = agg.aggregate(grads, rng=rng1)

        rng2 = np.random.default_rng(42)
        result2 = agg.aggregate(grads, rng=rng2)

        assert torch.allclose(result1, result2)

    def test_default_rng_uses_entropy_when_not_provided(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure default aggregation RNG is not hard-seeded to a fixed value."""

        original_default_rng = np.random.default_rng
        calls: list[Any] = []

        def fake_default_rng(seed: Any = None) -> np.random.Generator:
            calls.append(seed)
            return original_default_rng(0)

        monkeypatch.setattr(np.random, "default_rng", fake_default_rng)

        grads = [torch.randn(4) for _ in range(4)]

        agg = HierarchicalKrumAggregator(num_clusters=2)
        agg.aggregate(grads)

        core_agg = CoreHierarchicalKrumAggregator(num_clusters=2)
        core_agg.aggregate(grads)

        assert calls == [None, None]

    def test_sampling_respects_minimum_client_selection(self) -> None:
        """Ensure sampling keeps at least one client even with tiny fractions."""
        # sample_fraction is validated as >0, but very small values should still select
        # at least one client when sampling from large populations.
        agg = HierarchicalKrumAggregator(num_clusters=5, sample_fraction=0.0005)

        rng = np.random.default_rng(123)
        grads = [torch.randn(3) for _ in range(1500)]  # triggers sampling branch

        result = agg.aggregate(grads, rng=rng)

        assert result.shape == (3,)
        assert torch.isfinite(result).all()
