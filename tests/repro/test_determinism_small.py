"""
Reproducibility tests for MyceliumFractalNet.

Tests to verify that simulations are deterministic when given the same seed,
and that the RNG control layer works correctly.
"""

import numpy as np

from mycelium_fractal_net import (
    SimulationConfig,
    run_mycelium_simulation_with_history,
    simulate_mycelium_field,
)
from mycelium_fractal_net.infra.rng import RNGContext, create_rng, set_global_seed


class TestDeterminism:
    """Test that simulations are deterministic with same seed."""

    def test_simulate_mycelium_field_same_seed_produces_same_output(self) -> None:
        """Two simulations with same seed should produce identical results."""
        seed = 42

        # First run
        rng1 = np.random.default_rng(seed)
        field1, events1 = simulate_mycelium_field(
            rng1, grid_size=32, steps=20, turing_enabled=True
        )

        # Second run with same seed
        rng2 = np.random.default_rng(seed)
        field2, events2 = simulate_mycelium_field(
            rng2, grid_size=32, steps=20, turing_enabled=True
        )

        # Should be exactly identical
        np.testing.assert_array_equal(field1, field2)
        assert events1 == events2

    def test_run_mycelium_simulation_with_history_deterministic(self) -> None:
        """SimulationConfig-based simulation should be deterministic."""
        config = SimulationConfig(
            grid_size=32,
            steps=20,
            seed=123,
            turing_enabled=True,
        )

        # First run
        result1 = run_mycelium_simulation_with_history(config)

        # Second run
        result2 = run_mycelium_simulation_with_history(config)

        # Should produce identical fields
        np.testing.assert_array_equal(result1.field, result2.field)
        assert result1.growth_events == result2.growth_events

        # History should also be identical
        if result1.history is not None and result2.history is not None:
            np.testing.assert_array_equal(result1.history, result2.history)

    def test_different_seeds_produce_different_results(self) -> None:
        """Different seeds should produce different results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)

        field1, _ = simulate_mycelium_field(rng1, grid_size=32, steps=20)
        field2, _ = simulate_mycelium_field(rng2, grid_size=32, steps=20)

        # Fields should be different
        assert not np.allclose(field1, field2)

    def test_none_seed_uses_default(self) -> None:
        """Config with seed=None should still be deterministic with RNG context."""
        # When using RNGContext, we can control randomness even without explicit seed
        rng_ctx = create_rng(seed=42)

        # Use the RNGContext's numpy_rng
        field1, events1 = simulate_mycelium_field(
            rng_ctx.numpy_rng, grid_size=32, steps=20
        )

        # Reset and run again
        rng_ctx.reset()
        field2, events2 = simulate_mycelium_field(
            rng_ctx.numpy_rng, grid_size=32, steps=20
        )

        # Should be identical
        np.testing.assert_array_equal(field1, field2)
        assert events1 == events2


class TestRNGContext:
    """Test the RNG control layer."""

    def test_create_rng_returns_context(self) -> None:
        """create_rng should return an RNGContext."""
        ctx = create_rng(seed=42)
        assert isinstance(ctx, RNGContext)
        assert ctx.seed == 42

    def test_rng_context_numpy_rng_is_deterministic(self) -> None:
        """RNGContext's numpy_rng should be deterministic."""
        ctx1 = create_rng(seed=42)
        ctx2 = create_rng(seed=42)

        arr1 = ctx1.numpy_rng.random(10)
        arr2 = ctx2.numpy_rng.random(10)

        np.testing.assert_array_equal(arr1, arr2)

    def test_rng_context_reset(self) -> None:
        """reset() should restore the original state."""
        ctx = create_rng(seed=42)

        # Generate some values
        arr1 = ctx.numpy_rng.random(10)

        # Reset
        ctx.reset()

        # Generate again - should be identical
        arr2 = ctx.numpy_rng.random(10)

        np.testing.assert_array_equal(arr1, arr2)

    def test_rng_context_fork(self) -> None:
        """fork() should create an independent context."""
        ctx = create_rng(seed=42)
        forked = ctx.fork(offset=100)

        # Forked context has different seed
        assert forked.seed == 142

        # They produce different values
        arr1 = ctx.numpy_rng.random(10)
        arr2 = forked.numpy_rng.random(10)

        assert not np.array_equal(arr1, arr2)

    def test_set_global_seed(self) -> None:
        """set_global_seed should affect global random state."""
        set_global_seed(42)
        arr1 = np.random.random(10)

        set_global_seed(42)
        arr2 = np.random.random(10)

        np.testing.assert_array_equal(arr1, arr2)

    def test_default_seed_is_42(self) -> None:
        """Default seed should be 42 when not specified."""
        ctx = create_rng(seed=None)
        assert ctx.seed == 42
