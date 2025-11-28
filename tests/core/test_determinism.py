"""
Determinism tests for stochastic components.

Tests verify that with the same random_seed:
- Results are exactly reproducible
- Multiple runs produce identical outputs

This is critical for scientific reproducibility.
"""

from __future__ import annotations

import numpy as np

from mycelium_fractal_net.core import (
    FractalGrowthConfig,
    FractalGrowthEngine,
    MembraneEngine,
    MembraneEngineConfig,
    ReactionDiffusionConfig,
    ReactionDiffusionEngine,
)


class TestMembraneEngineDeterminism:
    """Determinism tests for membrane engine."""

    def test_same_seed_same_result(self) -> None:
        """Test same seed produces identical results."""
        config1 = MembraneEngineConfig(dt=0.001, random_seed=42)
        config2 = MembraneEngineConfig(dt=0.001, random_seed=42)

        engine1 = MembraneEngine(config1)
        engine2 = MembraneEngine(config2)

        V1, metrics1 = engine1.simulate(n_neurons=50, steps=100)
        V2, metrics2 = engine2.simulate(n_neurons=50, steps=100)

        np.testing.assert_array_equal(V1, V2)
        assert metrics1.v_mean == metrics2.v_mean
        assert metrics1.v_std == metrics2.v_std

    def test_different_seed_different_result(self) -> None:
        """Test different seeds produce different initial conditions."""
        config1 = MembraneEngineConfig(dt=0.001, random_seed=42)
        config2 = MembraneEngineConfig(dt=0.001, random_seed=123)

        engine1 = MembraneEngine(config1)
        engine2 = MembraneEngine(config2)

        # Use only 1 step to capture initial conditions before convergence
        V1, _ = engine1.simulate(n_neurons=50, steps=1)
        V2, _ = engine2.simulate(n_neurons=50, steps=1)

        # Different seeds should give different initial conditions
        # (simulations converge to equilibrium, so check early states)
        assert not np.array_equal(V1, V2)

    def test_reset_restores_determinism(self) -> None:
        """Test reset() allows reproducible re-runs."""
        config = MembraneEngineConfig(dt=0.001, random_seed=42)
        engine = MembraneEngine(config)

        V1, _ = engine.simulate(n_neurons=50, steps=100)

        engine.reset()

        V2, _ = engine.simulate(n_neurons=50, steps=100)

        np.testing.assert_array_equal(V1, V2)

    def test_with_spikes_determinism(self) -> None:
        """Test determinism with random spikes enabled."""
        config1 = MembraneEngineConfig(dt=0.001, random_seed=42)
        config2 = MembraneEngineConfig(dt=0.001, random_seed=42)

        engine1 = MembraneEngine(config1)
        engine2 = MembraneEngine(config2)

        V1, _ = engine1.simulate(
            n_neurons=50,
            steps=100,
            spike_probability=0.1,
            spike_amplitude=0.030,
        )
        V2, _ = engine2.simulate(
            n_neurons=50,
            steps=100,
            spike_probability=0.1,
            spike_amplitude=0.030,
        )

        np.testing.assert_array_equal(V1, V2)


class TestReactionDiffusionDeterminism:
    """Determinism tests for reaction-diffusion engine."""

    def test_same_seed_same_result(self) -> None:
        """Test same seed produces identical results."""
        config1 = ReactionDiffusionConfig(grid_size=32, steps=50, dt=0.1, random_seed=42)
        config2 = ReactionDiffusionConfig(grid_size=32, steps=50, dt=0.1, random_seed=42)

        engine1 = ReactionDiffusionEngine(config1)
        engine2 = ReactionDiffusionEngine(config2)

        a1, i1, m1 = engine1.simulate()
        a2, i2, m2 = engine2.simulate()

        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(i1, i2)
        assert m1.pattern_fraction == m2.pattern_fraction

    def test_different_seed_different_result(self) -> None:
        """Test different seeds produce different initial conditions."""
        config1 = ReactionDiffusionConfig(grid_size=32, steps=1, dt=0.1, random_seed=42)
        config2 = ReactionDiffusionConfig(grid_size=32, steps=1, dt=0.1, random_seed=123)

        engine1 = ReactionDiffusionEngine(config1)
        engine2 = ReactionDiffusionEngine(config2)

        # Use only 1 step to capture initial conditions before convergence
        a1, _, _ = engine1.simulate()
        a2, _, _ = engine2.simulate()

        # Initial random fields should differ
        assert not np.array_equal(a1, a2)

    def test_reset_restores_determinism(self) -> None:
        """Test reset() allows reproducible re-runs."""
        config = ReactionDiffusionConfig(grid_size=32, steps=50, dt=0.1, random_seed=42)
        engine = ReactionDiffusionEngine(config)

        a1, i1, _ = engine.simulate()

        engine.reset()

        a2, i2, _ = engine.simulate()

        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(i1, i2)

    def test_with_field_coupling_determinism(self) -> None:
        """Test determinism with field coupling."""
        config1 = ReactionDiffusionConfig(grid_size=32, steps=50, dt=0.1, random_seed=42)
        config2 = ReactionDiffusionConfig(grid_size=32, steps=50, dt=0.1, random_seed=42)

        engine1 = ReactionDiffusionEngine(config1)
        engine2 = ReactionDiffusionEngine(config2)

        field = np.full((32, 32), -0.070)

        _, _, f1, _ = engine1.simulate_with_field(field, field_coupling=0.005)
        _, _, f2, _ = engine2.simulate_with_field(field, field_coupling=0.005)

        np.testing.assert_array_equal(f1, f2)


class TestFractalGrowthDeterminism:
    """Determinism tests for fractal growth engine."""

    def test_ifs_same_seed_same_result(self) -> None:
        """Test IFS with same seed produces identical results."""
        config1 = FractalGrowthConfig(num_points=1000, num_transforms=4, random_seed=42)
        config2 = FractalGrowthConfig(num_points=1000, num_transforms=4, random_seed=42)

        engine1 = FractalGrowthEngine(config1)
        engine2 = FractalGrowthEngine(config2)

        points1, metrics1 = engine1.generate_ifs()
        points2, metrics2 = engine2.generate_ifs()

        np.testing.assert_array_equal(points1, points2)
        assert metrics1.lyapunov_exponent == metrics2.lyapunov_exponent
        assert metrics1.fractal_dimension == metrics2.fractal_dimension

    def test_ifs_different_seed_different_result(self) -> None:
        """Test IFS with different seeds produces different results."""
        config1 = FractalGrowthConfig(num_points=1000, num_transforms=4, random_seed=42)
        config2 = FractalGrowthConfig(num_points=1000, num_transforms=4, random_seed=123)

        engine1 = FractalGrowthEngine(config1)
        engine2 = FractalGrowthEngine(config2)

        points1, _ = engine1.generate_ifs()
        points2, _ = engine2.generate_ifs()

        # Different seeds should give different transforms and points
        assert not np.allclose(points1, points2)

    def test_dla_same_seed_same_result(self) -> None:
        """Test DLA with same seed produces identical results."""
        config1 = FractalGrowthConfig(grid_size=32, max_iterations=50, random_seed=42)
        config2 = FractalGrowthConfig(grid_size=32, max_iterations=50, random_seed=42)

        engine1 = FractalGrowthEngine(config1)
        engine2 = FractalGrowthEngine(config2)

        grid1, metrics1 = engine1.generate_dla()
        grid2, metrics2 = engine2.generate_dla()

        np.testing.assert_array_equal(grid1, grid2)
        assert metrics1.dla_particles == metrics2.dla_particles

    def test_reset_restores_determinism(self) -> None:
        """Test reset() allows reproducible re-runs."""
        config = FractalGrowthConfig(num_points=1000, num_transforms=4, random_seed=42)
        engine = FractalGrowthEngine(config)

        points1, _ = engine.generate_ifs()

        engine.reset()

        points2, _ = engine.generate_ifs()

        np.testing.assert_array_equal(points1, points2)

    def test_transforms_determinism(self) -> None:
        """Test that transforms are deterministically generated."""
        config1 = FractalGrowthConfig(num_points=100, num_transforms=4, random_seed=42)
        config2 = FractalGrowthConfig(num_points=100, num_transforms=4, random_seed=42)

        engine1 = FractalGrowthEngine(config1)
        engine2 = FractalGrowthEngine(config2)

        engine1.generate_ifs()
        engine2.generate_ifs()

        transforms1 = engine1.transforms
        transforms2 = engine2.transforms

        assert len(transforms1) == len(transforms2)
        for t1, t2 in zip(transforms1, transforms2):
            assert t1 == t2


class TestCrossEngineDeterminism:
    """Tests for determinism across multiple engines used together."""

    def test_combined_simulation_determinism(self) -> None:
        """Test that a combined workflow is deterministic."""
        seed = 42

        # First run
        membrane_config1 = MembraneEngineConfig(dt=0.001, random_seed=seed)
        rd_config1 = ReactionDiffusionConfig(grid_size=32, steps=50, dt=0.1, random_seed=seed)
        fractal_config1 = FractalGrowthConfig(num_points=1000, random_seed=seed)

        membrane1 = MembraneEngine(membrane_config1)
        rd1 = ReactionDiffusionEngine(rd_config1)
        fractal1 = FractalGrowthEngine(fractal_config1)

        V1, _ = membrane1.simulate(n_neurons=50, steps=100)
        a1, _, _ = rd1.simulate()
        pts1, _ = fractal1.generate_ifs()

        # Second run with same seeds
        membrane_config2 = MembraneEngineConfig(dt=0.001, random_seed=seed)
        rd_config2 = ReactionDiffusionConfig(grid_size=32, steps=50, dt=0.1, random_seed=seed)
        fractal_config2 = FractalGrowthConfig(num_points=1000, random_seed=seed)

        membrane2 = MembraneEngine(membrane_config2)
        rd2 = ReactionDiffusionEngine(rd_config2)
        fractal2 = FractalGrowthEngine(fractal_config2)

        V2, _ = membrane2.simulate(n_neurons=50, steps=100)
        a2, _, _ = rd2.simulate()
        pts2, _ = fractal2.generate_ifs()

        np.testing.assert_array_equal(V1, V2)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(pts1, pts2)
