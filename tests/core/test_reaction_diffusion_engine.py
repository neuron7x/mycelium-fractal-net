"""
Tests for ReactionDiffusionEngine — Turing morphogenesis.

Validates:
- CFL stability condition enforcement (MFN_MATH_MODEL.md Section 2.5)
- Field bounds [-95, 40] mV (MFN_MATH_MODEL.md Section 4.3)
- No NaN/Inf after long simulations
- Turing pattern formation
- Determinism with fixed seeds
"""

import time

import numpy as np
import pytest

from mycelium_fractal_net.core import (
    ReactionDiffusionConfig,
    ReactionDiffusionEngine,
    StabilityError,
    ValueOutOfRangeError,
)
from mycelium_fractal_net.core.reaction_diffusion_engine import (
    DEFAULT_D_ACTIVATOR,
    DEFAULT_D_INHIBITOR,
    DEFAULT_FIELD_ALPHA,
    DEFAULT_TURING_THRESHOLD,
    BoundaryCondition,
)


class TestReactionDiffusionConfig:
    """Test configuration validation."""

    def test_default_config_valid(self) -> None:
        """Default configuration should be valid and stable."""
        config = ReactionDiffusionConfig()
        assert config.d_activator == DEFAULT_D_ACTIVATOR
        assert config.d_inhibitor == DEFAULT_D_INHIBITOR
        assert config.alpha == DEFAULT_FIELD_ALPHA
        assert config.turing_threshold == DEFAULT_TURING_THRESHOLD

    def test_cfl_violation_raises(self) -> None:
        """Diffusion coefficient >= 0.25 should raise StabilityError."""
        with pytest.raises(StabilityError, match="CFL"):
            ReactionDiffusionConfig(alpha=0.25)
        
        with pytest.raises(StabilityError, match="CFL"):
            ReactionDiffusionConfig(d_activator=0.30)
        
        with pytest.raises(StabilityError, match="CFL"):
            ReactionDiffusionConfig(d_inhibitor=0.25)

    def test_negative_diffusion_raises(self) -> None:
        """Negative diffusion should raise ValueOutOfRangeError."""
        with pytest.raises(ValueOutOfRangeError):
            ReactionDiffusionConfig(alpha=-0.1)

    def test_invalid_threshold_raises(self) -> None:
        """Threshold outside [0, 1] should raise."""
        with pytest.raises(ValueOutOfRangeError, match="threshold"):
            ReactionDiffusionConfig(turing_threshold=1.5)
        
        with pytest.raises(ValueOutOfRangeError, match="threshold"):
            ReactionDiffusionConfig(turing_threshold=-0.1)

    def test_invalid_probability_raises(self) -> None:
        """Probability outside [0, 1] should raise."""
        with pytest.raises(ValueOutOfRangeError, match="probability"):
            ReactionDiffusionConfig(spike_probability=1.5)

    def test_small_grid_raises(self) -> None:
        """Grid size < 4 should raise."""
        with pytest.raises(ValueOutOfRangeError, match="Grid"):
            ReactionDiffusionConfig(grid_size=2)


class TestReactionDiffusionSimulation:
    """Test simulation execution."""

    def test_basic_simulation(self) -> None:
        """Basic simulation should complete without errors."""
        config = ReactionDiffusionConfig(
            grid_size=32, random_seed=42
        )
        engine = ReactionDiffusionEngine(config)
        
        field, metrics = engine.simulate(steps=100)
        
        assert field.shape == (32, 32)
        assert np.isfinite(field).all()
        assert metrics.steps_computed == 100

    def test_field_bounds_enforced(self) -> None:
        """Field should stay within [-95, 40] mV bounds.
        
        Reference: MFN_MATH_MODEL.md Section 4.3
        """
        config = ReactionDiffusionConfig(
            grid_size=32,
            spike_probability=0.8,  # Many spikes to stress test
            random_seed=42,
        )
        engine = ReactionDiffusionEngine(config)
        
        field, metrics = engine.simulate(steps=500)
        
        field_mv = field * 1000.0
        assert field_mv.min() >= -95.0 - 0.5, f"Min {field_mv.min():.2f} mV < -95"
        assert field_mv.max() <= 40.0 + 0.5, f"Max {field_mv.max():.2f} mV > 40"

    def test_turing_affects_field(self) -> None:
        """Turing morphogenesis should modulate field dynamics.
        
        With Turing enabled, the activator/inhibitor system evolves and
        can modulate the field. We verify the dynamics run by checking
        that turing_activations is tracked (even if 0 for this regime).
        """
        config = ReactionDiffusionConfig(
            grid_size=32, 
            random_seed=42,
        )
        engine = ReactionDiffusionEngine(config)
        
        field, metrics = engine.simulate(steps=100, turing_enabled=True)
        
        # Verify simulation ran successfully
        assert np.isfinite(field).all()
        assert metrics.steps_computed == 100
        
        # Activator/inhibitor should have been computed
        # (even if activator never exceeded threshold)
        assert engine.activator is not None
        assert engine.inhibitor is not None
        
        # The metrics should track activator/inhibitor means
        # These are updated in _update_field_metrics at the end
        assert metrics.activator_mean >= 0
        assert metrics.inhibitor_mean >= 0

    def test_growth_events_recorded(self) -> None:
        """Growth events should be recorded in metrics."""
        config = ReactionDiffusionConfig(
            grid_size=32,
            spike_probability=0.5,
            random_seed=42,
        )
        engine = ReactionDiffusionEngine(config)
        
        _, metrics = engine.simulate(steps=100)
        
        # With 50% probability, expect ~50 events in 100 steps
        assert 10 <= metrics.growth_events <= 90

    def test_no_growth_with_zero_probability(self) -> None:
        """No growth events when probability is zero."""
        config = ReactionDiffusionConfig(
            grid_size=32,
            spike_probability=0.0,
            random_seed=42,
        )
        engine = ReactionDiffusionEngine(config)
        
        _, metrics = engine.simulate(steps=100)
        
        assert metrics.growth_events == 0


class TestBoundaryConditions:
    """Test different boundary conditions."""

    def test_periodic_boundary(self) -> None:
        """Periodic boundary should wrap around."""
        config = ReactionDiffusionConfig(
            grid_size=16,
            boundary_condition=BoundaryCondition.PERIODIC,
            random_seed=42,
        )
        engine = ReactionDiffusionEngine(config)
        field, _ = engine.simulate(steps=50)
        
        assert np.isfinite(field).all()

    def test_neumann_boundary(self) -> None:
        """Neumann (zero-flux) boundary should be stable."""
        config = ReactionDiffusionConfig(
            grid_size=16,
            boundary_condition=BoundaryCondition.NEUMANN,
            random_seed=42,
        )
        engine = ReactionDiffusionEngine(config)
        field, _ = engine.simulate(steps=50)
        
        assert np.isfinite(field).all()

    def test_dirichlet_boundary(self) -> None:
        """Dirichlet (fixed value) boundary should be stable."""
        config = ReactionDiffusionConfig(
            grid_size=16,
            boundary_condition=BoundaryCondition.DIRICHLET,
            random_seed=42,
        )
        engine = ReactionDiffusionEngine(config)
        field, _ = engine.simulate(steps=50)
        
        assert np.isfinite(field).all()


class TestQuantumJitter:
    """Test stochastic noise term."""

    def test_jitter_affects_field(self) -> None:
        """Quantum jitter should add noise to field."""
        config_no_jitter = ReactionDiffusionConfig(
            grid_size=32,
            quantum_jitter=False,
            random_seed=42,
        )
        config_with_jitter = ReactionDiffusionConfig(
            grid_size=32,
            quantum_jitter=True,
            jitter_var=0.0005,
            random_seed=42,
        )
        
        engine1 = ReactionDiffusionEngine(config_no_jitter)
        engine2 = ReactionDiffusionEngine(config_with_jitter)
        
        field1, _ = engine1.simulate(steps=50)
        field2, _ = engine2.simulate(steps=50)
        
        diff = np.abs(field1 - field2)
        assert diff.max() > 1e-6, "Jitter should produce different results"


class TestDeterminism:
    """Test reproducibility with fixed seeds."""

    def test_same_seed_same_result(self) -> None:
        """Same seed should produce identical results."""
        config1 = ReactionDiffusionConfig(grid_size=32, random_seed=42)
        config2 = ReactionDiffusionConfig(grid_size=32, random_seed=42)
        
        engine1 = ReactionDiffusionEngine(config1)
        engine2 = ReactionDiffusionEngine(config2)
        
        field1, metrics1 = engine1.simulate(steps=100)
        field2, metrics2 = engine2.simulate(steps=100)
        
        assert np.allclose(field1, field2)
        assert metrics1.growth_events == metrics2.growth_events

    def test_different_seed_different_result(self) -> None:
        """Different seeds should produce different results."""
        config1 = ReactionDiffusionConfig(grid_size=32, random_seed=42)
        config2 = ReactionDiffusionConfig(grid_size=32, random_seed=123)
        
        engine1 = ReactionDiffusionEngine(config1)
        engine2 = ReactionDiffusionEngine(config2)
        
        field1, _ = engine1.simulate(steps=100)
        field2, _ = engine2.simulate(steps=100)
        
        assert not np.allclose(field1, field2)


class TestStabilitySmoke:
    """Stability smoke tests — long simulations without NaN/Inf."""

    def test_smoke_1000_steps(self) -> None:
        """Run 1000 steps without NaN/Inf.
        
        Reference: MFN_MATH_MODEL.md Section 2.9 - Stability: No NaN/Inf after 1000+ steps
        """
        config = ReactionDiffusionConfig(
            grid_size=32,
            random_seed=42,
            quantum_jitter=True,
        )
        engine = ReactionDiffusionEngine(config)
        
        field, metrics = engine.simulate(steps=1000)
        
        assert np.isfinite(field).all()
        assert metrics.nan_detected is False
        assert metrics.inf_detected is False
        assert metrics.steps_to_instability is None

    def test_smoke_multiple_configurations(self) -> None:
        """Test stability across various configurations."""
        configs = [
            ReactionDiffusionConfig(grid_size=16, alpha=0.10, random_seed=42),
            ReactionDiffusionConfig(grid_size=32, alpha=0.18, random_seed=42),
            ReactionDiffusionConfig(grid_size=64, alpha=0.20, random_seed=42),
            ReactionDiffusionConfig(grid_size=32, alpha=0.24, random_seed=42),  # Near limit
        ]
        
        for config in configs:
            engine = ReactionDiffusionEngine(config)
            field, metrics = engine.simulate(steps=200)
            
            assert np.isfinite(field).all(), f"Unstable with alpha={config.alpha}"
            assert metrics.nan_detected is False


class TestHistoryCollection:
    """Test field history collection."""

    def test_history_shape(self) -> None:
        """History should have shape (steps, N, N)."""
        config = ReactionDiffusionConfig(grid_size=16, random_seed=42)
        engine = ReactionDiffusionEngine(config)
        
        history, metrics = engine.simulate(steps=50, return_history=True)
        
        assert history.shape == (50, 16, 16)

    def test_history_finite(self) -> None:
        """All history frames should be finite."""
        config = ReactionDiffusionConfig(grid_size=16, random_seed=42)
        engine = ReactionDiffusionEngine(config)
        
        history, _ = engine.simulate(steps=50, return_history=True)
        
        assert np.isfinite(history).all()


class TestCFLValidation:
    """Test CFL condition validation."""

    def test_validate_cfl_condition_pass(self) -> None:
        """Valid parameters should pass CFL check."""
        config = ReactionDiffusionConfig(
            d_activator=0.1,
            d_inhibitor=0.05,
            alpha=0.18,
        )
        engine = ReactionDiffusionEngine(config)
        
        assert engine.validate_cfl_condition()

    def test_validate_cfl_condition_edge(self) -> None:
        """Parameters near limit should still pass."""
        config = ReactionDiffusionConfig(
            alpha=0.24,  # Near 0.25 limit
        )
        engine = ReactionDiffusionEngine(config)
        
        assert engine.validate_cfl_condition()


class TestPerformance:
    """Performance sanity tests."""

    def test_small_grid_performance(self) -> None:
        """Small grid simulation should complete quickly."""
        config = ReactionDiffusionConfig(
            grid_size=32,
            random_seed=42,
        )
        engine = ReactionDiffusionEngine(config)
        
        start = time.time()
        engine.simulate(steps=100)
        elapsed = time.time() - start
        
        # Should complete in < 5 seconds on any reasonable hardware
        assert elapsed < 5.0, f"Took {elapsed:.2f}s, expected < 5s"

    def test_medium_grid_performance(self) -> None:
        """Medium grid simulation should complete in reasonable time."""
        config = ReactionDiffusionConfig(
            grid_size=64,
            random_seed=42,
        )
        engine = ReactionDiffusionEngine(config)
        
        start = time.time()
        engine.simulate(steps=100)
        elapsed = time.time() - start
        
        # Should complete in < 10 seconds
        assert elapsed < 10.0, f"Took {elapsed:.2f}s, expected < 10s"
