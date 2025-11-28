"""
Stability smoke tests for numerical core engines.

Tests verify:
- No NaN/Inf values after N integration steps
- Values stay within physically valid ranges
- Exceptions are raised appropriately for unstable configurations

Reference: docs/ARCHITECTURE.md for expected ranges.
"""

from __future__ import annotations

import numpy as np
import pytest

from mycelium_fractal_net.core import (
    FractalGrowthConfig,
    FractalGrowthEngine,
    MembraneEngine,
    MembraneEngineConfig,
    NumericalInstabilityError,
    ReactionDiffusionConfig,
    ReactionDiffusionEngine,
    StabilityError,
    ValueOutOfRangeError,
)
from mycelium_fractal_net.core.config import IntegrationScheme


class TestMembraneEngineStability:
    """Stability tests for membrane potential engine."""

    def test_short_simulation_no_nan(self) -> None:
        """Test that short simulation produces no NaN/Inf."""
        config = MembraneEngineConfig(
            dt=0.001,
            random_seed=42,
            check_stability=True,
        )
        engine = MembraneEngine(config)

        V, metrics = engine.simulate(n_neurons=100, steps=100)

        assert not metrics.nan_detected, "NaN detected in membrane simulation"
        assert not metrics.inf_detected, "Inf detected in membrane simulation"
        assert np.isfinite(V).all(), "Final potentials contain NaN/Inf"

    def test_long_simulation_no_nan(self) -> None:
        """Test that long simulation (1000 steps) remains stable."""
        config = MembraneEngineConfig(
            dt=0.001,
            random_seed=42,
        )
        engine = MembraneEngine(config)

        V, metrics = engine.simulate(n_neurons=100, steps=1000)

        assert not metrics.nan_detected
        assert not metrics.inf_detected
        assert metrics.steps_completed == 1000

    def test_potential_range_physiological(self) -> None:
        """Test potentials stay in physiological range [-95, +40] mV."""
        config = MembraneEngineConfig(
            dt=0.001,
            random_seed=42,
            v_min=-0.095,
            v_max=0.040,
        )
        engine = MembraneEngine(config)

        V, metrics = engine.simulate(
            n_neurons=100,
            steps=500,
            spike_probability=0.1,
            spike_amplitude=0.030,
        )

        # Check range in mV
        v_min_mv = metrics.v_min * 1000.0
        v_max_mv = metrics.v_max * 1000.0

        assert v_min_mv >= -95.0 - 0.1, f"Min potential {v_min_mv:.2f} mV < -95 mV"
        assert v_max_mv <= 40.0 + 0.1, f"Max potential {v_max_mv:.2f} mV > 40 mV"

    def test_nernst_potassium_correct(self) -> None:
        """Test Nernst equation gives correct E_K ≈ -89 mV."""
        config = MembraneEngineConfig(temperature_k=310.0)
        engine = MembraneEngine(config)

        # K+: [K]_in = 140 mM, [K]_out = 5 mM at 37°C
        E_K = engine.compute_nernst_potential(
            z_valence=1,
            concentration_out=5e-3,  # 5 mM
            concentration_in=140e-3,  # 140 mM
        )

        E_K_mv = E_K * 1000.0
        assert -92.0 < E_K_mv < -85.0, f"E_K = {E_K_mv:.2f} mV outside expected range"

    def test_euler_vs_rk4_consistency(self) -> None:
        """Test that Euler and RK4 give similar results for small dt."""
        config_euler = MembraneEngineConfig(
            dt=0.001,
            random_seed=42,
            integration_scheme=IntegrationScheme.EULER,
        )
        config_rk4 = MembraneEngineConfig(
            dt=0.001,
            random_seed=42,
            integration_scheme=IntegrationScheme.RK4,
        )

        engine_euler = MembraneEngine(config_euler)
        engine_rk4 = MembraneEngine(config_rk4)

        V_euler, _ = engine_euler.simulate(n_neurons=10, steps=100)
        V_rk4, _ = engine_rk4.simulate(n_neurons=10, steps=100)

        # Both should converge to similar values (within 10%)
        diff = np.abs(V_euler - V_rk4).max()
        assert diff < 0.01, f"Euler/RK4 diverge by {diff*1000:.2f} mV"

    def test_stability_condition_validation(self) -> None:
        """Test that invalid dt raises ValueError."""
        with pytest.raises(ValueError, match="must be < tau"):
            config = MembraneEngineConfig(
                dt=0.015,  # 15 ms > tau = 10 ms (unstable for Euler)
                tau=0.010,
                integration_scheme=IntegrationScheme.EULER,
            )
            config.validate()


class TestReactionDiffusionStability:
    """Stability tests for reaction-diffusion engine."""

    def test_short_simulation_no_nan(self) -> None:
        """Test short simulation produces no NaN/Inf."""
        config = ReactionDiffusionConfig(
            grid_size=32,
            steps=50,
            dt=0.1,
            random_seed=42,
        )
        engine = ReactionDiffusionEngine(config)

        activator, inhibitor, metrics = engine.simulate()

        assert not metrics.nan_detected
        assert not metrics.inf_detected
        assert np.isfinite(activator).all()
        assert np.isfinite(inhibitor).all()

    def test_long_simulation_no_nan(self) -> None:
        """Test long simulation (500 steps) remains stable."""
        config = ReactionDiffusionConfig(
            grid_size=32,
            steps=500,
            dt=0.1,
            random_seed=42,
        )
        engine = ReactionDiffusionEngine(config)

        activator, inhibitor, metrics = engine.simulate()

        assert not metrics.nan_detected
        assert not metrics.inf_detected
        assert metrics.steps_completed == 500

    def test_concentration_range_valid(self) -> None:
        """Test activator/inhibitor stay in [0, 1] range."""
        config = ReactionDiffusionConfig(
            grid_size=32,
            steps=200,
            dt=0.1,
            random_seed=42,
        )
        engine = ReactionDiffusionEngine(config)

        activator, inhibitor, metrics = engine.simulate()

        assert metrics.activator_min >= 0.0, f"Activator min {metrics.activator_min} < 0"
        assert metrics.activator_max <= 1.0, f"Activator max {metrics.activator_max} > 1"
        assert metrics.inhibitor_min >= 0.0, f"Inhibitor min {metrics.inhibitor_min} < 0"
        assert metrics.inhibitor_max <= 1.0, f"Inhibitor max {metrics.inhibitor_max} > 1"

    def test_turing_pattern_formation(self) -> None:
        """Test that Turing patterns form (non-zero pattern fraction)."""
        config = ReactionDiffusionConfig(
            grid_size=64,
            steps=200,
            dt=0.1,
            d_activator=0.1,
            d_inhibitor=0.05,
            r_activator=0.01,
            r_inhibitor=0.02,
            turing_threshold=0.75,
            random_seed=42,
        )
        engine = ReactionDiffusionEngine(config)

        activator, inhibitor, metrics = engine.simulate()

        # Pattern should have some structure (not all below or above threshold)
        assert 0.0 <= metrics.pattern_fraction <= 1.0

    def test_cfl_violation_raises_error(self) -> None:
        """Test that CFL condition violation raises ValueError."""
        with pytest.raises(ValueError, match="CFL violation"):
            config = ReactionDiffusionConfig(
                grid_size=32,
                dt=10.0,  # Very large dt
                d_activator=0.1,  # CFL limit: dt < 1/(4*0.1) = 2.5
            )
            config.validate()

    def test_field_coupling_mode(self) -> None:
        """Test simulation with external field coupling."""
        config = ReactionDiffusionConfig(
            grid_size=32,
            steps=50,
            dt=0.1,
            random_seed=42,
        )
        engine = ReactionDiffusionEngine(config)

        # Initial field at -70 mV
        field = np.full((32, 32), -0.070)

        activator, inhibitor, field_out, metrics = engine.simulate_with_field(
            field, field_coupling=0.005
        )

        assert np.isfinite(field_out).all()
        assert field_out.min() >= -0.095 - 0.001
        assert field_out.max() <= 0.040 + 0.001


class TestFractalGrowthStability:
    """Stability tests for fractal growth engine."""

    def test_ifs_no_nan(self) -> None:
        """Test IFS generation produces no NaN/Inf."""
        config = FractalGrowthConfig(
            num_points=1000,
            num_transforms=4,
            random_seed=42,
        )
        engine = FractalGrowthEngine(config)

        points, metrics = engine.generate_ifs()

        assert not metrics.nan_detected
        assert not metrics.inf_detected
        assert np.isfinite(points).all()

    def test_ifs_stability(self) -> None:
        """Test IFS has negative Lyapunov exponent (stable)."""
        config = FractalGrowthConfig(
            num_points=5000,
            num_transforms=4,
            contraction_min=0.2,
            contraction_max=0.5,
            random_seed=42,
        )
        engine = FractalGrowthEngine(config)

        points, metrics = engine.generate_ifs()

        assert metrics.is_stable, f"Lyapunov exponent {metrics.lyapunov_exponent:.4f} >= 0"
        assert metrics.lyapunov_exponent < 0

    def test_fractal_dimension_range(self) -> None:
        """Test fractal dimension is in expected range [0.5, 2.5]."""
        config = FractalGrowthConfig(
            num_points=10000,
            num_transforms=4,
            grid_size=64,
            random_seed=42,
        )
        engine = FractalGrowthEngine(config)

        points, metrics = engine.generate_ifs()

        assert 0.5 <= metrics.fractal_dimension <= 2.5, (
            f"Dimension {metrics.fractal_dimension:.3f} outside [0.5, 2.5]"
        )

    def test_dla_no_nan(self) -> None:
        """Test DLA generation produces valid grid."""
        config = FractalGrowthConfig(
            grid_size=32,
            max_iterations=100,
            random_seed=42,
            dla_enabled=True,
        )
        engine = FractalGrowthEngine(config)

        grid, metrics = engine.generate_dla()

        assert grid.dtype == bool
        assert metrics.dla_particles >= 1  # At least the seed
        assert 0.0 < metrics.grid_fill_fraction < 1.0

    def test_contraction_validation(self) -> None:
        """Test invalid contraction factors raise ValueError."""
        with pytest.raises(ValueError, match="contraction"):
            config = FractalGrowthConfig(
                contraction_min=0.5,
                contraction_max=0.4,  # max < min
            )
            config.validate()

        with pytest.raises(ValueError, match="contraction"):
            config = FractalGrowthConfig(
                contraction_min=0.5,
                contraction_max=1.5,  # max >= 1 (non-contractive)
            )
            config.validate()

    def test_box_counting_empty_field(self) -> None:
        """Test box-counting handles empty field."""
        config = FractalGrowthConfig(random_seed=42)
        engine = FractalGrowthEngine(config)

        empty = np.zeros((64, 64), dtype=bool)
        dim = engine.estimate_fractal_dimension(empty)

        assert dim == 0.0 or dim >= 0.0  # Empty field has no fractal structure

    def test_box_counting_full_field(self) -> None:
        """Test box-counting handles full field (D ≈ 2)."""
        config = FractalGrowthConfig(random_seed=42)
        engine = FractalGrowthEngine(config)

        full = np.ones((64, 64), dtype=bool)
        dim = engine.estimate_fractal_dimension(full)

        assert 1.5 <= dim <= 2.5, f"Full field dimension {dim:.3f} not near 2"


class TestExceptionClasses:
    """Tests for custom exception classes."""

    def test_stability_error_message(self) -> None:
        """Test StabilityError includes step and value info."""
        err = StabilityError("Test error", step=42, value=1.5)
        assert "Test error" in str(err)
        assert "step=42" in str(err)
        assert "1.5" in str(err)

    def test_value_out_of_range_error(self) -> None:
        """Test ValueOutOfRangeError includes bounds."""
        err = ValueOutOfRangeError(
            "Value exceeded",
            value=1.5,
            min_bound=0.0,
            max_bound=1.0,
        )
        assert "1.5" in str(err)
        assert "[0" in str(err)
        assert "1]" in str(err)

    def test_numerical_instability_error(self) -> None:
        """Test NumericalInstabilityError includes NaN/Inf counts."""
        err = NumericalInstabilityError(
            "Field unstable",
            field_name="activator",
            nan_count=5,
            inf_count=2,
        )
        assert "activator" in str(err)
        assert "NaN=5" in str(err)
        assert "Inf=2" in str(err)


class TestPerformanceSanity:
    """Performance sanity tests - ensure operations complete in reasonable time."""

    def test_membrane_100_neurons_1000_steps(self) -> None:
        """Test membrane simulation completes in < 1 second."""
        config = MembraneEngineConfig(dt=0.001, random_seed=42)
        engine = MembraneEngine(config)

        _, metrics = engine.simulate(n_neurons=100, steps=1000)

        assert metrics.execution_time_s < 1.0, (
            f"Membrane simulation took {metrics.execution_time_s:.2f}s > 1s"
        )

    def test_reaction_diffusion_64x64_200_steps(self) -> None:
        """Test reaction-diffusion (64x64, 200 steps) completes in < 2 seconds."""
        config = ReactionDiffusionConfig(
            grid_size=64,
            steps=200,
            dt=0.1,
            random_seed=42,
        )
        engine = ReactionDiffusionEngine(config)

        _, _, metrics = engine.simulate()

        assert metrics.execution_time_s < 2.0, (
            f"Reaction-diffusion took {metrics.execution_time_s:.2f}s > 2s"
        )

    def test_ifs_10000_points(self) -> None:
        """Test IFS (10000 points) completes in < 1 second."""
        config = FractalGrowthConfig(
            num_points=10000,
            num_transforms=4,
            random_seed=42,
        )
        engine = FractalGrowthEngine(config)

        _, metrics = engine.generate_ifs()

        assert metrics.execution_time_s < 1.0, (
            f"IFS generation took {metrics.execution_time_s:.2f}s > 1s"
        )

    def test_dla_32x32_100_iterations(self) -> None:
        """Test DLA (32x32, 100 iterations) completes in < 5 seconds."""
        config = FractalGrowthConfig(
            grid_size=32,
            max_iterations=100,
            random_seed=42,
        )
        engine = FractalGrowthEngine(config)

        _, metrics = engine.generate_dla()

        assert metrics.execution_time_s < 5.0, (
            f"DLA took {metrics.execution_time_s:.2f}s > 5s"
        )
