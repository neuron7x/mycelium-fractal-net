"""
Test public API structure for MyceliumFractalNet.

This module validates that:
1. All functions documented in README are importable
2. Functions have expected signatures
3. Smoke tests pass for basic invocations

Reference: README.md, docs/MFN_SYSTEM_ROLE.md
"""

from __future__ import annotations

import inspect

import numpy as np


class TestPublicAPIImports:
    """Test that all documented public API components are importable."""

    def test_main_package_imports(self) -> None:
        """Test that main package level imports work."""
        # Functions documented in README
        from mycelium_fractal_net import (
            compute_lyapunov_exponent,
            compute_nernst_potential,
            estimate_fractal_dimension,
            generate_fractal_ifs,
            run_validation,
            simulate_mycelium_field,
        )

        assert callable(compute_nernst_potential)
        assert callable(simulate_mycelium_field)
        assert callable(estimate_fractal_dimension)
        assert callable(generate_fractal_ifs)
        assert callable(compute_lyapunov_exponent)
        assert callable(run_validation)

    def test_constants_importable(self) -> None:
        """Test that physical constants are importable."""
        from mycelium_fractal_net import (
            BODY_TEMPERATURE_K,
            FARADAY_CONSTANT,
            R_GAS_CONSTANT,
            SPARSE_TOPK,
            TURING_THRESHOLD,
        )

        # Validate constants have expected approximate values
        assert abs(R_GAS_CONSTANT - 8.314) < 0.001
        assert abs(FARADAY_CONSTANT - 96485.33) < 1.0
        assert abs(BODY_TEMPERATURE_K - 310.0) < 0.1
        assert abs(TURING_THRESHOLD - 0.75) < 0.01
        assert SPARSE_TOPK == 4

    def test_classes_importable(self) -> None:
        """Test that main classes are importable."""
        from mycelium_fractal_net import (
            HierarchicalKrumAggregator,
            MyceliumFractalNet,
            SparseAttention,
            STDPPlasticity,
            ValidationConfig,
        )

        assert STDPPlasticity is not None
        assert SparseAttention is not None
        assert HierarchicalKrumAggregator is not None
        assert MyceliumFractalNet is not None
        assert ValidationConfig is not None

    def test_core_engines_importable(self) -> None:
        """Test that core engines are importable."""
        from mycelium_fractal_net import (
            FractalGrowthEngine,
            MembraneEngine,
            ReactionDiffusionEngine,
        )

        assert MembraneEngine is not None
        assert ReactionDiffusionEngine is not None
        assert FractalGrowthEngine is not None

    def test_exceptions_importable(self) -> None:
        """Test that custom exceptions are importable."""
        from mycelium_fractal_net import (
            NumericalInstabilityError,
            StabilityError,
            ValueOutOfRangeError,
        )

        assert issubclass(StabilityError, Exception)
        assert issubclass(ValueOutOfRangeError, Exception)
        assert issubclass(NumericalInstabilityError, Exception)

    def test_simulation_types_importable(self) -> None:
        """Test that simulation types are importable."""
        from mycelium_fractal_net import (
            MyceliumField,
            SimulationConfig,
            SimulationResult,
        )

        assert SimulationConfig is not None
        assert SimulationResult is not None
        assert MyceliumField is not None

    def test_analytics_api_importable(self) -> None:
        """Test that analytics API is importable."""
        from mycelium_fractal_net import FeatureVector, compute_fractal_features

        assert FeatureVector is not None
        assert callable(compute_fractal_features)

    def test_aggregate_gradients_krum_importable(self) -> None:
        """Test that federated aggregation function is importable."""
        from mycelium_fractal_net import aggregate_gradients_krum

        assert callable(aggregate_gradients_krum)


class TestPublicAPISignatures:
    """Test that public API functions have expected signatures."""

    def test_compute_nernst_potential_signature(self) -> None:
        """Test compute_nernst_potential has expected parameters."""
        from mycelium_fractal_net import compute_nernst_potential

        sig = inspect.signature(compute_nernst_potential)
        params = list(sig.parameters.keys())

        assert "z_valence" in params
        assert "concentration_out_molar" in params
        assert "concentration_in_molar" in params
        assert "temperature_k" in params

    def test_simulate_mycelium_field_signature(self) -> None:
        """Test simulate_mycelium_field has expected parameters."""
        from mycelium_fractal_net import simulate_mycelium_field

        sig = inspect.signature(simulate_mycelium_field)
        params = list(sig.parameters.keys())

        assert "rng" in params
        assert "grid_size" in params
        assert "steps" in params
        assert "turing_enabled" in params

    def test_estimate_fractal_dimension_signature(self) -> None:
        """Test estimate_fractal_dimension has expected parameters."""
        from mycelium_fractal_net import estimate_fractal_dimension

        sig = inspect.signature(estimate_fractal_dimension)
        params = list(sig.parameters.keys())

        assert "binary_field" in params


class TestPublicAPISmoke:
    """Smoke tests for basic API invocations."""

    def test_compute_nernst_potential_smoke(self) -> None:
        """Test compute_nernst_potential returns expected value for K+."""
        from mycelium_fractal_net import compute_nernst_potential

        # K+ at standard concentrations
        e_k = compute_nernst_potential(
            z_valence=1,
            concentration_out_molar=5e-3,
            concentration_in_molar=140e-3,
            temperature_k=310.0,
        )

        # Should be approximately -89 mV
        e_k_mv = e_k * 1000
        assert -100 < e_k_mv < -80, f"Expected ~-89 mV, got {e_k_mv:.2f} mV"

    def test_simulate_mycelium_field_smoke(self) -> None:
        """Test simulate_mycelium_field produces valid output."""
        from mycelium_fractal_net import simulate_mycelium_field

        rng = np.random.default_rng(42)
        field, growth_events = simulate_mycelium_field(
            rng=rng,
            grid_size=16,
            steps=10,
            turing_enabled=True,
        )

        assert field.shape == (16, 16)
        assert isinstance(growth_events, int)
        assert growth_events >= 0
        # Field should be in physiological range
        assert field.min() >= -0.1  # -100 mV
        assert field.max() <= 0.05  # +50 mV

    def test_estimate_fractal_dimension_smoke(self) -> None:
        """Test estimate_fractal_dimension produces valid output."""
        from mycelium_fractal_net import estimate_fractal_dimension

        rng = np.random.default_rng(42)
        binary = rng.random((64, 64)) > 0.5
        dim = estimate_fractal_dimension(binary)

        # Dimension should be in valid range
        assert 0 <= dim <= 2

    def test_generate_fractal_ifs_smoke(self) -> None:
        """Test generate_fractal_ifs produces valid output."""
        from mycelium_fractal_net import generate_fractal_ifs

        rng = np.random.default_rng(42)
        points, lyapunov = generate_fractal_ifs(rng, num_points=100)

        assert points.shape == (100, 2)
        # Lyapunov should be negative (stable)
        assert lyapunov < 0


class TestCoreModuleImports:
    """Test that core domain modules are importable directly."""

    def test_nernst_module(self) -> None:
        """Test nernst module imports."""
        from mycelium_fractal_net.core.nernst import (
            BODY_TEMPERATURE_K,
            compute_nernst_potential,
        )

        assert callable(compute_nernst_potential)
        assert isinstance(BODY_TEMPERATURE_K, float)

    def test_turing_module(self) -> None:
        """Test turing module imports."""
        from mycelium_fractal_net.core.turing import (
            TURING_THRESHOLD,
            simulate_mycelium_field,
        )

        assert callable(simulate_mycelium_field)
        assert isinstance(TURING_THRESHOLD, float)

    def test_fractal_module(self) -> None:
        """Test fractal module imports."""
        from mycelium_fractal_net.core.fractal import (
            estimate_fractal_dimension,
            generate_fractal_ifs,
        )

        assert callable(estimate_fractal_dimension)
        assert callable(generate_fractal_ifs)

    def test_stdp_module(self) -> None:
        """Test stdp module imports."""
        from mycelium_fractal_net.core.stdp import (
            STDP_TAU_PLUS,
            STDPPlasticity,
        )

        assert STDPPlasticity is not None
        assert isinstance(STDP_TAU_PLUS, float)

    def test_federated_module(self) -> None:
        """Test federated module imports."""
        from mycelium_fractal_net.core.federated import (
            HierarchicalKrumAggregator,
            aggregate_gradients_krum,
        )

        assert HierarchicalKrumAggregator is not None
        assert callable(aggregate_gradients_krum)

    def test_stability_module(self) -> None:
        """Test stability module imports."""
        from mycelium_fractal_net.core.stability import (
            compute_lyapunov_exponent,
            is_stable,
        )

        assert callable(compute_lyapunov_exponent)
        assert callable(is_stable)
