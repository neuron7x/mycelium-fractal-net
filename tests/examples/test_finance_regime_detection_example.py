"""
Tests for finance_regime_detection.py example.

Verifies the finance use case: synthetic data → MFN features → regime classification.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add examples directory to path
examples_dir = Path(__file__).parent.parent.parent / "examples"
sys.path.insert(0, str(examples_dir))


class TestFinanceRegimeDetectionExample:
    """Tests for the finance regime detection example."""

    def test_run_finance_demo_returns_analysis(self) -> None:
        """Test that run_finance_demo returns valid RegimeAnalysis when requested."""
        from finance_regime_detection import run_finance_demo

        analysis = run_finance_demo(
            verbose=False,
            num_points=200,  # Smaller dataset for faster tests
            seed=42,
            return_analysis=True,
        )

        assert analysis is not None, "Should return RegimeAnalysis"
        assert hasattr(analysis, "regime"), "Should have regime attribute"
        assert hasattr(analysis, "fractal_dim"), "Should have fractal_dim attribute"
        assert hasattr(analysis, "lyapunov"), "Should have lyapunov attribute"

    def test_run_finance_demo_no_exceptions(self) -> None:
        """Test that run_finance_demo completes without exceptions."""
        from finance_regime_detection import run_finance_demo

        # Should not raise any exceptions with small dataset
        run_finance_demo(verbose=False, num_points=100, seed=42, return_analysis=False)

    def test_regime_classification_valid(self) -> None:
        """Test that detected regime is one of the valid options."""
        from finance_regime_detection import MarketRegime, run_finance_demo

        analysis = run_finance_demo(
            verbose=False,
            num_points=200,
            seed=42,
            return_analysis=True,
        )

        assert analysis is not None
        valid_regimes = {MarketRegime.HIGH_COMPLEXITY, MarketRegime.LOW_COMPLEXITY, MarketRegime.NORMAL}
        assert analysis.regime in valid_regimes, f"Invalid regime: {analysis.regime}"

    def test_analysis_features_valid_ranges(self) -> None:
        """Test that analysis features are in valid ranges."""
        from finance_regime_detection import run_finance_demo

        analysis = run_finance_demo(
            verbose=False,
            num_points=200,
            seed=42,
            return_analysis=True,
        )

        assert analysis is not None

        # Fractal dimension should be reasonable
        assert 0.0 <= analysis.fractal_dim <= 2.5, f"Invalid fractal_dim: {analysis.fractal_dim}"

        # Volatility should be non-negative
        assert analysis.volatility >= 0.0, f"Invalid volatility: {analysis.volatility}"

        # No NaN values
        assert not np.isnan(analysis.fractal_dim), "fractal_dim is NaN"
        assert not np.isnan(analysis.lyapunov), "lyapunov is NaN"
        assert not np.isnan(analysis.v_mean), "v_mean is NaN"
        assert not np.isnan(analysis.v_std), "v_std is NaN"

    def test_analysis_to_dict(self) -> None:
        """Test that analysis can be converted to dictionary."""
        from finance_regime_detection import run_finance_demo

        analysis = run_finance_demo(
            verbose=False,
            num_points=200,
            seed=42,
            return_analysis=True,
        )

        assert analysis is not None
        result_dict = analysis.to_dict()

        assert isinstance(result_dict, dict)
        assert "regime" in result_dict
        assert "fractal_dim" in result_dict
        assert "lyapunov" in result_dict
        assert "confidence" in result_dict


class TestFinanceDataGeneration:
    """Tests for synthetic market data generation."""

    def test_generate_synthetic_market_data(self) -> None:
        """Test synthetic market data generation."""
        from finance_regime_detection import generate_synthetic_market_data

        rng = np.random.default_rng(42)
        returns, labels = generate_synthetic_market_data(rng, num_points=300)

        assert len(returns) == 300
        assert len(labels) == 3  # Three regime segments
        assert not np.any(np.isnan(returns))
        assert not np.any(np.isinf(returns))

    def test_map_returns_to_field(self) -> None:
        """Test mapping returns to MFN field representation."""
        from finance_regime_detection import generate_synthetic_market_data, map_returns_to_field

        rng = np.random.default_rng(42)
        returns, _ = generate_synthetic_market_data(rng, num_points=300)
        field = map_returns_to_field(returns, grid_size=16)

        assert field.shape == (16, 16)
        assert not np.any(np.isnan(field))
        assert not np.any(np.isinf(field))

        # Field values should be in membrane potential range
        field_mv = field * 1000.0
        assert field_mv.min() >= -100.0, f"Field min {field_mv.min():.2f} too low"
        assert field_mv.max() <= 50.0, f"Field max {field_mv.max():.2f} too high"


class TestFinanceRegimeClassification:
    """Tests for regime classification logic."""

    def test_classify_regime_high_complexity(self) -> None:
        """Test classification of high complexity regime."""
        from finance_regime_detection import MarketRegime, classify_regime

        # High fractal dimension and positive Lyapunov
        regime, confidence = classify_regime(
            fractal_dim=1.8,
            v_std=10.0,
            lyapunov=0.5,
        )

        assert regime == MarketRegime.HIGH_COMPLEXITY

    def test_classify_regime_low_complexity(self) -> None:
        """Test classification of low complexity regime."""
        from finance_regime_detection import MarketRegime, classify_regime

        # Low fractal dimension, low volatility, very stable
        regime, confidence = classify_regime(
            fractal_dim=0.5,
            v_std=1.0,
            lyapunov=-3.0,
        )

        assert regime == MarketRegime.LOW_COMPLEXITY

    def test_classify_regime_normal(self) -> None:
        """Test classification of normal regime."""
        from finance_regime_detection import MarketRegime, classify_regime

        # Intermediate values
        regime, confidence = classify_regime(
            fractal_dim=1.3,
            v_std=5.0,
            lyapunov=-1.5,
        )

        assert regime == MarketRegime.NORMAL

    def test_main_function_exists(self) -> None:
        """Test that main() function exists and is callable."""
        from finance_regime_detection import main

        assert callable(main)
