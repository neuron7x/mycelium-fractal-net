"""
Tests for FractalGrowthEngine — IFS generation and box-counting.

Validates:
- IFS contraction requirement (MATH_MODEL.md Section 3.2)
- Lyapunov exponent negativity for stability
- Fractal dimension bounds [0, 2]
- No NaN/Inf in generated points
- Determinism with fixed seeds
"""

import time

import numpy as np
import pytest

from mycelium_fractal_net.core import (
    FractalConfig,
    FractalGrowthEngine,
    StabilityError,
    ValueOutOfRangeError,
)
from mycelium_fractal_net.core.fractal_growth_engine import (
    DEFAULT_NUM_POINTS,
    DEFAULT_NUM_TRANSFORMS,
    DEFAULT_SCALE_MAX,
    DEFAULT_SCALE_MIN,
    FRACTAL_DIM_MAX,
    FRACTAL_DIM_MIN,
    LYAPUNOV_STABLE_MAX,
)


class TestFractalConfig:
    """Test configuration validation."""

    def test_default_config_valid(self) -> None:
        """Default configuration should be valid."""
        config = FractalConfig()
        assert config.num_points == DEFAULT_NUM_POINTS
        assert config.num_transforms == DEFAULT_NUM_TRANSFORMS
        assert config.scale_min == DEFAULT_SCALE_MIN
        assert config.scale_max == DEFAULT_SCALE_MAX

    def test_scale_max_ge_1_raises(self) -> None:
        """Scale >= 1 should raise StabilityError (non-contractive)."""
        with pytest.raises(StabilityError, match="contractive"):
            FractalConfig(scale_max=1.0)
        
        with pytest.raises(StabilityError, match="contractive"):
            FractalConfig(scale_max=1.5)

    def test_negative_scale_raises(self) -> None:
        """Negative scale should raise."""
        with pytest.raises(ValueOutOfRangeError):
            FractalConfig(scale_min=-0.1)

    def test_scale_order_violation_raises(self) -> None:
        """scale_min > scale_max should raise."""
        with pytest.raises(ValueOutOfRangeError):
            FractalConfig(scale_min=0.6, scale_max=0.5)

    def test_low_num_points_raises(self) -> None:
        """Too few points should raise."""
        with pytest.raises(ValueOutOfRangeError, match="points"):
            FractalConfig(num_points=50)

    def test_low_num_scales_raises(self) -> None:
        """Too few scales should raise."""
        with pytest.raises(ValueOutOfRangeError, match="scales"):
            FractalConfig(num_scales=1)


class TestIFSGeneration:
    """Test IFS fractal generation."""

    def test_basic_generation(self) -> None:
        """Basic IFS generation should work."""
        config = FractalConfig(num_points=1000, random_seed=42)
        engine = FractalGrowthEngine(config)
        
        points, lyap = engine.generate_ifs()
        
        assert points.shape == (1000, 2)
        assert np.isfinite(points).all()
        assert np.isfinite(lyap)

    def test_lyapunov_negative(self) -> None:
        """Lyapunov exponent should be negative (contractive).
        
        Reference: MATH_MODEL.md Section 3.3 - λ < 0 indicates stable dynamics
        """
        config = FractalConfig(num_points=10000, random_seed=42)
        engine = FractalGrowthEngine(config)
        
        _, lyap = engine.generate_ifs()
        
        assert lyap < LYAPUNOV_STABLE_MAX, f"λ = {lyap:.3f} should be < 0"
        assert engine.metrics.is_contractive

    def test_lyapunov_range(self) -> None:
        """Lyapunov should be in expected range ~-2.1.
        
        Reference: MATH_MODEL.md Section 3.3 - Expected λ ≈ -2.1
        """
        lyap_values = []
        for seed in range(10):
            config = FractalConfig(num_points=5000, random_seed=seed)
            engine = FractalGrowthEngine(config)
            _, lyap = engine.generate_ifs()
            lyap_values.append(lyap)
        
        mean_lyap = np.mean(lyap_values)
        # Should be between -4 and -0.5 (reasonable contraction range)
        assert -4.0 < mean_lyap < -0.5, f"Mean λ = {mean_lyap:.3f}"

    def test_points_bounded(self) -> None:
        """Generated points should be bounded (finite attractor)."""
        config = FractalConfig(num_points=10000, random_seed=42)
        engine = FractalGrowthEngine(config)
        
        points, _ = engine.generate_ifs()
        
        max_dist = np.max(np.abs(points))
        assert max_dist < 100, f"Max distance {max_dist:.2f} too large"
        assert engine.metrics.points_bounded

    def test_contraction_validation(self) -> None:
        """Stored transforms should satisfy contraction requirement."""
        config = FractalConfig(num_points=1000, random_seed=42)
        engine = FractalGrowthEngine(config)
        
        engine.generate_ifs()
        
        assert engine.validate_contraction()
        assert engine.transforms is not None
        
        for a, b, c, d, e, f in engine.transforms:
            det = abs(a * d - b * c)
            assert det < 1.0, f"Determinant {det:.4f} >= 1 (non-contractive)"


class TestBoxCounting:
    """Test box-counting dimension estimation."""

    def test_basic_dimension(self) -> None:
        """Basic dimension estimation should work."""
        config = FractalConfig(random_seed=42)
        engine = FractalGrowthEngine(config)
        
        binary = np.random.default_rng(42).random((64, 64)) > 0.5
        dim = engine.estimate_dimension(binary)
        
        assert FRACTAL_DIM_MIN < dim < FRACTAL_DIM_MAX

    def test_full_field_dimension(self) -> None:
        """Full field should have dimension ~2."""
        config = FractalConfig()
        engine = FractalGrowthEngine(config)
        
        full = np.ones((64, 64), dtype=bool)
        dim = engine.estimate_dimension(full)
        
        assert 1.5 <= dim <= 2.5, f"Full field D = {dim:.3f}, expected ~2"

    def test_empty_field_dimension(self) -> None:
        """Empty field should have dimension 0."""
        config = FractalConfig()
        engine = FractalGrowthEngine(config)
        
        empty = np.zeros((64, 64), dtype=bool)
        dim = engine.estimate_dimension(empty)
        
        # Empty field has no occupied boxes → dimension 0
        assert dim == 0.0

    def test_line_dimension(self) -> None:
        """Horizontal line should have dimension ~1."""
        config = FractalConfig()
        engine = FractalGrowthEngine(config)
        
        line = np.zeros((64, 64), dtype=bool)
        line[32, :] = True
        dim = engine.estimate_dimension(line)
        
        assert 0.8 <= dim <= 1.2, f"Line D = {dim:.3f}, expected ~1"

    def test_dimension_r_squared(self) -> None:
        """R² metric should be recorded."""
        config = FractalConfig()
        engine = FractalGrowthEngine(config)
        
        binary = np.random.default_rng(42).random((64, 64)) > 0.5
        engine.estimate_dimension(binary)
        
        assert 0 <= engine.metrics.dimension_r_squared <= 1

    def test_non_square_raises(self) -> None:
        """Non-square field should raise ValueError."""
        config = FractalConfig()
        engine = FractalGrowthEngine(config)
        
        non_square = np.zeros((32, 64), dtype=bool)
        with pytest.raises(ValueError, match="square"):
            engine.estimate_dimension(non_square)

    def test_non_2d_raises(self) -> None:
        """Non-2D field should raise ValueError."""
        config = FractalConfig()
        engine = FractalGrowthEngine(config)
        
        field_3d = np.zeros((32, 32, 32), dtype=bool)
        with pytest.raises(ValueError, match="2D"):
            engine.estimate_dimension(field_3d)


class TestLyapunovFromHistory:
    """Test Lyapunov exponent from field history."""

    def test_stable_system(self) -> None:
        """Diffusive system should show bounded Lyapunov."""
        config = FractalConfig()
        engine = FractalGrowthEngine(config)
        
        # Create converging field history
        rng = np.random.default_rng(42)
        history = []
        field = rng.normal(0, 1, size=(32, 32))
        
        for _ in range(20):
            # Diffusion smooths the field (converging)
            field = field * 0.95  # Decay
            history.append(field.copy())
        
        history_arr = np.stack(history)
        lyap = engine.compute_lyapunov_from_history(history_arr)
        
        assert np.isfinite(lyap)

    def test_single_frame_returns_zero(self) -> None:
        """Single frame history should return 0."""
        config = FractalConfig()
        engine = FractalGrowthEngine(config)
        
        single = np.random.random((32, 32)).reshape(1, 32, 32)
        lyap = engine.compute_lyapunov_from_history(single)
        
        assert lyap == 0.0


class TestDimensionValidation:
    """Test dimension range validation."""

    def test_validate_dimension_general(self) -> None:
        """Validate general 2D dimension bounds."""
        config = FractalConfig()
        engine = FractalGrowthEngine(config)
        
        assert engine.validate_dimension_range(0.5, biological=False)
        assert engine.validate_dimension_range(1.5, biological=False)
        assert not engine.validate_dimension_range(-0.5, biological=False)
        assert not engine.validate_dimension_range(2.5, biological=False)

    def test_validate_dimension_biological(self) -> None:
        """Validate biological mycelium range [1.4, 1.9]."""
        config = FractalConfig()
        engine = FractalGrowthEngine(config)
        
        assert engine.validate_dimension_range(1.5, biological=True)
        assert engine.validate_dimension_range(1.7, biological=True)
        assert not engine.validate_dimension_range(1.0, biological=True)
        assert not engine.validate_dimension_range(2.0, biological=True)


class TestDeterminism:
    """Test reproducibility with fixed seeds."""

    def test_same_seed_same_points(self) -> None:
        """Same seed should produce identical points."""
        config1 = FractalConfig(num_points=1000, random_seed=42)
        config2 = FractalConfig(num_points=1000, random_seed=42)
        
        engine1 = FractalGrowthEngine(config1)
        engine2 = FractalGrowthEngine(config2)
        
        points1, lyap1 = engine1.generate_ifs()
        points2, lyap2 = engine2.generate_ifs()
        
        assert np.allclose(points1, points2)
        assert lyap1 == lyap2

    def test_same_seed_same_dimension(self) -> None:
        """Same seed should produce identical dimension."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        binary1 = rng1.random((64, 64)) > 0.5
        binary2 = rng2.random((64, 64)) > 0.5
        
        config = FractalConfig()
        engine = FractalGrowthEngine(config)
        
        dim1 = engine.estimate_dimension(binary1)
        engine.reset()
        dim2 = engine.estimate_dimension(binary2)
        
        assert dim1 == dim2


class TestStabilitySmoke:
    """Stability smoke tests."""

    def test_smoke_many_ifs_generations(self) -> None:
        """Generate many IFS fractals without errors."""
        for seed in range(50):
            config = FractalConfig(num_points=1000, random_seed=seed)
            engine = FractalGrowthEngine(config)
            
            points, lyap = engine.generate_ifs()
            
            assert np.isfinite(points).all(), f"NaN/Inf at seed={seed}"
            assert np.isfinite(lyap), f"NaN Lyapunov at seed={seed}"
            assert lyap < 0, f"Non-contractive at seed={seed}: λ={lyap:.3f}"

    def test_smoke_dimension_various_patterns(self) -> None:
        """Estimate dimension for various patterns."""
        config = FractalConfig()
        engine = FractalGrowthEngine(config)
        
        rng = np.random.default_rng(42)
        
        for threshold in [0.3, 0.5, 0.7, 0.9]:
            binary = rng.random((64, 64)) > threshold
            dim = engine.estimate_dimension(binary)
            
            assert 0 <= dim <= 2.5, f"Invalid dimension {dim:.3f} for threshold={threshold}"


class TestPerformance:
    """Performance sanity tests."""

    def test_ifs_performance(self) -> None:
        """IFS generation should complete quickly."""
        config = FractalConfig(num_points=10000, random_seed=42)
        engine = FractalGrowthEngine(config)
        
        start = time.time()
        engine.generate_ifs()
        elapsed = time.time() - start
        
        # Should complete in < 1 second
        assert elapsed < 1.0, f"Took {elapsed:.2f}s, expected < 1s"

    def test_dimension_performance(self) -> None:
        """Dimension estimation should complete quickly."""
        config = FractalConfig()
        engine = FractalGrowthEngine(config)
        
        binary = np.random.default_rng(42).random((128, 128)) > 0.5
        
        start = time.time()
        engine.estimate_dimension(binary)
        elapsed = time.time() - start
        
        # Should complete in < 1 second
        assert elapsed < 1.0, f"Took {elapsed:.2f}s, expected < 1s"
