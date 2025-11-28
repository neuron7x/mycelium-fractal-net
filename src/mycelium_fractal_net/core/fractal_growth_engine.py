"""
Fractal Growth Engine for IFS and DLA Models.

Implements stable stochastic fractal generation using:
- Iterated Function System (IFS) with contractive affine maps
- Diffusion-Limited Aggregation (DLA) for lattice growth
- Box-counting fractal dimension estimation

Mathematical Model (from docs/ARCHITECTURE.md Section 3):
    IFS: x_{n+1} = A_i * x_n + b_i  (random selection of affine map i)

    Lyapunov exponent:
    λ = lim_{n→∞} (1/n) Σ ln|det(J_i)|

    Box-counting dimension:
    D = lim_{ε→0} ln(N(ε)) / ln(1/ε)

Stability:
    - contraction_max < 1.0 ensures IFS convergence
    - Lyapunov exponent should be negative (λ < 0)

Reference:
    - Mandelbrot, B. (1982). The fractal geometry of nature.
    - Barnsley, M. (1988). Fractals Everywhere.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
from numpy.typing import NDArray

from mycelium_fractal_net.core.config import FractalGrowthConfig
from mycelium_fractal_net.core.exceptions import (
    NumericalInstabilityError,
    StabilityError,
)


@dataclass
class FractalGrowthMetrics:
    """
    Metrics collected during fractal growth simulation.

    Attributes:
        points_generated: Number of IFS points generated.
        lyapunov_exponent: Estimated Lyapunov exponent (should be < 0).
        fractal_dimension: Estimated box-counting dimension.
        dla_particles: Number of DLA particles aggregated.
        grid_fill_fraction: Fraction of grid cells occupied (DLA).
        nan_detected: Whether NaN was detected.
        inf_detected: Whether Inf was detected.
        is_stable: Whether dynamics are stable (λ < 0).
        execution_time_s: Execution time in seconds.
    """

    points_generated: int = 0
    lyapunov_exponent: float = 0.0
    fractal_dimension: float = 0.0
    dla_particles: int = 0
    grid_fill_fraction: float = 0.0
    nan_detected: bool = False
    inf_detected: bool = False
    is_stable: bool = True
    execution_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "points_generated": self.points_generated,
            "lyapunov_exponent": self.lyapunov_exponent,
            "fractal_dimension": self.fractal_dimension,
            "dla_particles": self.dla_particles,
            "grid_fill_fraction": self.grid_fill_fraction,
            "nan_detected": self.nan_detected,
            "inf_detected": self.inf_detected,
            "is_stable": self.is_stable,
            "execution_time_s": self.execution_time_s,
        }


class FractalGrowthEngine:
    """
    Numerical engine for fractal growth using IFS and DLA.

    Implements stable fractal generation with:
    - Contractive IFS ensuring convergence
    - Lyapunov exponent monitoring for stability
    - Box-counting dimension estimation
    - Optional DLA lattice growth

    Example:
        >>> config = FractalGrowthConfig(num_points=10000, random_seed=42)
        >>> engine = FractalGrowthEngine(config)
        >>> points, metrics = engine.generate_ifs()
        >>> print(f"Lyapunov: {metrics.lyapunov_exponent:.3f}")
        >>> print(f"Dimension: {metrics.fractal_dimension:.3f}")

    Reference: docs/ARCHITECTURE.md Section 3
    """

    def __init__(self, config: FractalGrowthConfig | None = None) -> None:
        """
        Initialize fractal growth engine.

        Args:
            config: Configuration parameters. Uses defaults if None.

        Raises:
            ValueError: If configuration is invalid.
        """
        self.config = config or FractalGrowthConfig()
        self.config.validate()
        self._rng = np.random.default_rng(self.config.random_seed)
        self._metrics = FractalGrowthMetrics()
        self._transforms: List[Tuple[float, ...]] = []

    def _generate_transforms(self) -> None:
        """
        Generate random contractive affine transformations.

        Each transform has the form:
            x' = a*x + b*y + e
            y' = c*x + d*y + f

        With rotation/scaling matrix:
            [a, b]   [s*cos(θ), -s*sin(θ)]
            [c, d] = [s*sin(θ),  s*cos(θ)]

        Contraction factor s ∈ [contraction_min, contraction_max].
        """
        self._transforms = []
        for _ in range(self.config.num_transforms):
            # Random contraction and rotation
            scale = self._rng.uniform(
                self.config.contraction_min, self.config.contraction_max
            )
            angle = self._rng.uniform(0, 2 * np.pi)

            a = scale * np.cos(angle)
            b = -scale * np.sin(angle)
            c = scale * np.sin(angle)
            d = scale * np.cos(angle)
            e = self._rng.uniform(-1, 1)
            f = self._rng.uniform(-1, 1)

            self._transforms.append((a, b, c, d, e, f))

    def _check_point_stability(
        self, x: float, y: float, step: int
    ) -> None:
        """
        Check if point is finite.

        Raises:
            NumericalInstabilityError: If point is NaN/Inf.
        """
        if not (np.isfinite(x) and np.isfinite(y)):
            self._metrics.nan_detected = not np.isfinite(x) or not np.isfinite(y)
            self._metrics.inf_detected = np.isinf(x) or np.isinf(y)
            if self.config.check_stability:
                raise NumericalInstabilityError(
                    "IFS point became NaN/Inf",
                    field_name="point",
                    step=step,
                )

    def generate_ifs(self) -> Tuple[NDArray[Any], FractalGrowthMetrics]:
        """
        Generate fractal pattern using Iterated Function System.

        Returns:
            points: Generated points of shape (num_points, 2).
            metrics: Generation metrics including Lyapunov exponent.

        Raises:
            NumericalInstabilityError: If generation becomes unstable.
            StabilityError: If Lyapunov exponent is positive (unstable).
        """
        start_time = time.perf_counter()

        # Generate transforms
        self._generate_transforms()

        num_points = self.config.num_points
        points = np.zeros((num_points, 2))

        x, y = 0.0, 0.0
        log_jacobian_sum = 0.0

        for i in range(num_points):
            # Random transform selection
            idx = self._rng.integers(0, self.config.num_transforms)
            a, b, c, d, e, f = self._transforms[idx]

            # Apply affine transformation
            x_new = a * x + b * y + e
            y_new = c * x + d * y + f
            x, y = x_new, y_new

            # Check stability
            if self.config.check_stability:
                self._check_point_stability(x, y, i)

            points[i] = [x, y]

            # Accumulate Jacobian determinant for Lyapunov exponent
            det = abs(a * d - b * c)
            if det > 1e-10:
                log_jacobian_sum += np.log(det)

        # Compute Lyapunov exponent
        lyapunov = log_jacobian_sum / num_points
        is_stable = lyapunov < 0

        self._metrics.points_generated = num_points
        self._metrics.lyapunov_exponent = lyapunov
        self._metrics.is_stable = is_stable

        # Check stability
        if self.config.check_stability and not is_stable:
            raise StabilityError(
                f"IFS dynamics unstable: Lyapunov exponent = {lyapunov:.4f} >= 0",
                value=lyapunov,
            )

        # Estimate fractal dimension
        dimension = self._estimate_dimension_from_points(points)
        self._metrics.fractal_dimension = dimension

        self._metrics.execution_time_s = time.perf_counter() - start_time

        return points, self._metrics

    def _estimate_dimension_from_points(
        self, points: NDArray[Any]
    ) -> float:
        """
        Estimate fractal dimension from point cloud using box-counting.

        Converts points to binary grid, then applies box-counting.
        """
        # Normalize points to [0, 1] range
        pts = points.copy()
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        ranges = maxs - mins
        ranges[ranges < 1e-10] = 1.0  # Prevent division by zero
        pts = (pts - mins) / ranges

        # Create binary grid
        grid_size = self.config.grid_size
        binary_grid = np.zeros((grid_size, grid_size), dtype=bool)

        # Map points to grid cells
        indices = (pts * (grid_size - 1)).astype(int)
        indices = np.clip(indices, 0, grid_size - 1)
        binary_grid[indices[:, 0], indices[:, 1]] = True

        return self.estimate_fractal_dimension(binary_grid)

    def estimate_fractal_dimension(
        self,
        binary_field: NDArray[Any],
        min_box_size: int | None = None,
        max_box_size: int | None = None,
        num_scales: int | None = None,
    ) -> float:
        """
        Estimate fractal dimension using box-counting method.

        D = lim_{ε→0} ln(N(ε)) / ln(1/ε)

        Fitted via linear regression on log-log plot.

        Args:
            binary_field: Boolean 2D array.
            min_box_size: Minimum box size.
            max_box_size: Maximum box size.
            num_scales: Number of logarithmic scales.

        Returns:
            Estimated fractal dimension.

        Reference: docs/ARCHITECTURE.md Section 3
        """
        if binary_field.ndim != 2:
            raise ValueError("binary_field must be 2D")

        n = binary_field.shape[0]

        min_size = min_box_size or self.config.box_min_size
        num_sc = num_scales or self.config.box_num_scales

        if max_box_size is None:
            max_size = min_size * (2 ** (num_sc - 1))
            max_size = min(max_size, n // 2 if n >= 4 else n)
        else:
            max_size = max_box_size

        if max_size < min_size:
            max_size = min_size

        sizes = np.geomspace(min_size, max_size, num_sc).astype(int)
        sizes = np.unique(sizes)
        counts = []

        for size in sizes:
            if size <= 0:
                continue
            n_boxes = n // size
            if n_boxes == 0:
                continue

            # Reshape and count occupied boxes
            cropped = binary_field[: n_boxes * size, : n_boxes * size]
            reshaped = cropped.reshape(n_boxes, size, n_boxes, size)
            occupied = reshaped.any(axis=(1, 3))
            counts.append(occupied.sum())

        counts_arr = np.array(counts, dtype=float)
        valid = counts_arr > 0
        if valid.sum() < 2:
            return 0.0

        sizes = sizes[valid]
        counts_arr = counts_arr[valid]

        # Linear regression on log-log scale
        inv_eps = 1.0 / sizes.astype(float)
        log_inv_eps = np.log(inv_eps)
        log_counts = np.log(counts_arr)

        coeffs = np.polyfit(log_inv_eps, log_counts, 1)
        return float(coeffs[0])

    def generate_dla(
        self,
        seed_position: Tuple[int, int] | None = None,
    ) -> Tuple[NDArray[Any], FractalGrowthMetrics]:
        """
        Generate fractal pattern using Diffusion-Limited Aggregation.

        Particles perform random walk until they stick to existing cluster.

        Args:
            seed_position: Initial seed position. Default: center of grid.

        Returns:
            grid: Boolean grid with aggregated pattern.
            metrics: Generation metrics.
        """
        start_time = time.perf_counter()

        N = self.config.grid_size
        grid = np.zeros((N, N), dtype=bool)

        # Place seed
        if seed_position is None:
            cx, cy = N // 2, N // 2
        else:
            cx, cy = seed_position
        grid[cx, cy] = True
        particle_count = 1

        # DLA iterations
        for _ in range(self.config.max_iterations):
            # Start particle at random edge
            edge = self._rng.integers(0, 4)
            if edge == 0:  # Top
                x, y = self._rng.integers(0, N), 0
            elif edge == 1:  # Bottom
                x, y = self._rng.integers(0, N), N - 1
            elif edge == 2:  # Left
                x, y = 0, self._rng.integers(0, N)
            else:  # Right
                x, y = N - 1, self._rng.integers(0, N)

            # Random walk until stuck or escaped
            stuck = False
            for _ in range(N * N):  # Max walk steps
                # Check if adjacent to cluster
                neighbors = [
                    (x - 1, y), (x + 1, y),
                    (x, y - 1), (x, y + 1),
                ]
                for nx, ny in neighbors:
                    if 0 <= nx < N and 0 <= ny < N and grid[nx, ny]:
                        grid[x, y] = True
                        particle_count += 1
                        stuck = True
                        break

                if stuck:
                    break

                # Random walk step
                direction = self._rng.integers(0, 4)
                if direction == 0:
                    x = max(0, x - 1)
                elif direction == 1:
                    x = min(N - 1, x + 1)
                elif direction == 2:
                    y = max(0, y - 1)
                else:
                    y = min(N - 1, y + 1)

                # Escape check
                if x == 0 or x == N - 1 or y == 0 or y == N - 1:
                    if self._rng.random() < 0.1:  # 10% escape probability
                        break

        # Compute metrics
        self._metrics.dla_particles = particle_count
        self._metrics.grid_fill_fraction = float(grid.sum()) / (N * N)
        self._metrics.fractal_dimension = self.estimate_fractal_dimension(grid)
        self._metrics.execution_time_s = time.perf_counter() - start_time

        return grid, self._metrics

    def compute_lyapunov_from_history(
        self,
        field_history: NDArray[Any],
        dt: float = 1.0,
    ) -> float:
        """
        Compute Lyapunov exponent from field evolution history.

        λ = (1/T) Σ ln(|δx(t+1)| / |δx(t)|)

        Args:
            field_history: Array of shape (T, N, N).
            dt: Time step between states.

        Returns:
            Estimated Lyapunov exponent.
        """
        if len(field_history) < 2:
            return 0.0

        T = len(field_history)
        log_divergence = 0.0
        count = 0

        for t in range(1, T):
            diff = np.abs(field_history[t] - field_history[t - 1])
            norm_diff = np.sqrt(np.sum(diff ** 2))
            if norm_diff > 1e-10:
                log_divergence += np.log(norm_diff)
                count += 1

        if count == 0:
            return 0.0

        return log_divergence / (count * dt)

    @property
    def metrics(self) -> FractalGrowthMetrics:
        """Get current metrics."""
        return self._metrics

    @property
    def transforms(self) -> List[Tuple[float, ...]]:
        """Get current IFS transforms."""
        return self._transforms.copy()

    def reset(self) -> None:
        """Reset engine state."""
        self._rng = np.random.default_rng(self.config.random_seed)
        self._metrics = FractalGrowthMetrics()
        self._transforms = []
