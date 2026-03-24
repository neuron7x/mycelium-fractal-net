"""
Fractal Feature Extraction for SimulationResult.

Provides integration between the core simulation results and fractal analytics
while implementing the API specified in ``MFN_FEATURE_SCHEMA.md``.

Usage:
    >>> from mycelium_fractal_net import run_mycelium_simulation, SimulationConfig
    >>> from mycelium_fractal_net.analytics.fractal_features import compute_fractal_features
    >>> result = run_mycelium_simulation(SimulationConfig(steps=100))
    >>> features = compute_fractal_features(result)
    >>> print(features.values)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

# Import the core feature computation from the canonical package implementation
from .legacy_features import FeatureConfig
from .legacy_features import FeatureVector as AnalyticsFeatureVector
from .legacy_features import compute_basic_stats as _compute_basic_stats
from .legacy_features import compute_features as _compute_features
from .legacy_features import compute_fractal_features as _compute_fractal_dim

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mycelium_fractal_net.core.types import SimulationResult

__all__ = [
    "BasinFractalityResult",
    "BasinInvariantResult",
    "DFAResult",
    "FeatureVector",
    "FractalArsenalReport",
    "FractalDynamicsReport",
    "LacunarityProfile",
    "MultifractalSpectrum",
    "SpectralEvolution",
    "compute_basic_stats",
    "compute_basin_fractality",
    "compute_basin_invariant",
    "compute_box_counting_dimension",
    "compute_dfa",
    "compute_dlambda_dt",
    "compute_fractal_arsenal",
    "compute_fractal_features",
    "compute_lacunarity",
    "compute_multifractal_spectrum",
    "compute_spectral_evolution",
]


@dataclass
class FeatureVector:
    """
    Structured feature vector from fractal analysis.

    Contains all features defined in MFN_FEATURE_SCHEMA.md.
    Values are stored as a dictionary for easy access and serialization.

    Attributes
    ----------
    values : Dict[str, float]
        Dictionary mapping feature names to their values.
        All 18 features from MFN_FEATURE_SCHEMA.md are included.

    Examples
    --------
    >>> fv = FeatureVector(values={"D_box": 1.65, "V_mean": -70.0, ...})
    >>> fv.values["D_box"]
    1.65
    >>> fv.to_array()
    array([1.65, 0.95, -80.0, ...])
    """

    values: dict[str, float] = field(default_factory=dict)

    # Feature names in canonical order (from MFN_FEATURE_SCHEMA.md)
    _FEATURE_NAMES: tuple[str, ...] = (
        "D_box",
        "D_r2",
        "V_min",
        "V_max",
        "V_mean",
        "V_std",
        "V_skew",
        "V_kurt",
        "dV_mean",
        "dV_max",
        "T_stable",
        "E_trend",
        "f_active",
        "N_clusters_low",
        "N_clusters_med",
        "N_clusters_high",
        "max_cluster_size",
        "cluster_size_std",
    )

    @classmethod
    def feature_names(cls) -> tuple[str, ...]:
        """Return canonical feature names in order."""
        return cls._FEATURE_NAMES

    def to_array(self) -> NDArray[np.float64]:
        """
        Convert to numpy array in canonical order.

        Returns
        -------
        NDArray[np.float64]
            Array of shape (18,) with features in order defined by MFN_FEATURE_SCHEMA.md.
        """
        return np.array(
            [float(self.values.get(name, 0.0)) for name in self._FEATURE_NAMES],
            dtype=np.float64,
        )

    @classmethod
    def from_analytics_vector(cls, av: AnalyticsFeatureVector) -> FeatureVector:
        """
        Create FeatureVector from analytics module FeatureVector.

        Parameters
        ----------
        av : AnalyticsFeatureVector
            FeatureVector from the analytics module.

        Returns
        -------
        FeatureVector
            New FeatureVector with values from the analytics vector.
        """
        return cls(values=av.to_dict())

    def __contains__(self, key: str) -> bool:
        """Check if feature exists."""
        return key in self.values

    def __getitem__(self, key: str) -> float:
        """Get feature value by name."""
        return self.values[key]


def compute_box_counting_dimension(
    field: NDArray[np.floating[Any]],
    *,
    num_scales: int = 8,
    threshold: float = -0.060,
) -> float:
    """
    Compute box-counting fractal dimension for a 2D field.

    Uses the box-counting algorithm to estimate the fractal dimension D
    of the active region in the field. The active region is defined as
    cells with values above the threshold.

    Algorithm:
    1. Binarize field at threshold
    2. Count occupied boxes at multiple scales (box sizes)
    3. Fit log(N) vs log(1/ε) to estimate D

    For a fractal set: N(ε) ~ ε^(-D)
    where N is the number of occupied boxes and ε is the box size.

    Parameters
    ----------
    field : NDArray[np.floating]
        2D field array. Values should be in Volts (biological range: ~-0.095 to 0.040).
        Must be a square array.
    num_scales : int, optional
        Number of scales (box sizes) to use for regression. Default is 8.
        More scales provide better estimates but require larger fields.
    threshold : float, optional
        Threshold for binarization in Volts. Default is -0.060 V (-60 mV).
        Cells above this threshold are considered "active".

    Returns
    -------
    float
        Estimated fractal dimension D.
        - D ≈ 0 for empty/sparse fields
        - D ≈ 1 for line-like structures
        - D ∈ [1.4, 1.9] for biological mycelium patterns
        - D ≈ 2 for filled regions

    Raises
    ------
    ValueError
        If field is not 2D or not square.
        If field contains NaN or infinite values.

    Notes
    -----
    Implementation details are described in docs/MFN_FEATURE_SCHEMA.md.

    Examples
    --------
    >>> import numpy as np
    >>> field = np.random.randn(64, 64) * 0.01 - 0.070
    >>> D = compute_box_counting_dimension(field)
    >>> print(f"Fractal dimension: {D:.3f}")
    """
    if not np.isfinite(field).all():
        raise ValueError("field contains NaN or Inf values")

    if field.ndim != 2:
        raise ValueError(f"field must be 2D, got {field.ndim}D")
    if field.shape[0] != field.shape[1]:
        raise ValueError(f"field must be square, got shape {field.shape}")

    # Convert threshold from Volts (user API) to mV (internal FeatureConfig expects mV)
    threshold_mv = threshold * 1000.0
    config = FeatureConfig(num_scales=num_scales, threshold_low_mv=threshold_mv)

    D_box, _ = _compute_fractal_dim(field, config)
    return D_box


def compute_basic_stats(field: NDArray[np.floating[Any]]) -> dict[str, float]:
    """
    Compute basic statistics for a 2D field.

    Calculates min, max, mean, and std of the field values.
    All outputs are in millivolts (mV) for interpretability.

    Parameters
    ----------
    field : NDArray[np.floating]
        2D field array. Values should be in Volts.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        - "min": Minimum value in mV
        - "max": Maximum value in mV
        - "mean": Mean value in mV
        - "std": Standard deviation in mV

    Examples
    --------
    >>> import numpy as np
    >>> field = np.full((32, 32), -0.070)  # -70 mV constant field
    >>> stats = compute_basic_stats(field)
    >>> print(stats["mean"])  # Should be close to -70.0
    """
    V_min, V_max, V_mean, V_std, _, _ = _compute_basic_stats(field)
    return {
        "min": V_min,
        "max": V_max,
        "mean": V_mean,
        "std": V_std,
    }


def compute_fractal_features(result: SimulationResult) -> FeatureVector:
    """
    Compute complete feature vector from SimulationResult.

    Extracts all 18 features defined in MFN_FEATURE_SCHEMA.md from
    the simulation result. Uses the final field state for static features
    and the full history (if available) for temporal features.

    This function does not modify the input result.

    Args:
        result: SimulationResult containing at minimum the final field.
            If history is available (result.has_history is True), temporal
            features will be computed from the history array. Otherwise,
            temporal features will be set to default values (0.0).

    Returns:
        FeatureVector with all 18 features in values dict:
            - Fractal: D_box (box-counting dimension), D_r2 (regression fit)
            - Basic stats: V_min, V_max, V_mean, V_std, V_skew, V_kurt (mV)
            - Temporal: dV_mean, dV_max, T_stable, E_trend
            - Structural: f_active, N_clusters_low/med/high, max_cluster_size, cluster_size_std

    Raises:
        TypeError: If result is not a SimulationResult instance.
        ValueError: If result.field has invalid shape (must be 2D square).

    Examples:
        >>> from mycelium_fractal_net import run_mycelium_simulation, SimulationConfig
        >>> from mycelium_fractal_net import compute_fractal_features
        >>> result = run_mycelium_simulation(SimulationConfig(steps=50, seed=42))
        >>> features = compute_fractal_features(result)
        >>> print(f"D_box: {features.values['D_box']:.3f}")
        >>> print(f"V_mean: {features.values['V_mean']:.1f} mV")

    Notes:
        - All voltage values are converted to mV in the output.
        - NaN/Inf values are not expected (controlled by numerical core).
        - For valid results, D_box should be in [0, 2.5], biological range [1.4, 1.9].
        - Does not modify the input result object.
        - For temporal features, use run_mycelium_simulation_with_history.

    See Also:
        MFN_FEATURE_SCHEMA.md: Complete feature specification.
        compute_box_counting_dimension: Box-counting algorithm details.
    """
    # Import here to avoid circular imports
    from mycelium_fractal_net.core.types import SimulationResult as SR

    # Validate input type
    if not isinstance(result, SR):
        raise TypeError(
            f"Expected SimulationResult, got {type(result).__name__}. "
            "Use analytics.compute_features() for raw numpy arrays."
        )

    # Determine input data shape
    if result.has_history and result.history is not None:
        # Use full history for temporal features
        field_data = result.history
    else:
        # Single snapshot - temporal features will be defaults
        field_data = result.field

    # Compute features using the analytics module
    analytics_fv = _compute_features(field_data)

    # Convert to our FeatureVector format
    return FeatureVector.from_analytics_vector(analytics_fv)


# ═══════════════════════════════════════════════════════════════════════════════
# FRACTAL ARSENAL
# Three mechanisms: multifractal spectrum + lacunarity + basin entropy
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MultifractalSpectrum:
    """Chhabra-Jensen multifractal singularity spectrum f(alpha).

    Ref: Chhabra & Jensen (1989) Phys Rev Lett 62:1327
         Takahara & Sato (2025) Mathematics 13:3234

    Nine features: delta_alpha, alpha_0, f_max, asymmetry, D0, D1, D2,
    D0_minus_D2, AUS. Genuine multifractality requires delta_alpha > 0.2.
    """

    alpha_q: np.ndarray
    f_q: np.ndarray
    q_values: np.ndarray
    r_squared: np.ndarray

    @property
    def delta_alpha(self) -> float:
        """Spectrum width = degree of multifractality."""
        valid = self.r_squared >= 0.9
        if valid.sum() < 3:
            return 0.0
        return float(self.alpha_q[valid].max() - self.alpha_q[valid].min())

    @property
    def alpha_0(self) -> float:
        """Dominant Holder exponent (at peak of f(alpha))."""
        return float(self.alpha_q[np.argmax(self.f_q)])

    @property
    def f_max(self) -> float:
        """Peak of singularity spectrum."""
        return float(self.f_q.max())

    @property
    def asymmetry(self) -> float:
        """Asymmetry: (alpha_max - alpha_0) / (alpha_0 - alpha_min)."""
        valid = self.r_squared >= 0.9
        if valid.sum() < 3:
            return 1.0
        a_valid = self.alpha_q[valid]
        a_max, a_min = a_valid.max(), a_valid.min()
        denom = self.alpha_0 - a_min
        return float((a_max - self.alpha_0) / denom) if abs(denom) > 1e-10 else 1.0

    def _dq(self, q: float) -> float:
        """Generalized dimension at specific q."""
        idx = int(np.argmin(np.abs(self.q_values - q)))
        if self.r_squared[idx] < 0.9:
            return float("nan")
        if abs(q - 1) < 0.1:
            return float(self.f_q[idx])
        return float(self.f_q[idx] / (q - 1))

    @property
    def D0(self) -> float:
        """Box-counting dimension."""
        return self._dq(0.0)

    @property
    def D1(self) -> float:
        """Information dimension."""
        return self.alpha_0

    @property
    def D2(self) -> float:
        """Correlation dimension."""
        return self._dq(2.0)

    @property
    def D0_minus_D2(self) -> float:
        """Multifractality index D0 - D2."""
        if np.isnan(self.D0) or np.isnan(self.D2):
            return 0.0
        return float(self.D0 - self.D2)

    @property
    def AUS(self) -> float:
        """Area under D_q spectrum."""
        valid = self.r_squared >= 0.9
        if valid.sum() < 2:
            return 0.0
        return float(np.trapezoid(self.f_q[valid], self.alpha_q[valid]))

    @property
    def is_genuine(self) -> bool:
        """True if multifractality is genuine (not finite-size artifact)."""
        return self.delta_alpha > 0.2

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "delta_alpha": round(self.delta_alpha, 4),
            "alpha_0": round(self.alpha_0, 4),
            "f_max": round(self.f_max, 4),
            "asymmetry": round(self.asymmetry, 4),
            "D0": round(self.D0, 4) if not np.isnan(self.D0) else None,
            "D1": round(self.D1, 4),
            "D2": round(self.D2, 4) if not np.isnan(self.D2) else None,
            "D0_minus_D2": round(self.D0_minus_D2, 4),
            "AUS": round(self.AUS, 4),
            "is_genuine": self.is_genuine,
            "n_valid_q": int(np.sum(self.r_squared >= 0.9)),
        }


def compute_multifractal_spectrum(
    field_input: np.ndarray,
    q_values: np.ndarray | None = None,
    min_box: int = 2,
) -> MultifractalSpectrum:
    """Compute Chhabra-Jensen multifractal singularity spectrum for 2D field.

    Uses dyadic box-counting with q-weighted measures and batch regression.
    """
    if q_values is None:
        q_values = np.linspace(-3, 5, 17)

    f = np.asarray(field_input, dtype=np.float64)
    f = f - f.min() + 1e-12
    N = f.shape[0]

    sizes = [
        min_box * 2**k
        for k in range(int(np.log2(max(N // min_box, 1))) + 1)
        if min_box * 2**k <= N // 2
    ]

    if len(sizes) < 3:
        return MultifractalSpectrum(
            alpha_q=np.ones(len(q_values)) * 2.0,
            f_q=np.ones(len(q_values)) * 2.0,
            q_values=q_values,
            r_squared=np.zeros(len(q_values)),
        )

    n_q = len(q_values)
    n_s = len(sizes)

    # Pre-compute box sums for all scales once (avoid repeated reduceat)
    box_sums: list[np.ndarray] = []
    for eps in sizes:
        sums = np.add.reduceat(
            np.add.reduceat(f, np.arange(0, N, eps), axis=0),
            np.arange(0, N, eps),
            axis=1,
        )
        box_sums.append(sums)

    # Build Ma and Mf matrices: (n_q, n_s) — all q-values × all scales
    Ma = np.zeros((n_q, n_s))
    Mf = np.zeros((n_q, n_s))

    for si, sums in enumerate(box_sums):
        p_arr = (sums / sums.sum()).ravel()
        p_arr = p_arr[p_arr > 1e-20]
        log_p = np.log(p_arr)

        for qi, q in enumerate(q_values):
            if abs(q) < 1e-6:
                mu = np.ones_like(p_arr) / len(p_arr)
            else:
                pq = p_arr**q
                mu = pq / pq.sum()
            Ma[qi, si] = float(np.sum(mu * log_p))
            Mf[qi, si] = float(np.sum(mu * np.log(mu + 1e-30)))

    # Batch linear regression: vectorized slope + R² for all q at once
    log_eps = np.log(np.array(sizes, dtype=float))
    x_mean = log_eps.mean()
    x_var = np.sum((log_eps - x_mean) ** 2)

    def _batch_slopes(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute slope and R² for each row of Y vs log_eps."""
        y_mean = Y.mean(axis=1, keepdims=True)
        slopes = ((Y - y_mean) * (log_eps - x_mean)).sum(axis=1) / x_var
        y_pred = y_mean.ravel()[:, None] + slopes[:, None] * (log_eps - x_mean)
        ss_res = np.sum((Y - y_pred) ** 2, axis=1)
        ss_tot = np.sum((Y - y_mean) ** 2, axis=1)
        r2_arr = np.where(ss_tot > 1e-30, 1.0 - ss_res / ss_tot, 0.0)
        return slopes, r2_arr

    alpha_q, r2_a = _batch_slopes(Ma)
    f_q, r2_f = _batch_slopes(Mf)
    r2 = np.minimum(r2_a, r2_f)

    return MultifractalSpectrum(alpha_q=alpha_q, f_q=f_q, q_values=q_values, r_squared=r2)


@dataclass
class LacunarityProfile:
    """Gliding-box lacunarity profile Lambda(r).

    Ref: Allain & Cloitre (1991) Phys Rev A 44:3552
    Lambda = 1: homogeneous. Lambda >> 1: large gaps.
    """

    box_sizes: np.ndarray
    lambda_r: np.ndarray
    lambda_prefactor: float
    decay_exponent: float

    @property
    def lambda_at_4(self) -> float:
        """Lambda at r=4."""
        idx = np.where(self.box_sizes == 4)[0]
        return float(self.lambda_r[idx[0]]) if len(idx) > 0 else float("nan")

    @property
    def lambda_at_8(self) -> float:
        """Lambda at r=8."""
        idx = np.where(self.box_sizes == 8)[0]
        return float(self.lambda_r[idx[0]]) if len(idx) > 0 else float("nan")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        valid = ~np.isnan(self.lambda_r)
        return {
            "lambda_at_4": round(self.lambda_at_4, 4) if not np.isnan(self.lambda_at_4) else None,
            "lambda_at_8": round(self.lambda_at_8, 4) if not np.isnan(self.lambda_at_8) else None,
            "lambda_prefactor": round(self.lambda_prefactor, 4),
            "decay_exponent": round(self.decay_exponent, 4),
            "lambda_mean": round(float(np.nanmean(self.lambda_r)), 4),
            "n_valid_scales": int(valid.sum()),
        }


def compute_lacunarity(
    field_input: np.ndarray,
    box_sizes: list[int] | None = None,
) -> LacunarityProfile:
    """Compute gliding-box lacunarity profile. < 3ms for N=32."""
    from scipy.signal import fftconvolve as _fftconvolve
    from scipy.stats import linregress as _linregress

    f = np.asarray(field_input, dtype=np.float64)
    f = f - f.min() + 1e-12
    N = f.shape[0]

    if box_sizes is None:
        box_sizes = [2**k for k in range(1, int(np.log2(max(N, 2))))]

    lambda_r = np.full(len(box_sizes), np.nan)
    for i, r in enumerate(box_sizes):
        if r >= N:
            continue
        kernel = np.ones((r, r), dtype=np.float64)
        box_mass = _fftconvolve(f, kernel, mode="valid")
        mu = float(np.mean(box_mass))
        if mu > 1e-12:
            lambda_r[i] = float(np.var(box_mass) / mu**2) + 1.0

    valid = ~np.isnan(lambda_r)
    if valid.sum() >= 2:
        log_r = np.log(np.array(box_sizes, dtype=float)[valid])
        log_l = np.log(np.maximum(lambda_r[valid] - 1.0 + 1e-12, 1e-12))
        reg = _linregress(log_r, log_l)
        prefactor = float(np.exp(reg.intercept))
        exponent = float(reg.slope)
    else:
        prefactor, exponent = 1.0, 0.0

    return LacunarityProfile(
        box_sizes=np.array(box_sizes),
        lambda_r=lambda_r,
        lambda_prefactor=prefactor,
        decay_exponent=exponent,
    )


@dataclass
class BasinFractalityResult:
    """Basin entropy and fractality test (Daza et al. 2016).

    Ref: Daza et al. (2016) Sci Rep 6:31416
    S_bb > ln(2) is a SUFFICIENT condition for fractal basin boundaries.
    """

    S_bb: float
    is_fractal: bool
    n_mixed_boxes: int
    n_total_boxes: int
    mixed_fraction: float
    box_size: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "S_bb": round(self.S_bb, 4),
            "is_fractal": self.is_fractal,
            "n_mixed_boxes": self.n_mixed_boxes,
            "n_total_boxes": self.n_total_boxes,
            "mixed_fraction": round(self.mixed_fraction, 4),
            "box_size": self.box_size,
            "ln2_threshold": round(float(np.log(2)), 4),
        }


def compute_basin_fractality(
    basin_grid: np.ndarray,
    box_size: int = 4,
) -> BasinFractalityResult:
    """Compute basin entropy to test for fractal basin boundaries. < 2ms for N=32."""
    grid = np.asarray(basin_grid)
    N = grid.shape[0]
    entropies: list[float] = []
    n_mixed = 0

    for i in range(0, N - box_size + 1, box_size):
        for j in range(0, N - box_size + 1, box_size):
            patch = grid[i : i + box_size, j : j + box_size].ravel()
            unique, counts = np.unique(patch, return_counts=True)
            if len(unique) > 1:
                n_mixed += 1
                p = counts / counts.sum()
                entropies.append(float(-np.sum(p * np.log(p))))

    n_total = (N // box_size) ** 2
    s_bb = float(np.mean(entropies)) if entropies else 0.0

    return BasinFractalityResult(
        S_bb=s_bb,
        is_fractal=s_bb > float(np.log(2)),
        n_mixed_boxes=n_mixed,
        n_total_boxes=n_total,
        mixed_fraction=float(n_mixed / n_total) if n_total > 0 else 0.0,
        box_size=box_size,
    )


def compute_dlambda_dt(
    history: np.ndarray,
    r: int = 4,
    stride: int = 1,
) -> np.ndarray:
    """Lacunarity rate of change dLambda/dt — novel EWS for morphological transitions."""
    from scipy.signal import fftconvolve as _fftconvolve

    T = history.shape[0]
    frames = list(range(0, T, stride))
    lambda_t = np.zeros(len(frames))

    for i, t in enumerate(frames):
        f = history[t].astype(np.float64)
        f = f - f.min() + 1e-12
        kernel = np.ones((r, r), dtype=np.float64)
        bm = _fftconvolve(f, kernel, mode="valid")
        mu = float(np.mean(bm))
        lambda_t[i] = float(np.var(bm) / mu**2) + 1.0 if mu > 1e-12 else 1.0

    return np.diff(lambda_t)


@dataclass
class FractalArsenalReport:
    """Unified report from all fractal arsenal computations."""

    multifractal: MultifractalSpectrum
    lacunarity: LacunarityProfile
    basin_fractality: BasinFractalityResult | None = None

    def summary(self) -> str:
        """Single-line summary."""
        mf = self.multifractal
        lac = self.lacunarity
        genuine = "GENUINE" if mf.is_genuine else "spurious"
        basin = ""
        if self.basin_fractality:
            fb = self.basin_fractality
            basin = f" | S_bb={fb.S_bb:.3f} {'FRACTAL' if fb.is_fractal else 'smooth'}"
        return (
            f"[FRACTAL] da={mf.delta_alpha:.3f}({genuine}) "
            f"a0={mf.alpha_0:.3f} D0-D2={mf.D0_minus_D2:.3f} "
            f"asym={mf.asymmetry:.2f} | "
            f"L(4)={lac.lambda_at_4:.3f} decay={lac.decay_exponent:.3f}"
            f"{basin}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        d: dict[str, Any] = {
            "multifractal": self.multifractal.to_dict(),
            "lacunarity": self.lacunarity.to_dict(),
        }
        if self.basin_fractality:
            d["basin_fractality"] = self.basin_fractality.to_dict()
        return d


def compute_fractal_arsenal(
    field_input: np.ndarray,
    basin_grid: np.ndarray | None = None,
) -> FractalArsenalReport:
    """Compute all fractal arsenal metrics for one field snapshot. < 35ms for N=32."""
    mf = compute_multifractal_spectrum(field_input)
    lac = compute_lacunarity(field_input)
    bf = compute_basin_fractality(basin_grid) if basin_grid is not None else None
    return FractalArsenalReport(multifractal=mf, lacunarity=lac, basin_fractality=bf)


# ═══════════════════════════════════════════════════════════════════════════════
# FRACTAL DYNAMICS V2
# Three vectors: spectral evolution + DFA Hurst + S_bb×S_B invariant
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SpectralEvolution:
    """Temporal evolution of multifractal spectrum width delta_alpha(t).

    Tracks how system complexity changes over time:
      d(delta_alpha)/dt > 0: expansion — system building complexity
      delta_alpha → 0: collapse — monofractalization → critical transition

    Ref: Kantelhardt et al. (2002) Physica A 316:87-114
    """

    delta_alpha_t: np.ndarray
    d_delta_alpha_dt: np.ndarray
    timestamps: np.ndarray
    is_collapsing: bool
    collapse_onset: int | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "delta_alpha_final": round(float(self.delta_alpha_t[-1]), 4),
            "delta_alpha_max": round(float(self.delta_alpha_t.max()), 4),
            "d_da_dt_mean": round(float(np.mean(self.d_delta_alpha_dt)), 6),
            "d_da_dt_max": round(float(np.max(np.abs(self.d_delta_alpha_dt))), 6),
            "is_collapsing": self.is_collapsing,
            "collapse_onset": self.collapse_onset,
            "n_frames": len(self.delta_alpha_t),
        }


def compute_spectral_evolution(
    history: np.ndarray,
    stride: int = 1,
    q_values: np.ndarray | None = None,
) -> SpectralEvolution:
    """Track delta_alpha(t) across field history.

    Computes multifractal spectrum width at each frame, then derives
    d(delta_alpha)/dt to detect complexity expansion or collapse.

    Args:
        history: (T, N, N) field history
        stride: compute every `stride` frames (default 1)
        q_values: q-values for multifractal computation

    Returns:
        SpectralEvolution with delta_alpha trajectory and collapse detection.
    """
    T = history.shape[0]
    frames = list(range(0, T, stride))
    da_t = np.zeros(len(frames))

    for i, t in enumerate(frames):
        spec = compute_multifractal_spectrum(history[t], q_values=q_values)
        da_t[i] = spec.delta_alpha

    d_da = np.diff(da_t)
    timestamps = np.array(frames, dtype=float)

    # Collapse detection: delta_alpha decreasing over last 30% of trajectory
    tail_start = max(1, int(len(da_t) * 0.7))
    tail = da_t[tail_start:]
    is_collapsing = bool(len(tail) >= 2 and tail[-1] < tail[0] * 0.7)

    # Find onset: first frame where d_da becomes persistently negative
    collapse_onset: int | None = None
    if is_collapsing and len(d_da) >= 3:
        # Rolling window of 3: if all negative, mark onset
        for k in range(len(d_da) - 2):
            if d_da[k] < 0 and d_da[k + 1] < 0 and d_da[k + 2] < 0:
                collapse_onset = int(frames[k])
                break

    return SpectralEvolution(
        delta_alpha_t=da_t,
        d_delta_alpha_dt=d_da,
        timestamps=timestamps,
        is_collapsing=is_collapsing,
        collapse_onset=collapse_onset,
    )


@dataclass
class DFAResult:
    """Detrended Fluctuation Analysis result.

    Ref: Peng et al. (1994) Phys Rev E 49:1685
         Kantelhardt et al. (2002) Physica A 316:87-114

    H < 0.5: anti-persistent (mean-reverting)
    H = 0.5: uncorrelated (random walk)
    H > 0.5: persistent (trending, has memory)
    H → 1.0: critical slowing down — maximum persuadability window
    """

    hurst_exponent: float
    r_squared: float
    fluctuations: np.ndarray
    scales: np.ndarray
    is_persistent: bool
    is_critical: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "hurst_exponent": round(self.hurst_exponent, 4),
            "r_squared": round(self.r_squared, 4),
            "is_persistent": self.is_persistent,
            "is_critical": self.is_critical,
            "n_scales": len(self.scales),
        }


def compute_dfa(
    time_series: np.ndarray,
    min_scale: int = 4,
    max_scale: int | None = None,
    n_scales: int = 10,
) -> DFAResult:
    """Detrended Fluctuation Analysis for Hurst exponent estimation.

    Computes the scaling exponent H of a time series via:
    1. Integrate (cumulative sum of detrended signal)
    2. Divide into windows of size s
    3. Fit linear trend in each window, compute RMS of residuals F(s)
    4. H = slope of log(F) vs log(s)

    H → 1.0 signals critical slowing down = maximum persuadability.

    Args:
        time_series: 1D array (e.g., field mean over time)
        min_scale: smallest window size
        max_scale: largest window (default T//4)
        n_scales: number of log-spaced scales

    Returns:
        DFAResult with Hurst exponent and diagnostics.
    """
    from scipy.stats import linregress as _linregress

    x = np.asarray(time_series, dtype=np.float64)
    T = len(x)
    if T < 16:
        return DFAResult(
            hurst_exponent=0.5,
            r_squared=0.0,
            fluctuations=np.array([]),
            scales=np.array([]),
            is_persistent=False,
            is_critical=False,
        )

    # Integrate: cumulative sum of centered signal
    y = np.cumsum(x - np.mean(x))

    if max_scale is None:
        max_scale = T // 4
    max_scale = max(max_scale, min_scale + 1)

    scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales).astype(int))
    scales = scales[scales >= min_scale]
    scales = scales[scales <= max_scale]

    if len(scales) < 3:
        return DFAResult(
            hurst_exponent=0.5,
            r_squared=0.0,
            fluctuations=np.array([]),
            scales=np.array([]),
            is_persistent=False,
            is_critical=False,
        )

    fluctuations = np.zeros(len(scales))

    for i, s in enumerate(scales):
        n_windows = T // s
        if n_windows < 1:
            fluctuations[i] = np.nan
            continue
        rms_list: list[float] = []
        for w in range(n_windows):
            segment = y[w * s : (w + 1) * s]
            # Linear detrend
            t_axis = np.arange(s, dtype=np.float64)
            coeffs = np.polyfit(t_axis, segment, 1)
            trend = np.polyval(coeffs, t_axis)
            rms_list.append(float(np.sqrt(np.mean((segment - trend) ** 2))))
        fluctuations[i] = float(np.mean(rms_list)) if rms_list else np.nan

    valid = ~np.isnan(fluctuations) & (fluctuations > 0)
    if valid.sum() < 3:
        return DFAResult(
            hurst_exponent=0.5,
            r_squared=0.0,
            fluctuations=fluctuations,
            scales=scales,
            is_persistent=False,
            is_critical=False,
        )

    log_s = np.log(scales[valid].astype(float))
    log_f = np.log(fluctuations[valid])
    reg = _linregress(log_s, log_f)
    H = float(reg.slope)
    r2 = float(reg.rvalue**2)

    return DFAResult(
        hurst_exponent=H,
        r_squared=r2,
        fluctuations=fluctuations,
        scales=scales,
        is_persistent=H > 0.55,
        is_critical=H > 0.85,
    )


@dataclass
class BasinInvariantResult:
    """S_bb x S_B anti-correlation diagnostic.

    Novel MFN invariant: basin entropy (boundary fractality) and basin
    stability (return probability) are anti-correlated near transitions.

    chi = S_bb * S_B should be approximately constant in stable regimes.
    Deviation signals topological reorganization.

    No prior publication connects Menck (2013) basin stability with
    Daza (2016) basin entropy in a single diagnostic.
    """

    S_bb: float
    S_B: float
    chi: float
    chi_interpretation: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "S_bb": round(self.S_bb, 4),
            "S_B": round(self.S_B, 4),
            "chi": round(self.chi, 4),
            "interpretation": self.chi_interpretation,
        }


def compute_basin_invariant(
    S_bb: float,
    S_B: float,
) -> BasinInvariantResult:
    """Compute the S_bb x S_B anti-correlation invariant.

    Args:
        S_bb: basin entropy from compute_basin_fractality()
        S_B: basin stability from BasinStabilityAnalyzer

    Returns:
        BasinInvariantResult with chi diagnostic.

    Interpretation:
        S_B high + S_bb low → stable, smooth boundaries (healthy)
        S_B low + S_bb high → unstable, fractal boundaries (critical)
        Both low → single-basin dominated (trivial)
        Both high → paradoxical (check data quality)
    """
    chi = S_bb * S_B

    if S_B > 0.7 and S_bb < 0.5:
        interp = "STABLE: robust basin with smooth boundaries"
    elif S_B < 0.4 and S_bb > float(np.log(2)):
        interp = "CRITICAL: fractal boundaries + low stability — intervention window"
    elif S_B < 0.4 and S_bb < 0.3:
        interp = "COLLAPSING: single basin absorbing — loss of multistability"
    elif S_B > 0.7 and S_bb > float(np.log(2)):
        interp = "PARADOX: high stability + fractal boundaries — verify data"
    else:
        interp = "TRANSITIONAL: system between regimes"

    return BasinInvariantResult(
        S_bb=S_bb,
        S_B=S_B,
        chi=chi,
        chi_interpretation=interp,
    )


@dataclass
class FractalDynamicsReport:
    """Unified report from Fractal Dynamics V2 computations."""

    spectral_evolution: SpectralEvolution
    dfa: DFAResult
    basin_invariant: BasinInvariantResult | None = None

    def summary(self) -> str:
        """Single-line summary."""
        se = self.spectral_evolution
        dfa = self.dfa
        collapse = "COLLAPSING" if se.is_collapsing else "expanding"
        critical = (
            "CRITICAL"
            if dfa.is_critical
            else ("persistent" if dfa.is_persistent else "uncorrelated")
        )
        basin = ""
        if self.basin_invariant:
            bi = self.basin_invariant
            basin = f" | chi={bi.chi:.3f} {bi.chi_interpretation.split(':')[0]}"
        return (
            f"[DYNAMICS] da(T)={se.delta_alpha_t[-1]:.3f}({collapse}) "
            f"H={dfa.hurst_exponent:.3f}({critical})"
            f"{basin}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        d: dict[str, Any] = {
            "spectral_evolution": self.spectral_evolution.to_dict(),
            "dfa": self.dfa.to_dict(),
        }
        if self.basin_invariant:
            d["basin_invariant"] = self.basin_invariant.to_dict()
        return d
