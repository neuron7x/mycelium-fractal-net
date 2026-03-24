"""Fractal Arsenal: multifractal spectrum + lacunarity + basin entropy.

Ref: Chhabra & Jensen (1989) Phys Rev Lett 62:1327
     Allain & Cloitre (1991) Phys Rev A 44:3552
     Daza et al. (2016) Sci Rep 6:31416
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "BasinFractalityResult",
    "FractalArsenalReport",
    "LacunarityProfile",
    "MultifractalSpectrum",
    "compute_basin_fractality",
    "compute_dlambda_dt",
    "compute_fractal_arsenal",
    "compute_lacunarity",
    "compute_multifractal_spectrum",
]

# FRACTAL ARSENAL


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
