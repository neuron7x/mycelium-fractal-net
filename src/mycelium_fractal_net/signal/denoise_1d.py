"""Fractal-inspired 1D denoiser."""

from __future__ import annotations

import logging
from typing import Callable, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _canonicalize_1d(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], torch.dtype]:
    """Convert inputs to [B, C, L] and return a reshape callback."""
    original_dtype = tensor.dtype

    if tensor.dim() == 1:

        def reshaper(out: torch.Tensor) -> torch.Tensor:
            return out.squeeze(0).squeeze(0)

        return tensor.unsqueeze(0).unsqueeze(0), reshaper, original_dtype

    if tensor.dim() == 2:

        def reshaper(out: torch.Tensor) -> torch.Tensor:
            return out.squeeze(1)

        return tensor.unsqueeze(1), reshaper, original_dtype

    if tensor.dim() == 3:

        def reshaper(out: torch.Tensor) -> torch.Tensor:
            return out

        return tensor, reshaper, original_dtype

    raise ValueError("Expected input shape [L], [B, L], or [B, C, L]")


def _normalized_kernel(window: int) -> torch.Tensor:
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd number >= 3")
    kernel = torch.ones(1, 1, window, dtype=torch.float64)
    return kernel / float(window)


class OptimizedFractalDenoise1D(nn.Module):
    """Multi-scale denoiser that preserves structure while suppressing spikes."""

    def __init__(
        self,
        *,
        base_window: int = 5,
        trend_scaling: float = 0.6,
        detail_preservation: float = 0.85,
        spike_threshold: float = 3.5,
        spike_damping: float = 0.35,
        iterations: int = 2,
        mode: Literal["multiscale", "fractal"] = "multiscale",
        range_size: int = 8,
        domain_scale: int = 4,
        population_size: int = 128,
        iterations_fractal: int = 1,
        s_threshold: float = 1e-3,
        s_max: float = 1.0,
        overlap: bool = True,
        smooth_kernel: int = 5,
        do_no_harm: bool = True,
        harm_ratio: float = 0.95,
        ridge_lambda: float = 1e-4,
        max_population_ci: int = 256,
        fractal_dim_threshold: float = 1.85,
        log_mutual_information: bool = True,
        inhibition_epsilon: float | None = None,
        acceptor_iterations: int | None = None,
    ) -> None:
        super().__init__()
        self.trend_scaling = trend_scaling
        self.detail_preservation = detail_preservation
        self.spike_threshold = spike_threshold
        self.spike_damping = spike_damping
        self.iterations = iterations
        self.iterations_fractal = iterations_fractal
        self.mode = mode
        self.range_size = range_size
        self.domain_scale = domain_scale
        if population_size < 1:
            raise ValueError("population_size must be a positive integer")
        self.population_size = population_size
        if max_population_ci < 1:
            raise ValueError("max_population_ci must be a positive integer")
        self.max_population_ci = max_population_ci
        if smooth_kernel < 3 or smooth_kernel % 2 == 0:
            raise ValueError("smooth_kernel must be an odd integer >= 3")
        self.s_threshold = s_threshold
        self.s_max = s_max
        self.overlap = overlap
        self.smooth_kernel = smooth_kernel
        self.do_no_harm = do_no_harm
        self.harm_ratio = harm_ratio
        self.inhibition_epsilon = (
            inhibition_epsilon if inhibition_epsilon is not None else 1.0 - harm_ratio
        )
        self.ridge_lambda = ridge_lambda
        self.fractal_dim_threshold = fractal_dim_threshold
        self.log_mutual_information = log_mutual_information
        self.last_mutual_information: float | None = None
        self.stride = max(1, base_window // 2)
        self.r = base_window
        requested_acceptor_iters = (
            acceptor_iterations if acceptor_iterations is not None else iterations_fractal
        )
        # Minimum of 7 iterations matches CFDE fixed-point stabilization requirement
        self.acceptor_iterations = max(requested_acceptor_iters, 7)
        self._eps = torch.finfo(torch.float64).eps
        self.register_buffer("detail_kernel", _normalized_kernel(base_window))
        self.register_buffer("trend_kernel", _normalized_kernel(base_window * 2 + 1))
        self.register_buffer("fractal_kernel", _normalized_kernel(smooth_kernel))

    def _smooth(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        pad = kernel.shape[-1] // 2
        pad_mode = "reflect" if x.shape[-1] > pad else "replicate"
        padded = F.pad(x, (pad, pad), mode=pad_mode)
        channels = padded.shape[1]
        weight = kernel.expand(channels, -1, -1)
        return F.conv1d(padded, weight, groups=channels)

    def _denoise_signal(self, x: torch.Tensor, *, canonical: bool = False) -> torch.Tensor:
        """Denoise a canonical [B, C, L] signal with grouped smoothing and overlap-add."""
        reshape: Callable[[torch.Tensor], torch.Tensor]
        if canonical:
            canonical_signal = x
            original_dtype = x.dtype

            def reshape(out: torch.Tensor) -> torch.Tensor:
                return out
        else:
            canonical_signal, reshape, original_dtype = _canonicalize_1d(x)
        x_working = canonical_signal.to(torch.float64)
        B, C, L = x_working.shape
        device = x_working.device

        trend_kernel = cast(torch.Tensor, self.trend_kernel)
        detail_kernel = cast(torch.Tensor, self.detail_kernel)
        trend = self._smooth(x_working, trend_kernel)
        local = self._smooth(x_working, detail_kernel)
        residual = x_working - local
        scale = residual.std(dim=-1, keepdim=True).clamp_min(self._eps)
        threshold = scale * self.spike_threshold
        residual = torch.where(
            residual.abs() > threshold,
            residual * self.spike_damping,
            residual,
        )
        residual = residual * self.detail_preservation
        blended = (1.0 - self.trend_scaling) * local + self.trend_scaling * trend
        x_processed = blended + residual

        remainder = (L - self.r) % self.stride
        pad_right = 0 if remainder == 0 else self.stride - remainder
        if pad_right > 0:
            x_processed = F.pad(x_processed, (0, pad_right), mode="reflect")
        L_pad = x_processed.shape[-1]

        ranges = x_processed.unfold(dimension=-1, size=self.r, step=self.stride)
        N_ranges = ranges.shape[2]

        output = torch.zeros(B * C, L_pad, device=device, dtype=x_processed.dtype)
        count = torch.zeros(B * C, L_pad, device=device, dtype=x_processed.dtype)

        starts = torch.arange(N_ranges, device=device) * self.stride
        idx = starts.unsqueeze(1) + torch.arange(self.r, device=device).unsqueeze(0)
        idx_flat = idx.reshape(-1)
        idx_expanded = idx_flat.unsqueeze(0).expand(B * C, -1)

        segments_flat = ranges.reshape(B * C, N_ranges * self.r)
        ones = torch.ones_like(segments_flat)

        output.scatter_add_(1, idx_expanded, segments_flat)
        count.scatter_add_(1, idx_expanded, ones)

        output = output / count.clamp(min=1.0)
        output = output.view(B, C, L_pad)
        return reshape(output[:, :, :L]).to(original_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply denoising.

        Supports shapes [L], [B, L], [B, C, L] and preserves the original shape.
        """
        canonical, reshape, original_dtype = _canonicalize_1d(x)
        input_fp64 = canonical.to(torch.float64)
        current = input_fp64
        inhibit_mask: torch.Tensor | None = None

        if self.mode == "multiscale":
            for _ in range(self.iterations):
                current = self._denoise_signal(current, canonical=True)
        elif self.mode == "fractal":
            fd = self._estimate_fractal_dimension(input_fp64)
            inhibit_mask = fd > self.fractal_dim_threshold
            effective_acceptor_iterations = (
                self.acceptor_iterations if not self.do_no_harm else max(1, self.iterations_fractal)
            )
            if not inhibit_mask.all():
                for _ in range(effective_acceptor_iterations):
                    current = self._denoise_fractal(current, canonical=True)
                if inhibit_mask.any():
                    current = torch.where(inhibit_mask.unsqueeze(-1), input_fp64, current)
            else:
                current = input_fp64
        else:
            raise ValueError(
                f"Unsupported mode '{self.mode}'. Supported modes are: 'multiscale', 'fractal'"
            )

        if self.log_mutual_information:
            mi = self._mutual_information(input_fp64, current)
            self.last_mutual_information = float(mi.item())
            logger.info("Mutual information (input→output): %.6f", self.last_mutual_information)

        return reshape(current).to(original_dtype)

    def _fractal_params(self) -> tuple[int, int, int, int]:
        r = self.range_size
        if r <= 0:
            raise ValueError("range_size must be positive")
        if self.domain_scale <= 0:
            raise ValueError("domain_scale must be positive")
        d = r * self.domain_scale
        if d % r != 0:
            raise ValueError(
                f"domain length (domain_scale * range_size) must be divisible by range_size; "
                f"range_size={r}, domain_scale={self.domain_scale}"
            )
        stride = r // 2 if self.overlap else r
        if stride <= 0:
            raise ValueError(
                f"Invalid stride ({stride}) derived from range_size ({r}) and overlap setting. "
                "Ensure range_size is positive."
            )
        factor = d // r
        return r, d, stride, factor

    def _estimate_fractal_dimension(self, x: torch.Tensor) -> torch.Tensor:
        """Approximate 1D fractal dimension using a Higuchi-style estimator."""
        B, C, L = x.shape
        ks = (1, 2, 4, 8)
        lengths = []
        ks_used = []
        for k in ks:
            if k >= L:
                break
            diff = x[..., k:] - x[..., :-k]
            if diff.numel() == 0:
                continue
            norm = (L - 1) / (diff.shape[-1] * k)
            lengths.append(diff.abs().sum(dim=-1) * norm)
            ks_used.append(k)

        if len(lengths) < 2:
            return torch.zeros(B, C, device=x.device, dtype=x.dtype)

        Lks = torch.stack(lengths, dim=-1)
        log_L = torch.log(Lks.clamp_min(self._eps))
        k_tensor = torch.tensor(ks_used, device=x.device, dtype=x.dtype)
        log_k_inv = torch.log(1.0 / k_tensor)
        log_k_centered = log_k_inv - log_k_inv.mean()
        denominator = (log_k_centered**2).sum().clamp_min(self._eps)
        slope = (log_L * log_k_centered).sum(dim=-1) / denominator
        return slope

    def _mutual_information(self, x: torch.Tensor, y: torch.Tensor, bins: int = 32) -> torch.Tensor:
        """Estimate mutual information between flattened tensors."""
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        eps = self._eps

        x_min, x_max = x_flat.min(), x_flat.max()
        y_min, y_max = y_flat.min(), y_flat.max()

        if (x_max - x_min).abs() < eps or (y_max - y_min).abs() < eps:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        x_bins = (
            ((x_flat - x_min) / (x_max - x_min + eps) * (bins - 1))
            .long()
            .clamp(0, bins - 1)
        )
        y_bins = (
            ((y_flat - y_min) / (y_max - y_min + eps) * (bins - 1))
            .long()
            .clamp(0, bins - 1)
        )

        joint_idx = x_bins * bins + y_bins
        joint_hist = torch.bincount(joint_idx, minlength=bins * bins).to(
            dtype=x.dtype, device=x.device
        )
        joint_prob = joint_hist / joint_hist.sum().clamp_min(eps)
        joint_prob_2d = joint_prob.view(bins, bins)
        p_x = joint_prob_2d.sum(dim=1)
        p_y = joint_prob_2d.sum(dim=0)
        denom = (p_x.unsqueeze(1) * p_y.unsqueeze(0)).clamp_min(eps)
        return (joint_prob_2d * torch.log((joint_prob_2d + eps) / denom)).sum()

    def _denoise_fractal(self, x: torch.Tensor, *, canonical: bool = False) -> torch.Tensor:
        reshape: Callable[[torch.Tensor], torch.Tensor]
        if canonical:
            canonical_signal = x
            original_dtype = x.dtype

            def reshape(out: torch.Tensor) -> torch.Tensor:
                return out
        else:
            canonical_signal, reshape, original_dtype = _canonicalize_1d(x)

        x_working = canonical_signal.to(torch.float64)
        B, C, L = x_working.shape
        device = x_working.device

        r, d, stride, factor = self._fractal_params()

        pad_base = max(L, d, r)
        pad_align = (stride - ((pad_base - r) % stride)) % stride
        L_pad = pad_base + pad_align
        pad_right = max(0, L_pad - L)
        pad_mode = "reflect" if (L > 1 and pad_right < L) else "replicate"

        if pad_right > 0:
            x_padded = F.pad(x_working, (0, pad_right), mode=pad_mode)
        else:
            x_padded = x_working

        smooth_kernel = cast(torch.Tensor, self.fractal_kernel)
        smoothed = self._smooth(x_working, smooth_kernel)
        if pad_right > 0:
            smoothed = F.pad(smoothed, (0, pad_right), mode=pad_mode)

        ranges = x_padded.unfold(dimension=-1, size=r, step=stride)
        N_ranges = ranges.shape[2]
        baseline_mean = ranges.mean(dim=-1, keepdim=True)
        baseline_mse = ((ranges - baseline_mean) ** 2).mean(dim=-1)

        domains = smoothed.unfold(dimension=-1, size=d, step=1)
        N_domains = domains.shape[2]
        domains_flat = domains.reshape(B * C * N_domains, 1, d)
        contracted = F.avg_pool1d(domains_flat, kernel_size=factor, stride=factor)
        domain_pool = contracted.view(B, C, N_domains, r)
        flipped = torch.flip(domain_pool, dims=[-1])
        domain_pool = torch.cat([domain_pool, flipped], dim=2)

        num_domains = domain_pool.shape[2]
        if num_domains == 0:
            return reshape(x_padded[:, :, :L]).to(original_dtype)

        candidate_pool = min(num_domains, min(self.population_size * 2, self.max_population_ci))
        top_k = candidate_pool
        pop_size = min(self.population_size, top_k)
        if top_k == 0 or pop_size == 0:
            return reshape(x_padded[:, :, :L]).to(original_dtype)
        var_pool = domain_pool.var(dim=-1, unbiased=False)
        _, top_idx = torch.topk(var_pool, k=top_k, dim=2, largest=False)
        top_domains = torch.gather(
            domain_pool,
            2,
            top_idx.unsqueeze(-1).expand(-1, -1, -1, r),
        )

        top_domains_expanded = top_domains.unsqueeze(2).expand(-1, -1, N_ranges, -1, -1)
        base_idx = torch.arange(N_ranges * pop_size, device=device).view(1, 1, N_ranges, pop_size)
        sample_idx = torch.remainder(base_idx, top_k).expand(B, C, -1, -1)
        candidate_domains = torch.gather(
            top_domains_expanded,
            3,
            sample_idx.unsqueeze(-1).expand(-1, -1, -1, -1, r),
        )

        ranges_expanded = ranges.unsqueeze(3)
        range_mean = baseline_mean.unsqueeze(3)
        domain_mean = candidate_domains.mean(dim=-1, keepdim=True)
        cov = ((ranges_expanded - range_mean) * (candidate_domains - domain_mean)).mean(dim=-1)
        var_d = candidate_domains.var(dim=-1, unbiased=False) + self.ridge_lambda

        s = cov / (var_d + self._eps)
        s = s.clamp(min=-self.s_max, max=self.s_max)
        s = torch.where(s.abs() < self.s_threshold, torch.zeros_like(s), s)
        o = range_mean.squeeze(-1) - s * domain_mean.squeeze(-1)
        recon = s.unsqueeze(-1) * candidate_domains + o.unsqueeze(-1)
        mse = ((recon - ranges_expanded) ** 2).mean(dim=-1)

        best_idx = mse.argmin(dim=3)
        mse_best = torch.gather(mse, 3, best_idx.unsqueeze(-1)).squeeze(-1)
        recon_best = torch.gather(
            recon,
            3,
            best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, r),
        ).squeeze(3)

        gate_ratio = (
            1.0 - self.inhibition_epsilon
            if self.inhibition_epsilon is not None
            else self.harm_ratio
        )
        if self.do_no_harm:
            apply_fractal = mse_best < baseline_mse * gate_ratio
        else:
            apply_fractal = torch.ones_like(mse_best, dtype=torch.bool)
        blended = 0.5 * (recon_best + ranges)
        selected = torch.where(apply_fractal.unsqueeze(-1), blended, ranges)

        output = torch.zeros(B * C, L_pad, device=device, dtype=x_working.dtype)
        count = torch.zeros_like(output)

        starts = torch.arange(N_ranges, device=device) * stride
        idx = starts.unsqueeze(1) + torch.arange(r, device=device).unsqueeze(0)
        idx_flat = idx.reshape(-1)
        idx_expanded = idx_flat.unsqueeze(0).expand(B * C, -1)

        segments_flat = selected.reshape(B * C, N_ranges * r)
        ones = torch.ones_like(segments_flat)

        output.scatter_add_(1, idx_expanded, segments_flat)
        count.scatter_add_(1, idx_expanded, ones)

        output = output / count.clamp(min=1.0)
        output = output.view(B, C, L_pad)
        return reshape(output[:, :, :L]).to(original_dtype)
