"""Fractal-inspired 1D denoiser."""

from __future__ import annotations

from typing import Callable, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    kernel = torch.ones(1, 1, window, dtype=torch.float32)
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
    ) -> None:
        super().__init__()
        self.trend_scaling = trend_scaling
        self.detail_preservation = detail_preservation
        self.spike_threshold = spike_threshold
        self.spike_damping = spike_damping
        self.iterations = iterations
        self.smooth_kernel = base_window
        self.stride = max(1, base_window // 2)
        self.r = base_window
        self._eps = 1e-6
        self.register_buffer("detail_kernel", _normalized_kernel(base_window))
        self.register_buffer("trend_kernel", _normalized_kernel(base_window * 2 + 1))

    def _smooth(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        pad = kernel.shape[-1] // 2
        padded = F.pad(x, (pad, pad), mode="reflect")
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
        x_working = canonical_signal.to(torch.float32)
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
        current = canonical.to(torch.float32)

        for _ in range(self.iterations):
            current = self._denoise_signal(current, canonical=True)

        return reshape(current).to(original_dtype)
