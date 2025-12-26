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
        self._eps = 1e-6
        self.register_buffer("detail_kernel", _normalized_kernel(base_window))
        self.register_buffer("trend_kernel", _normalized_kernel(base_window * 2 + 1))

    def _smooth(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        pad = kernel.shape[-1] // 2
        padded = F.pad(x, (pad, pad), mode="reflect")
        channels = padded.shape[1]
        weight = kernel.expand(channels, -1, -1)
        return F.conv1d(padded, weight, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply denoising.

        Supports shapes [L], [B, L], [B, C, L] and preserves the original shape.
        """
        canonical, reshape, original_dtype = _canonicalize_1d(x)
        current = canonical.to(torch.float32)

        for _ in range(self.iterations):
            trend_kernel = cast(torch.Tensor, self.trend_kernel)
            detail_kernel = cast(torch.Tensor, self.detail_kernel)
            trend = self._smooth(current, trend_kernel)
            local = self._smooth(current, detail_kernel)
            residual = current - local

            scale = residual.std(dim=-1, keepdim=True).clamp_min(self._eps)
            threshold = scale * self.spike_threshold
            residual = torch.where(
                residual.abs() > threshold,
                residual * self.spike_damping,
                residual,
            )

            residual = residual * self.detail_preservation
            combined = (1.0 - self.trend_scaling) * local + self.trend_scaling * trend
            current = combined + residual

        return reshape(current).to(original_dtype)
