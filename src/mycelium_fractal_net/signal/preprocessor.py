"""Preprocessing pipeline for 1D signals."""

from __future__ import annotations

from typing import Literal, Mapping, TypedDict

import torch
import torch.nn as nn

from .denoise_1d import OptimizedFractalDenoise1D, _canonicalize_1d


class _PresetConfig(TypedDict):
    base_window: int
    trend_scaling: float
    detail_preservation: float
    spike_threshold: float
    spike_damping: float
    iterations: int
    normalize: bool
    center: bool
    post_gain: float


class Fractal1DPreprocessor(nn.Module):
    """Preset-driven 1D preprocessing with fractal denoising."""

    _PRESETS: Mapping[
        Literal["generic", "markets", "ecg"],
        _PresetConfig,
    ] = {
        "generic": {
            "base_window": 5,
            "trend_scaling": 0.55,
            "detail_preservation": 0.88,
            "spike_threshold": 3.5,
            "spike_damping": 0.4,
            "iterations": 2,
            "normalize": True,
            "center": True,
            "post_gain": 1.0,
        },
        "markets": {
            "base_window": 7,
            "trend_scaling": 0.6,
            "detail_preservation": 0.85,
            "spike_threshold": 3.2,
            "spike_damping": 0.35,
            "iterations": 2,
            "normalize": True,
            "center": True,
            "post_gain": 1.0,
        },
        "ecg": {
            "base_window": 5,
            "trend_scaling": 0.5,
            "detail_preservation": 0.9,
            "spike_threshold": 3.0,
            "spike_damping": 0.3,
            "iterations": 3,
            "normalize": True,
            "center": True,
            "post_gain": 1.0,
        },
    }

    def __init__(self, preset: Literal["generic", "markets", "ecg"] = "generic") -> None:
        super().__init__()
        if preset not in self._PRESETS:
            raise ValueError(f"Unsupported preset: {preset}")

        cfg = self._PRESETS[preset]
        self.normalize = cfg["normalize"]
        self.center = cfg["center"]
        self.post_gain = cfg["post_gain"]
        self._eps = 1e-6

        self.denoiser = OptimizedFractalDenoise1D(
            base_window=cfg["base_window"],
            trend_scaling=cfg["trend_scaling"],
            detail_preservation=cfg["detail_preservation"],
            spike_threshold=cfg["spike_threshold"],
            spike_damping=cfg["spike_damping"],
            iterations=cfg["iterations"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run preset preprocessing pipeline."""
        canonical, reshape, original_dtype = _canonicalize_1d(x)
        processed = canonical.to(torch.float32)

        if self.center:
            processed = processed - processed.mean(dim=-1, keepdim=True)

        processed = self.denoiser(processed)

        if self.normalize:
            mean = processed.mean(dim=-1, keepdim=True)
            std = processed.std(dim=-1, keepdim=True).clamp_min(self._eps)
            processed = (processed - mean) / std

        if self.post_gain != 1.0:
            processed = processed * self.post_gain

        return reshape(processed).to(original_dtype)
