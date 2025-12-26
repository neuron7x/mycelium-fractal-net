"""
Fractal Analysis Module.

This module provides the public API for fractal dimension estimation and
IFS fractal generation, re-exporting validated implementations for
backward compatibility, plus the new FractalGrowthEngine for advanced use.

Conceptual domain: Fractal geometry, box-counting dimension

Reference:
    - docs/MFN_MATH_MODEL.md Section 3 (Fractal Growth and Dimension)
    - docs/ARCHITECTURE.md Section 3 (Fractal Analysis)
    - docs/MFN_FEATURE_SCHEMA.md (D_box feature)

Mathematical Model:
    Box-counting dimension:
        D = lim(ε→0) ln(N(ε)) / ln(1/ε)

    IFS transformation:
        [x', y'] = [[a,b],[c,d]] * [x,y] + [e,f]

    Contraction requirement: |ad - bc| < 1

Expected ranges:
    D ∈ [1.4, 1.9] for biological mycelium patterns
    D ≈ 1.585 for Sierpinski triangle (exact)

Example:
    >>> from mycelium_fractal_net.core.fractal import estimate_fractal_dimension
    >>> import numpy as np
    >>> binary = np.random.default_rng(42).random((64, 64)) > 0.5
    >>> D = estimate_fractal_dimension(binary)
    >>> 1.0 < D < 2.5
    True
"""

from __future__ import annotations

from ..model import estimate_fractal_dimension, generate_fractal_ifs
from .fractal_growth_engine import (
    DEFAULT_MIN_BOX_SIZE,
    DEFAULT_NUM_POINTS,
    DEFAULT_NUM_SCALES,
    DEFAULT_NUM_TRANSFORMS,
    DEFAULT_SCALE_MAX,
    DEFAULT_SCALE_MIN,
    DEFAULT_TRANSLATION_RANGE,
    FractalConfig,
    FractalGrowthEngine,
    FractalMetrics,
)

__all__ = [
    # Constants
    "DEFAULT_NUM_POINTS",
    "DEFAULT_NUM_TRANSFORMS",
    "DEFAULT_SCALE_MIN",
    "DEFAULT_SCALE_MAX",
    "DEFAULT_TRANSLATION_RANGE",
    "DEFAULT_MIN_BOX_SIZE",
    "DEFAULT_NUM_SCALES",
    # Classes
    "FractalConfig",
    "FractalGrowthEngine",
    "FractalMetrics",
    # Functions
    "estimate_fractal_dimension",
    "generate_fractal_ifs",
]
