"""
Tests for MyceliumFractalNet Core Numerical Engines.

This module provides:
1. Stability smoke tests (NaN/Inf, range checks)
2. Determinism tests (reproducibility with fixed seed)
3. Performance sanity tests (reasonable execution time)

Reference: docs/ARCHITECTURE.md for expected parameter ranges.
"""

from __future__ import annotations

