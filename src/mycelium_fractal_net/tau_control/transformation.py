"""Transformation protocol — Gamma(C, Phi) with bounded-jump guard.

The ONLY writer to meta-rules C.
Activates when Phi >= tau.
MANDATORY bounded-jump check before any C write.

Gamma(C, Phi):
  scale = tanh(Phi / 5) in [0, 1)
  new learning_rate_bounds: expand proportionally to scale
  new contraction_factor: soften proportionally to scale

If |V_new - V_old| > delta_max -> REJECT, return (old_C, False).

Read-only on all other system state. Writes ONLY to MetaRuleSpace.
gamma (the scaling exponent) is NEVER read here.

Ref: Vasylenko (2026)
"""

from __future__ import annotations

import numpy as np

from .lyapunov import LyapunovMonitor
from .types import MetaRuleSpace, NormSpace

__all__ = ["TransformationProtocol"]


class TransformationProtocol:
    """Bounded transformation of meta-rules C."""

    def __init__(
        self,
        lyapunov: LyapunovMonitor | None = None,
    ) -> None:
        self.lyapunov = lyapunov or LyapunovMonitor()
        self._transform_count: int = 0
        self._reject_count: int = 0

    @property
    def transform_count(self) -> int:
        return self._transform_count

    @property
    def reject_count(self) -> int:
        return self._reject_count

    def transform(
        self,
        meta: MetaRuleSpace,
        phi: float,
        free_energy: float,
        norm: NormSpace,
        norm_origin: NormSpace,
        meta_origin: MetaRuleSpace,
    ) -> tuple[MetaRuleSpace, bool]:
        """Apply Gamma(C, Phi) with bounded-jump guard.

        Returns (new_C, accepted). If rejected, returns (old_C, False).
        """
        # Compute V before transformation
        v_old = self.lyapunov.compute(
            free_energy, norm, norm_origin, meta, meta_origin,
        )

        # Gamma(C, Phi): scale = tanh(Phi / 5)
        scale = float(np.tanh(phi / 5.0))

        # Expand learning rate bounds proportionally
        lr_min, lr_max = meta.learning_rate_bounds
        lr_expansion = scale * 0.05  # max 5% expansion per transform
        new_lr_bounds = (
            max(lr_min * (1.0 - lr_expansion), 1e-6),
            min(lr_max * (1.0 + lr_expansion), 1.0),
        )

        # Soften contraction factor
        cf_softening = scale * 0.02  # max 2% softening
        new_cf = min(meta.contraction_factor + cf_softening, 0.999)

        new_meta = MetaRuleSpace(
            learning_rate_bounds=new_lr_bounds,
            contraction_factor=new_cf,
            entropy_target=meta.entropy_target,
        )

        # Compute V after proposed transformation
        v_new = self.lyapunov.compute(
            free_energy, norm, norm_origin, new_meta, meta_origin,
        )

        # MANDATORY: bounded-jump check
        if not self.lyapunov.bounded_jump_ok(v_old.v_total, v_new.v_total):
            self._reject_count += 1
            return meta, False

        self._transform_count += 1
        return new_meta, True
