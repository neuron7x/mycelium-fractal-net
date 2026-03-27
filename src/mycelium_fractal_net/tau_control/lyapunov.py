"""Lyapunov monitor — composite stability functional V.

V = V_x + alpha * V_S + beta * V_C

V_x = max(0, F) from ThermodynamicKernel energy_trajectory
  # APPROXIMATION: V_x from last energy value, not exact free energy functional
  # GAP: full Friston proof requires differentiable F path through state space

V_S = ||centroid(S) - centroid(S_0)||^2 + (1 - confidence)
V_C = KL(C || C_0) + mu * max(0, H* - H(C))^2 + nu * max(0, H(C) - H*)^2

Read-only: does not modify system state.

Ref: Friston (2010), Vasylenko (2026)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .types import MetaRuleSpace, NormSpace

__all__ = ["LyapunovMonitor", "LyapunovState"]


@dataclass(frozen=True)
class LyapunovState:
    v_x: float
    v_s: float
    v_c: float
    v_total: float

    def to_dict(self) -> dict[str, Any]:
        return {"v_x": self.v_x, "v_s": self.v_s, "v_c": self.v_c, "v_total": self.v_total}


class LyapunovMonitor:
    """Composite Lyapunov functional V = V_x + alpha*V_S + beta*V_C."""

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.01,
        mu: float = 1.0,
        nu: float = 1.0,
        delta_max: float = 2.0,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.nu = nu
        self.delta_max = delta_max
        self._history: list[float] = []

    @property
    def history(self) -> list[float]:
        return list(self._history)

    def compute(
        self,
        free_energy: float,
        norm: NormSpace,
        norm_origin: NormSpace,
        meta: MetaRuleSpace,
        meta_origin: MetaRuleSpace,
    ) -> LyapunovState:
        """Compute composite Lyapunov value.

        # APPROXIMATION: V_x from heuristic free energy, not exact Friston functional
        """
        # V_x: free energy component
        v_x = max(0.0, free_energy)

        # V_S: norm space drift + uncertainty
        drift = norm.drift_from_origin(norm_origin)
        v_s = drift**2 + (1.0 - norm.confidence)

        # V_C: meta-rule divergence + entropy penalty
        kl = meta.kl_divergence(meta_origin)
        h = meta.entropy()
        h_star = meta.entropy_target
        inertia_penalty = self.mu * max(0.0, h_star - h) ** 2
        chaos_penalty = self.nu * max(0.0, h - h_star) ** 2
        v_c = kl + inertia_penalty + chaos_penalty

        v_total = v_x + self.alpha * v_s + self.beta * v_c
        self._history.append(v_total)

        return LyapunovState(v_x=v_x, v_s=v_s, v_c=v_c, v_total=v_total)

    def bounded_jump_ok(self, v_old: float, v_new: float) -> bool:
        """True if |dV| <= delta_max (bounded jump constraint)."""
        return abs(v_new - v_old) <= self.delta_max

    def meta_stable_trend(self, window: int = 50) -> float:
        """Mean dV over recent window. <= 0 means stable."""
        if len(self._history) < 2:
            return 0.0
        recent = self._history[-window:]
        if len(recent) < 2:
            return 0.0
        diffs = np.diff(recent)
        return float(np.mean(diffs))

    def reset(self) -> None:
        self._history.clear()
