"""Mathematical Frontier — unified report from 5 mechanisms.

Single call: report = run_math_frontier(seq)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .causal_emergence import (
    compute_causal_emergence,
    discretize_turing_field,
)
from .rmt_spectral import RMTDiagnostics, rmt_diagnostics
from .tda_ews import TopologicalSignature, compute_tda
from .wasserstein_geometry import wasserstein_distance

__all__ = ["MathFrontierReport", "run_math_frontier"]


@dataclass
class MathFrontierReport:
    """Unified report from all 5 mathematical mechanisms."""

    topology: TopologicalSignature
    w2_trajectory_speed: float
    causal_emergence_score: float
    rmt: RMTDiagnostics | None
    compute_time_ms: float

    def summary(self) -> str:
        """Single-line summary."""
        topo = (
            f"b0={self.topology.beta_0} b1={self.topology.beta_1} "
            f"TP0={self.topology.total_pers_0:.3f}"
        )
        w2 = f"W2={self.w2_trajectory_speed:.4f}"
        ce = f"CE={self.causal_emergence_score:.4f}"
        rmt = (
            f"r={self.rmt.r_ratio:.3f}({self.rmt.structure_type[:8]})"
            if self.rmt
            else "RMT=skip"
        )
        return f"[MATH] {topo} | {w2} | {ce} | {rmt} ({self.compute_time_ms:.0f}ms)"

    def to_dict(self) -> dict[str, Any]:
        """Serialize."""
        return {
            "topology": self.topology.to_dict(),
            "w2_trajectory_speed": round(self.w2_trajectory_speed, 4),
            "causal_emergence": round(self.causal_emergence_score, 4),
            "rmt": self.rmt.to_dict() if self.rmt else None,
            "compute_time_ms": round(self.compute_time_ms, 1),
        }


def run_math_frontier(
    seq: Any,
    run_rmt: bool = True,
) -> MathFrontierReport:
    """Run all 5 mechanisms on a FieldSequence. ~100ms for N=32."""
    t0 = time.perf_counter()

    # 1. TDA
    topo = compute_tda(seq.field, min_persistence_frac=0.005)

    # 2. W2 trajectory speed
    w2_speed = wasserstein_distance(seq.history[0], seq.field, method="sliced")

    # 3. Causal Emergence
    n_steps = seq.history.shape[0]
    states = np.array([discretize_turing_field(seq.history[t]) for t in range(n_steps)])
    tpm = np.zeros((4, 4))
    for t in range(len(states) - 1):
        tpm[states[t], states[t + 1]] += 1
    row_s = tpm.sum(axis=1, keepdims=True)
    row_s[row_s < 1] = 1
    tpm /= row_s
    ce_result = compute_causal_emergence(tpm)
    ce_score = float(ce_result.CE_macro)

    # 4. RMT (from Physarum Laplacian)
    rmt_result = None
    if run_rmt:
        try:
            from mycelium_fractal_net.bio.memory_anonymization import GapJunctionDiffuser
            from mycelium_fractal_net.bio.physarum import PhysarumEngine

            N = seq.field.shape[0]
            eng = PhysarumEngine(N)
            src = seq.field > 0
            snk = seq.field < -0.05
            phys = eng.initialize(src, snk)
            for _ in range(3):
                phys = eng.step(phys, src, snk)
            diff = GapJunctionDiffuser()
            L = diff.build_laplacian(phys.D_h, phys.D_v).toarray()
            rmt_result = rmt_diagnostics(L)
        except Exception:  # noqa: S110
            pass  # RMT requires bio extras — graceful skip

    elapsed = (time.perf_counter() - t0) * 1000
    return MathFrontierReport(
        topology=topo,
        w2_trajectory_speed=w2_speed,
        causal_emergence_score=ce_score,
        rmt=rmt_result,
        compute_time_ms=elapsed,
    )
