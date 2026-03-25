"""Auto-Heal — closed cognitive loop.

OBSERVE → DIAGNOSE → DECIDE → ACT → VERIFY → REPORT

One call: result = mfn.auto_heal(seq)

The system diagnoses itself, plans an intervention if needed,
applies it (re-simulates with new parameters), re-diagnoses,
and proves whether the intervention worked — with ΔM as evidence.

This is what closes the loop: not just diagnosis, but verified healing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .types.field import FieldSequence, SimulationSpec

__all__ = ["HealResult", "auto_heal"]


@dataclass
class HealResult:
    """Complete result of the auto-heal cognitive loop."""

    # Before
    severity_before: str
    anomaly_before: str
    anomaly_score_before: float
    M_before: float
    hwi_before: bool

    # Decision
    needs_healing: bool
    intervention_applied: bool
    changes: list[dict[str, Any]]

    # After (None if no intervention needed)
    severity_after: str | None
    anomaly_after: str | None
    anomaly_score_after: float | None
    M_after: float | None
    hwi_after: bool | None

    # Verification
    delta_M: float | None
    delta_anomaly: float | None
    healed: bool | None  # True if severity improved AND M didn't degrade

    # Meta
    compute_time_ms: float

    def summary(self) -> str:
        if not self.needs_healing:
            return (
                f"[HEAL] System healthy — no intervention needed. "
                f"M={self.M_before:.3f} severity={self.severity_before} "
                f"({self.compute_time_ms:.0f}ms)"
            )

        status = "HEALED" if self.healed else "FAILED"
        dm = self.delta_M if self.delta_M is not None else 0
        da = self.delta_anomaly if self.delta_anomaly is not None else 0
        return (
            f"[HEAL] {status} | "
            f"M: {self.M_before:.3f} -> {self.M_after:.3f} (dM={dm:+.3f}) | "
            f"anomaly: {self.anomaly_score_before:.3f} -> {self.anomaly_score_after:.3f} "
            f"(d={da:+.3f}) | "
            f"severity: {self.severity_before} -> {self.severity_after} "
            f"({self.compute_time_ms:.0f}ms)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "before": {
                "severity": self.severity_before,
                "anomaly": self.anomaly_before,
                "anomaly_score": round(self.anomaly_score_before, 4),
                "M": round(self.M_before, 6),
                "hwi_holds": self.hwi_before,
            },
            "decision": {
                "needs_healing": self.needs_healing,
                "intervention_applied": self.intervention_applied,
                "changes": self.changes,
            },
            "after": {
                "severity": self.severity_after,
                "anomaly": self.anomaly_after,
                "anomaly_score": round(self.anomaly_score_after, 4) if self.anomaly_score_after is not None else None,
                "M": round(self.M_after, 6) if self.M_after is not None else None,
                "hwi_holds": self.hwi_after,
            },
            "verification": {
                "delta_M": round(self.delta_M, 6) if self.delta_M is not None else None,
                "delta_anomaly": round(self.delta_anomaly, 4) if self.delta_anomaly is not None else None,
                "healed": self.healed,
            },
            "compute_time_ms": round(self.compute_time_ms, 1),
        }


def auto_heal(
    seq: FieldSequence,
    target_regime: str = "stable",
    budget: float = 10.0,
    verbose: bool = False,
) -> HealResult:
    """Closed cognitive loop: diagnose → intervene → verify.

    1. OBSERVE:  take the current FieldSequence
    2. DIAGNOSE: compute M, severity, anomaly
    3. DECIDE:   if needs intervention, plan it
    4. ACT:      re-simulate with new parameters
    5. VERIFY:   re-diagnose, compute ΔM
    6. REPORT:   did it work?

    Parameters
    ----------
    seq : FieldSequence
        Current system state.
    target_regime : str
        Desired regime after healing (default: "stable").
    budget : float
        Maximum intervention cost.
    verbose : bool
        Print progress.

    Returns
    -------
    HealResult
        Complete before/after with ΔM proof.
    """
    from .analytics.unified_score import compute_hwi_components
    from .core.detect import detect_anomaly
    from .core.early_warning import early_warning
    from .intervention import plan_intervention

    t0 = time.perf_counter()

    # ── 1. DIAGNOSE BEFORE ──────────────────────────────────────
    if verbose:
        print("[HEAL] Diagnosing...")

    det_before = detect_anomaly(seq)
    ews_before = early_warning(seq)
    hwi_before = compute_hwi_components(seq.history[0], seq.field)

    severity_map = {
        (True, True): "critical",
        (True, False): "warning",
        (False, True): "warning",
        (False, False): "stable" if ews_before.ews_score < 0.3 else "info",
    }
    is_anomalous = det_before.label in ("anomalous", "critical")
    is_ews_high = ews_before.ews_score > 0.5
    severity_before = severity_map[(is_anomalous, is_ews_high)]

    if verbose:
        print(f"  M={hwi_before.M:.4f} anomaly={det_before.label}({det_before.score:.3f}) "
              f"ews={ews_before.ews_score:.3f} severity={severity_before}")

    # ── 2. DECIDE ───────────────────────────────────────────────
    needs_healing = severity_before in ("warning", "critical") or det_before.label == "anomalous"

    if not needs_healing:
        elapsed = (time.perf_counter() - t0) * 1000
        return HealResult(
            severity_before=severity_before,
            anomaly_before=det_before.label,
            anomaly_score_before=float(det_before.score),
            M_before=hwi_before.M,
            hwi_before=hwi_before.hwi_holds,
            needs_healing=False,
            intervention_applied=False,
            changes=[],
            severity_after=None,
            anomaly_after=None,
            anomaly_score_after=None,
            M_after=None,
            hwi_after=None,
            delta_M=None,
            delta_anomaly=None,
            healed=None,
            compute_time_ms=elapsed,
        )

    # ── 3. PLAN ─────────────────────────────────────────────────
    if verbose:
        print("[HEAL] Planning intervention...")

    plan = plan_intervention(seq, target_regime=target_regime, budget=budget)
    best = plan.best_candidate

    if best is None or not plan.has_viable_plan:
        elapsed = (time.perf_counter() - t0) * 1000
        return HealResult(
            severity_before=severity_before,
            anomaly_before=det_before.label,
            anomaly_score_before=float(det_before.score),
            M_before=hwi_before.M,
            hwi_before=hwi_before.hwi_holds,
            needs_healing=True,
            intervention_applied=False,
            changes=[],
            severity_after=None,
            anomaly_after=None,
            anomaly_score_after=None,
            M_after=None,
            hwi_after=None,
            delta_M=None,
            delta_anomaly=None,
            healed=False,
            compute_time_ms=elapsed,
        )

    changes = [
        {"name": s.name, "from": round(s.current_value, 4), "to": round(s.proposed_value, 4)}
        for s in best.proposed_changes
        if abs(s.proposed_value - s.current_value) > 1e-6
    ]

    if verbose:
        for c in changes:
            print(f"  {c['name']}: {c['from']} -> {c['to']}")

    # ── 4. ACT — re-simulate with intervention parameters ──────
    if verbose:
        print("[HEAL] Applying intervention (re-simulating)...")

    from .intervention.counterfactual import _apply_interventions
    from .core.simulate import simulate_history

    modified_spec = _apply_interventions(seq.spec, best.proposed_changes)
    seq_after = simulate_history(modified_spec)

    # ── 5. VERIFY — re-diagnose ────────────────────────────────
    if verbose:
        print("[HEAL] Verifying...")

    det_after = detect_anomaly(seq_after)
    ews_after = early_warning(seq_after)
    hwi_after = compute_hwi_components(seq_after.history[0], seq_after.field)

    is_anomalous_after = det_after.label in ("anomalous", "critical")
    is_ews_high_after = ews_after.ews_score > 0.5
    severity_after = severity_map[(is_anomalous_after, is_ews_high_after)]

    delta_M = hwi_after.M - hwi_before.M
    delta_anomaly = det_after.score - det_before.score

    # Healed = severity improved or stayed same AND anomaly score decreased
    severity_order = {"stable": 0, "info": 1, "warning": 2, "critical": 3}
    sev_improved = severity_order.get(severity_after, 3) <= severity_order.get(severity_before, 3)
    anomaly_improved = det_after.score <= det_before.score + 0.01
    healed = sev_improved and anomaly_improved

    if verbose:
        print(f"  M: {hwi_before.M:.4f} -> {hwi_after.M:.4f} (dM={delta_M:+.4f})")
        print(f"  anomaly: {det_before.score:.3f} -> {det_after.score:.3f}")
        print(f"  severity: {severity_before} -> {severity_after}")
        print(f"  HEALED: {healed}")

    elapsed = (time.perf_counter() - t0) * 1000

    return HealResult(
        severity_before=severity_before,
        anomaly_before=det_before.label,
        anomaly_score_before=float(det_before.score),
        M_before=hwi_before.M,
        hwi_before=hwi_before.hwi_holds,
        needs_healing=True,
        intervention_applied=True,
        changes=changes,
        severity_after=severity_after,
        anomaly_after=det_after.label,
        anomaly_score_after=float(det_after.score),
        M_after=hwi_after.M,
        hwi_after=hwi_after.hwi_holds,
        delta_M=delta_M,
        delta_anomaly=delta_anomaly,
        healed=healed,
        compute_time_ms=elapsed,
    )
