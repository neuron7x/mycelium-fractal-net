"""Auto-Heal — closed cognitive loop with experience memory.

OBSERVE → DIAGNOSE → DECIDE → ACT → VERIFY → LEARN → REPORT

The system diagnoses itself, plans an intervention if needed,
applies it (re-simulates), re-diagnoses, and proves whether
the intervention worked — with ΔM as evidence.

After each heal, the outcome is stored in ExperienceMemory.
After enough experiences, the system predicts outcomes before
running expensive counterfactual simulations. Prediction error
reveals where the system doesn't understand itself.

    result = mfn.auto_heal(seq)           # first call: brute-force
    ...                                    # 50 calls later
    result = mfn.auto_heal(seq)           # uses learned predictions
    print(result.prediction_used)          # True
    print(result.prediction_error)         # 0.03 (low = understood)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .types.field import FieldSequence, SimulationSpec

__all__ = ["ExperienceMemory", "HealResult", "auto_heal"]


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIENCE MEMORY — the system learns from its own interventions
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class _Experience:
    """One (state, action, outcome) triple."""

    # State before
    M_before: float
    anomaly_before: float
    alpha: float
    threshold: float

    # Action (intervention deltas)
    delta_alpha: float
    delta_anomaly_target: float

    # Outcome
    M_after: float
    anomaly_after: float
    healed: bool


class ExperienceMemory:
    """Accumulates auto_heal experiences. Predicts outcomes via k-NN.

    After min_experiences calls, predict() returns expected M_after
    for a given (state, action) pair using the k nearest historical
    experiences. Prediction error = where the system doesn't understand itself.
    """

    def __init__(self, min_experiences: int = 20, k: int = 3) -> None:
        self.min_experiences = min_experiences
        self.k = k
        self._experiences: list[_Experience] = []

    @property
    def size(self) -> int:
        return len(self._experiences)

    @property
    def can_predict(self) -> bool:
        return self.size >= self.min_experiences

    def store(self, exp: _Experience) -> None:
        self._experiences.append(exp)

    def predict(
        self, M_before: float, anomaly_before: float,
        alpha: float, threshold: float,
        delta_alpha: float,
    ) -> tuple[float, float, float]:
        """Predict (M_after, anomaly_after, confidence) from k nearest experiences.

        Returns (predicted_M_after, predicted_anomaly_after, confidence).
        Confidence = 1/(1 + mean_distance). Higher = more certain.
        """
        if not self.can_predict:
            return 0.0, 0.0, 0.0

        query = np.array([M_before, anomaly_before, alpha, threshold, delta_alpha])
        # Normalize: each feature scaled by its range in memory
        states = np.array([
            [e.M_before, e.anomaly_before, e.alpha, e.threshold, e.delta_alpha]
            for e in self._experiences
        ])
        # Safe normalization
        ranges = states.max(axis=0) - states.min(axis=0)
        ranges = np.where(ranges > 1e-12, ranges, 1.0)
        normed_states = (states - states.min(axis=0)) / ranges
        normed_query = (query - states.min(axis=0)) / ranges

        dists = np.linalg.norm(normed_states - normed_query, axis=1)
        k = min(self.k, len(self._experiences))
        nearest_idx = np.argpartition(dists, k)[:k]

        weights = 1.0 / (dists[nearest_idx] + 1e-6)
        weights /= weights.sum()

        M_pred = sum(weights[i] * self._experiences[nearest_idx[i]].M_after for i in range(k))
        a_pred = sum(weights[i] * self._experiences[nearest_idx[i]].anomaly_after for i in range(k))
        confidence = float(1.0 / (1.0 + np.mean(dists[nearest_idx])))

        return float(M_pred), float(a_pred), confidence

    def stats(self) -> dict[str, Any]:
        if not self._experiences:
            return {"size": 0}
        healed = sum(1 for e in self._experiences if e.healed)
        Ms_after = [e.M_after for e in self._experiences]
        return {
            "size": self.size,
            "heal_rate": round(healed / self.size, 3),
            "M_after_mean": round(float(np.mean(Ms_after)), 4),
            "M_after_std": round(float(np.std(Ms_after)), 4),
            "can_predict": self.can_predict,
        }


# Global experience memory — persists across calls within process
_MEMORY = ExperienceMemory()


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
    healed: bool | None

    # Learning
    prediction_used: bool = False
    predicted_M_after: float | None = None
    prediction_error: float | None = None
    experience_count: int = 0

    # Meta
    compute_time_ms: float = 0.0

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
        pred = ""
        if self.prediction_used and self.prediction_error is not None:
            pred = f" pred_err={self.prediction_error:.3f}"
        exp = f" exp={self.experience_count}" if self.experience_count > 0 else ""
        return (
            f"[HEAL] {status} | "
            f"M: {self.M_before:.3f} -> {self.M_after:.3f} (dM={dm:+.3f}) | "
            f"anomaly: {self.anomaly_score_before:.3f} -> {self.anomaly_score_after:.3f} "
            f"(d={da:+.3f}) | "
            f"severity: {self.severity_before} -> {self.severity_after}"
            f"{pred}{exp} ({self.compute_time_ms:.0f}ms)"
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
            "learning": {
                "prediction_used": self.prediction_used,
                "predicted_M_after": round(self.predicted_M_after, 6) if self.predicted_M_after is not None else None,
                "prediction_error": round(self.prediction_error, 6) if self.prediction_error is not None else None,
                "experience_count": self.experience_count,
            },
            "compute_time_ms": round(self.compute_time_ms, 1),
        }


def get_experience_memory() -> ExperienceMemory:
    """Access the global experience memory for inspection."""
    return _MEMORY


def auto_heal(
    seq: FieldSequence,
    target_regime: str = "stable",
    budget: float = 10.0,
    verbose: bool = False,
    memory: ExperienceMemory | None = None,
) -> HealResult:
    """Closed cognitive loop: diagnose → predict → intervene → verify → learn.

    1. OBSERVE:  take the current FieldSequence
    2. DIAGNOSE: compute M, severity, anomaly
    3. PREDICT:  if enough experience, predict outcome before simulation
    4. DECIDE:   if needs intervention, plan it
    5. ACT:      re-simulate with new parameters
    6. VERIFY:   re-diagnose, compute ΔM, compare with prediction
    7. LEARN:    store (state, action, outcome) in experience memory

    After ~20 calls, the system predicts intervention outcomes from
    experience. Prediction error reveals where self-understanding fails.
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

    # ── 6. PREDICT (if memory available) ─────────────────────────
    mem = memory if memory is not None else _MEMORY
    prediction_used = False
    predicted_M = None
    prediction_error = None

    alpha_val = seq.spec.alpha if seq.spec else 0.18
    threshold_val = seq.spec.turing_threshold if seq.spec else 0.75
    delta_alpha_val = 0.0
    for s in best.proposed_changes:
        if s.name == "diffusion_alpha":
            delta_alpha_val = s.proposed_value - s.current_value

    if mem.can_predict:
        predicted_M, _, confidence = mem.predict(
            hwi_before.M, float(det_before.score),
            alpha_val, threshold_val, delta_alpha_val,
        )
        prediction_used = True
        prediction_error = abs(predicted_M - hwi_after.M)
        if verbose:
            print(f"  [LEARN] Predicted M_after={predicted_M:.4f}, actual={hwi_after.M:.4f}, "
                  f"error={prediction_error:.4f}, confidence={confidence:.3f}")

    # ── 7. LEARN — store experience ───────────────────────────────
    mem.store(_Experience(
        M_before=hwi_before.M,
        anomaly_before=float(det_before.score),
        alpha=alpha_val,
        threshold=threshold_val,
        delta_alpha=delta_alpha_val,
        delta_anomaly_target=float(det_after.score - det_before.score),
        M_after=hwi_after.M,
        anomaly_after=float(det_after.score),
        healed=healed,
    ))

    if verbose:
        print(f"  M: {hwi_before.M:.4f} -> {hwi_after.M:.4f} (dM={delta_M:+.4f})")
        print(f"  anomaly: {det_before.score:.3f} -> {det_after.score:.3f}")
        print(f"  severity: {severity_before} -> {severity_after}")
        print(f"  HEALED: {healed}")
        print(f"  [LEARN] experiences={mem.size}")

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
        prediction_used=prediction_used,
        predicted_M_after=predicted_M,
        prediction_error=prediction_error,
        experience_count=mem.size,
        compute_time_ms=elapsed,
    )
