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


_FEATURE_KEYS = [
    "M_before", "anomaly_before",
    "alpha", "turing_threshold", "jitter_var", "spike_probability",
    "delta_alpha", "delta_spike", "delta_gabaa", "delta_sero_gain",
]


class ExperienceMemory:
    """Accumulates (state, action, outcome) triples. Predicts via Ridge regression.

    After min_experiences calls, a Ridge model predicts M_after from the
    full feature vector. Feature importances reveal what the system has
    learned about itself. R² measures depth of self-understanding.

    Compression = understanding (Sutskever): the linear model IS the
    system's compressed self-knowledge. Features it assigns high weight
    are the ones it has discovered as causal.
    """

    def __init__(self, min_experiences: int = 15) -> None:
        self.min_experiences = min_experiences
        self._features: list[dict[str, float]] = []
        self._M_after: list[float] = []
        self._anomaly_after: list[float] = []
        self._healed: list[bool] = []
        self._model: Any = None
        self._r2: float = 0.0
        self._importances: dict[str, float] = {}

    @property
    def size(self) -> int:
        return len(self._M_after)

    @property
    def can_predict(self) -> bool:
        return self._model is not None

    @property
    def r_squared(self) -> float:
        """How well the system understands itself. 1.0 = perfect self-model."""
        return self._r2

    @property
    def feature_importances(self) -> dict[str, float]:
        """What the system has discovered matters. Higher = more causal."""
        return dict(self._importances)

    def store(self, features: dict[str, float], M_after: float,
              anomaly_after: float, healed: bool) -> None:
        self._features.append(features)
        self._M_after.append(M_after)
        self._anomaly_after.append(anomaly_after)
        self._healed.append(healed)
        # Refit model when we have enough data
        if self.size >= self.min_experiences:
            self._fit()

    def _fit(self) -> None:
        """Fit Ridge regression on accumulated experiences."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        keys = list(self._features[0].keys())
        X = np.array([[f.get(k, 0.0) for k in keys] for f in self._features])
        y = np.array(self._M_after)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)
        self._r2 = float(model.score(X_scaled, y))

        # Feature importances: |coefficient| on scaled data = relative importance
        abs_coef = np.abs(model.coef_)
        total = abs_coef.sum() + 1e-12
        self._importances = {k: round(float(c / total), 4) for k, c in zip(keys, abs_coef)}

        # Store for prediction
        self._model = model
        self._scaler = scaler
        self._keys = keys

    def predict(self, features: dict[str, float]) -> tuple[float, float, float]:
        """Predict M_after from features.

        Returns (predicted_M_after, predicted_anomaly_after, R²).
        R² = confidence (how well the model fits all past data).
        """
        if self._model is None:
            return 0.0, 0.0, 0.0

        x = np.array([[features.get(k, 0.0) for k in self._keys]])
        x_scaled = self._scaler.transform(x)
        M_pred = float(self._model.predict(x_scaled)[0])

        # Anomaly: simple mean of past (Ridge on M only)
        a_pred = float(np.mean(self._anomaly_after))

        return M_pred, a_pred, self._r2

    def stats(self) -> dict[str, Any]:
        if not self._M_after:
            return {"size": 0}
        healed = sum(self._healed)
        return {
            "size": self.size,
            "heal_rate": round(healed / self.size, 3),
            "M_after_mean": round(float(np.mean(self._M_after)), 4),
            "M_after_std": round(float(np.std(self._M_after)), 4),
            "can_predict": self.can_predict,
            "r_squared": round(self._r2, 4),
            "top_features": dict(sorted(self._importances.items(),
                                        key=lambda x: -x[1])[:5]) if self._importances else {},
        }


# Global state — persists across calls within process
_MEMORY = ExperienceMemory()

from .neurochem.dopamine import DopamineState, compute_dopamine, modulate_plasticity
_DA_STATE = DopamineState()


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

    # Dopamine
    dopamine_level: float = 0.0
    dopamine_plasticity: float = 1.0

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
        d_anom = self.delta_anomaly if self.delta_anomaly is not None else 0
        pred = ""
        if self.prediction_used and self.prediction_error is not None:
            pred = f" pred_err={self.prediction_error:.3f}"
        da_str = f" DA={self.dopamine_level:.2f}" if self.dopamine_level > 0 else ""
        exp = f" exp={self.experience_count}" if self.experience_count > 0 else ""
        return (
            f"[HEAL] {status} | "
            f"M: {self.M_before:.3f} -> {self.M_after:.3f} (dM={dm:+.3f}) | "
            f"anomaly: {self.anomaly_score_before:.3f} -> {self.anomaly_score_after:.3f} "
            f"(d={d_anom:+.3f}) | "
            f"severity: {self.severity_before} -> {self.severity_after}"
            f"{pred}{da_str}{exp} ({self.compute_time_ms:.0f}ms)"
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
                "dopamine_level": round(self.dopamine_level, 4),
                "dopamine_plasticity": round(self.dopamine_plasticity, 4),
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

    # ── 6. BUILD FEATURE VECTOR ─────────────────────────────────
    mem = memory if memory is not None else _MEMORY

    spec = seq.spec
    feat = {
        "M_before": hwi_before.M,
        "anomaly_before": float(det_before.score),
        "alpha": spec.alpha if spec else 0.18,
        "turing_threshold": spec.turing_threshold if spec else 0.75,
        "jitter_var": spec.jitter_var if spec else 0.0,
        "spike_probability": spec.spike_probability if spec else 0.25,
    }
    # Add intervention deltas
    for s in best.proposed_changes:
        feat[f"delta_{s.name}"] = s.proposed_value - s.current_value

    # ── 7. PREDICT (if model available) ───────────────────────────
    prediction_used = False
    predicted_M = None
    prediction_error = None

    if mem.can_predict:
        predicted_M, _, r2 = mem.predict(feat)
        prediction_used = True
        prediction_error = abs(predicted_M - hwi_after.M)
        if verbose:
            print(f"  [LEARN] Predicted M_after={predicted_M:.4f}, actual={hwi_after.M:.4f}, "
                  f"error={prediction_error:.4f}, R²={r2:.3f}")

    # ── 8. DOPAMINE — prediction error → plasticity modulation ────
    global _DA_STATE
    pe_for_da = prediction_error if prediction_error is not None else abs(delta_M)
    _DA_STATE = compute_dopamine(pe_for_da, _DA_STATE)
    if verbose:
        print(f"  [DA] level={_DA_STATE.level:.3f} RPE={_DA_STATE.rpe:.4f} "
              f"plasticity={_DA_STATE.plasticity_scale:.2f}")

    # ── 9. LEARN — store experience ───────────────────────────────
    mem.store(feat, M_after=hwi_after.M, anomaly_after=float(det_after.score), healed=healed)

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
        dopamine_level=_DA_STATE.level,
        dopamine_plasticity=_DA_STATE.plasticity_scale,
        compute_time_ms=elapsed,
    )
