"""DiagnosisReport — unified output of mfn.diagnose()."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mycelium_fractal_net.intervention.types import InterventionPlan
    from mycelium_fractal_net.types.causal import CausalValidationResult
    from mycelium_fractal_net.types.detection import AnomalyEvent
    from mycelium_fractal_net.types.ews import CriticalTransitionWarning
    from mycelium_fractal_net.types.features import MorphologyDescriptor
    from mycelium_fractal_net.types.forecast import ForecastResult

SEVERITY_CRITICAL = "critical"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"
SEVERITY_STABLE = "stable"


@dataclass(frozen=True)
class DiagnosisReport:
    """Unified diagnostic output from mfn.diagnose().

    Combines: anomaly detection + EWS + causal validation + intervention plan.
    """

    severity: str
    anomaly: AnomalyEvent
    warning: CriticalTransitionWarning
    forecast: ForecastResult
    causal: CausalValidationResult
    descriptor: MorphologyDescriptor
    plan: InterventionPlan | None
    narrative: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_ok(self) -> bool:
        """True if severity is stable or info."""
        return self.severity in (SEVERITY_STABLE, SEVERITY_INFO)

    def needs_intervention(self) -> bool:
        """True if plan exists and has viable candidates."""
        return self.plan is not None and self.plan.has_viable_plan

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict (JSON-safe)."""
        result: dict[str, Any] = {
            "severity": self.severity,
            "narrative": self.narrative,
            "metadata": dict(self.metadata),
            "anomaly": {
                "label": self.anomaly.label,
                "score": float(self.anomaly.score),
                "confidence": float(self.anomaly.confidence),
                "regime": self.anomaly.regime.label if self.anomaly.regime else "none",
            },
            "warning": self.warning.to_dict(),
            "causal": {
                "decision": self.causal.decision.value,
                "error_count": sum(
                    1
                    for r in self.causal.rule_results
                    if not r.passed and r.severity.value in ("error", "fatal")
                ),
                "warning_count": sum(
                    1
                    for r in self.causal.rule_results
                    if not r.passed and r.severity.value == "warn"
                ),
            },
            "plan": None,
        }
        if (
            self.plan is not None
            and self.plan.has_viable_plan
            and self.plan.best_candidate is not None
        ):
            bc = self.plan.best_candidate
            result["plan"] = {
                "viable": True,
                "composite_score": float(bc.composite_score),
                "causal_decision": bc.causal_decision,
                "changes": [
                    {
                        "name": ch.name,
                        "from": float(ch.current_value),
                        "to": float(ch.proposed_value),
                        "cost": float(ch.cost),
                    }
                    for ch in bc.proposed_changes
                    if abs(ch.proposed_value - ch.current_value) > 1e-9
                ],
            }
        return result

    def summary(self) -> str:
        """One-line status string."""
        ews = self.warning
        ev = self.anomaly
        causal_ok = self.causal.decision.value
        plan_str = ""
        if (
            self.plan is not None
            and self.plan.has_viable_plan
            and self.plan.best_candidate is not None
        ):
            n = len(
                [
                    ch
                    for ch in self.plan.best_candidate.proposed_changes
                    if abs(ch.proposed_value - ch.current_value) > 1e-9
                ]
            )
            if n > 0:
                plan_str = f" plan={n}changes"
        return (
            f"[DIAGNOSIS:{self.severity.upper()}] "
            f"anomaly={ev.label}({ev.score:.2f}) "
            f"ews={ews.transition_type}({ews.ews_score:.2f}) "
            f"causal={causal_ok}{plan_str}"
        )

    def diff(self, other: DiagnosisReport) -> DiagnosisDiff:
        """Compare this report with another and return what changed."""
        return DiagnosisDiff(
            severity_changed=self.severity != other.severity,
            severity_before=self.severity,
            severity_after=other.severity,
            anomaly_label_changed=self.anomaly.label != other.anomaly.label,
            anomaly_label_before=self.anomaly.label,
            anomaly_label_after=other.anomaly.label,
            anomaly_score_delta=round(other.anomaly.score - self.anomaly.score, 6),
            ews_score_delta=round(other.warning.ews_score - self.warning.ews_score, 6),
            ews_type_changed=self.warning.transition_type != other.warning.transition_type,
            ews_type_before=self.warning.transition_type,
            ews_type_after=other.warning.transition_type,
            causal_changed=self.causal.decision != other.causal.decision,
            causal_before=self.causal.decision.value,
            causal_after=other.causal.decision.value,
        )


@dataclass(frozen=True)
class DiagnosisDiff:
    """Difference between two DiagnosisReport instances."""

    severity_changed: bool
    severity_before: str
    severity_after: str
    anomaly_label_changed: bool
    anomaly_label_before: str
    anomaly_label_after: str
    anomaly_score_delta: float
    ews_score_delta: float
    ews_type_changed: bool
    ews_type_before: str
    ews_type_after: str
    causal_changed: bool
    causal_before: str
    causal_after: str

    @property
    def has_changes(self) -> bool:
        return (
            self.severity_changed
            or self.anomaly_label_changed
            or self.ews_type_changed
            or self.causal_changed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": {
                "changed": self.severity_changed,
                "before": self.severity_before,
                "after": self.severity_after,
            },
            "anomaly_label": {
                "changed": self.anomaly_label_changed,
                "before": self.anomaly_label_before,
                "after": self.anomaly_label_after,
            },
            "anomaly_score_delta": self.anomaly_score_delta,
            "ews_score_delta": self.ews_score_delta,
            "ews_type": {
                "changed": self.ews_type_changed,
                "before": self.ews_type_before,
                "after": self.ews_type_after,
            },
            "causal": {
                "changed": self.causal_changed,
                "before": self.causal_before,
                "after": self.causal_after,
            },
            "has_changes": self.has_changes,
        }

    def summary(self) -> str:
        parts = []
        if self.severity_changed:
            parts.append(f"severity: {self.severity_before}→{self.severity_after}")
        if self.anomaly_label_changed:
            parts.append(f"anomaly: {self.anomaly_label_before}→{self.anomaly_label_after}")
        if abs(self.ews_score_delta) > 0.01:
            parts.append(f"ews: {self.ews_score_delta:+.3f}")
        if self.causal_changed:
            parts.append(f"causal: {self.causal_before}→{self.causal_after}")
        return ", ".join(parts) if parts else "no changes"
