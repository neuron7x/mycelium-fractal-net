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
