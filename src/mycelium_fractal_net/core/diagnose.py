"""diagnose() — single-call full diagnostic pipeline.

Modes:
- fast: skip intervention, permissive causal (lowest latency)
- full: complete pipeline with intervention planning (default)
- streaming: yields each step as it completes (for progress tracking)
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any, Generator, Literal

from mycelium_fractal_net.analytics.morphology import compute_morphology_descriptor
from mycelium_fractal_net.core.causal_validation import validate_causal_consistency
from mycelium_fractal_net.core.detect import detect_anomaly
from mycelium_fractal_net.core.early_warning import early_warning
from mycelium_fractal_net.core.forecast import forecast_next
from mycelium_fractal_net.intervention import plan_intervention
from mycelium_fractal_net.types.diagnosis import (
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_STABLE,
    SEVERITY_WARNING,
    DiagnosisReport,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from mycelium_fractal_net.intervention.types import InterventionPlan
    from mycelium_fractal_net.types.causal import CausalValidationResult
    from mycelium_fractal_net.types.detection import AnomalyEvent
    from mycelium_fractal_net.types.ews import CriticalTransitionWarning
    from mycelium_fractal_net.types.features import MorphologyDescriptor
    from mycelium_fractal_net.types.field import FieldSequence
    from mycelium_fractal_net.types.forecast import ForecastResult

__all__ = ["diagnose", "diagnose_streaming"]


def _compute_severity(
    anomaly_label: str,
    anomaly_score: float,
    ews_score: float,
    causal_decision: str,
) -> str:
    if causal_decision == "fail":
        return SEVERITY_CRITICAL
    if anomaly_label == "anomalous" and ews_score >= 0.5:
        return SEVERITY_CRITICAL
    if anomaly_label in ("anomalous", "watch"):
        return SEVERITY_WARNING
    if ews_score >= 0.7:
        return SEVERITY_WARNING
    if ews_score >= 0.3:
        return SEVERITY_INFO
    return SEVERITY_STABLE


def _build_narrative(
    severity: str,
    anomaly_label: str,
    anomaly_score: float,
    ews_transition_type: str,
    ews_score: float,
    ews_time: float,
    causal_decision: str,
    has_plan: bool,
    plan_changes: list[dict[str, Any]] | None,
) -> str:
    lines: list[str] = []

    if anomaly_label == "nominal":
        lines.append("System is operating in nominal regime.")
    elif anomaly_label == "watch":
        lines.append(f"System is in watch state (score={anomaly_score:.2f}) — monitoring advised.")
    else:
        lines.append(f"Anomalous activity detected (score={anomaly_score:.2f}).")

    if ews_transition_type == "stable":
        lines.append("No early warning signals detected.")
    else:
        t_str = f"{ews_time:.0f} steps" if not math.isinf(ews_time) else "unknown horizon"
        lines.append(
            f"EWS signals a {ews_transition_type.replace('_', ' ')} transition "
            f"(score={ews_score:.2f}, estimated T≈{t_str})."
        )

    if causal_decision == "pass":
        lines.append("Causal validation passed.")
    elif causal_decision == "degraded":
        lines.append("Causal validation degraded — some rules raised warnings.")
    else:
        lines.append("Causal validation FAILED — conclusions may be unreliable.")

    if has_plan and plan_changes:
        non_zero = [c for c in plan_changes if abs(c["to"] - c["from"]) > 1e-9]
        if non_zero:
            change_strs = [f"{c['name']} {c['from']:.3f}→{c['to']:.3f}" for c in non_zero[:3]]
            lines.append(f"Intervention plan available: {', '.join(change_strs)}.")
    elif severity in (SEVERITY_STABLE, SEVERITY_INFO):
        lines.append("No intervention required.")

    return " ".join(lines)


def _build_report(
    seq: FieldSequence,
    descriptor: MorphologyDescriptor,
    anomaly: AnomalyEvent,
    warning_obj: CriticalTransitionWarning,
    forecast_result: ForecastResult,
    causal: CausalValidationResult,
    plan: InterventionPlan | None,
    t_start: float,
    forecast_horizon: int,
    causal_mode: str,
) -> DiagnosisReport:
    """Assemble the final DiagnosisReport from pipeline outputs."""
    severity = _compute_severity(
        anomaly_label=anomaly.label,
        anomaly_score=float(anomaly.score),
        ews_score=warning_obj.ews_score,
        causal_decision=causal.decision.value,
    )

    plan_changes: list[dict[str, Any]] | None = None
    if plan is not None and plan.has_viable_plan and plan.best_candidate is not None:
        plan_changes = [
            {"name": ch.name, "from": float(ch.current_value), "to": float(ch.proposed_value)}
            for ch in plan.best_candidate.proposed_changes
        ]

    narrative = _build_narrative(
        severity=severity,
        anomaly_label=anomaly.label,
        anomaly_score=float(anomaly.score),
        ews_transition_type=warning_obj.transition_type,
        ews_score=warning_obj.ews_score,
        ews_time=warning_obj.time_to_transition,
        causal_decision=causal.decision.value,
        has_plan=plan is not None and plan.has_viable_plan,
        plan_changes=plan_changes,
    )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    spec = seq.spec
    metadata: dict[str, object] = {
        "diagnosis_time_ms": round(elapsed_ms, 2),
        "causal_certificate": warning_obj.causal_certificate,
        "grid_size": int(spec.grid_size) if spec else 0,
        "steps": int(seq.history.shape[0]) if seq.history is not None else 0,
        "seed": int(spec.seed) if spec and spec.seed is not None else 0,
        "forecast_horizon": forecast_horizon,
        "causal_mode": causal_mode,
    }

    return DiagnosisReport(
        severity=severity,
        anomaly=anomaly,
        warning=warning_obj,
        forecast=forecast_result,
        causal=causal,
        descriptor=descriptor,
        plan=plan,
        narrative=narrative,
        metadata=metadata,
    )


def diagnose(
    seq: FieldSequence,
    *,
    mode: Literal["fast", "full"] = "full",
    forecast_horizon: int = 8,
    intervention_budget: float = 10.0,
    intervention_max_candidates: int = 16,
    allowed_levers: list[str] | None = None,
    skip_intervention: bool = False,
    causal_mode: str = "strict",
) -> DiagnosisReport:
    """Full diagnostic pipeline in one call.

    Parameters
    ----------
    seq : FieldSequence
        Output of mfn.simulate().
    mode : "fast" | "full"
        fast: skip intervention + permissive causal (low latency).
        full: complete pipeline (default).
        For streaming, use diagnose_streaming() instead.
    forecast_horizon : int
        Steps ahead for forecast.
    intervention_budget : float
        Max total cost for intervention plan.
    intervention_max_candidates : int
        How many candidates to evaluate.
    allowed_levers : list[str] | None
        Which parameters the planner can change.
    skip_intervention : bool
        Skip planning even if severity >= warning.
    causal_mode : str
        'strict' (default), 'observe', or 'permissive'.

    Returns
    -------
    DiagnosisReport
    """
    if mode == "fast":
        skip_intervention = True
        causal_mode = "permissive"

    t_start = time.perf_counter()

    descriptor = compute_morphology_descriptor(seq)
    anomaly = detect_anomaly(seq)
    warning_obj = early_warning(seq)
    forecast = forecast_next(seq, horizon=forecast_horizon)
    causal = validate_causal_consistency(
        seq,
        descriptor=descriptor,
        detection=anomaly,
        forecast=forecast,
        mode=causal_mode,
    )

    severity = _compute_severity(
        anomaly_label=anomaly.label,
        anomaly_score=float(anomaly.score),
        ews_score=warning_obj.ews_score,
        causal_decision=causal.decision.value,
    )

    plan = None
    if not skip_intervention and severity in (SEVERITY_WARNING, SEVERITY_CRITICAL):
        try:
            plan = plan_intervention(
                seq,
                target_regime="stable",
                allowed_levers=allowed_levers,
                budget=intervention_budget,
                max_candidates=intervention_max_candidates,
            )
        except Exception:
            plan = None

    return _build_report(
        seq,
        descriptor,
        anomaly,
        warning_obj,
        forecast,
        causal,
        plan,
        t_start,
        forecast_horizon,
        causal_mode,
    )


def diagnose_streaming(
    seq: FieldSequence,
    *,
    forecast_horizon: int = 8,
    intervention_budget: float = 10.0,
    intervention_max_candidates: int = 16,
    allowed_levers: list[str] | None = None,
    causal_mode: str = "strict",
) -> Generator[tuple[str, object], None, DiagnosisReport]:
    """Streaming diagnostic — yields each pipeline step as it completes.

    Yields (step_name, result) tuples as each stage finishes.
    The final return value is the complete DiagnosisReport.

    Usage::

        gen = mfn.diagnose_streaming(seq)
        try:
            while True:
                step, result = next(gen)
                print(f"  {step}: done")
        except StopIteration as e:
            report = e.value
            print(report.summary())

    Or with a for loop (discards final report)::

        for step, result in mfn.diagnose_streaming(seq):
            print(f"  {step}: done")
    """
    t_start = time.perf_counter()

    descriptor = compute_morphology_descriptor(seq)
    yield ("extract", descriptor)

    anomaly = detect_anomaly(seq)
    yield ("anomaly", anomaly)

    warning_obj = early_warning(seq)
    yield ("warning", warning_obj)

    forecast = forecast_next(seq, horizon=forecast_horizon)
    yield ("forecast", forecast)

    causal = validate_causal_consistency(
        seq,
        descriptor=descriptor,
        detection=anomaly,
        forecast=forecast,
        mode=causal_mode,
    )
    yield ("causal", causal)

    severity = _compute_severity(
        anomaly_label=anomaly.label,
        anomaly_score=float(anomaly.score),
        ews_score=warning_obj.ews_score,
        causal_decision=causal.decision.value,
    )

    plan = None
    if severity in (SEVERITY_WARNING, SEVERITY_CRITICAL):
        try:
            plan = plan_intervention(
                seq,
                target_regime="stable",
                allowed_levers=allowed_levers,
                budget=intervention_budget,
                max_candidates=intervention_max_candidates,
            )
            yield ("plan", plan)
        except Exception:
            plan = None

    return _build_report(
        seq,
        descriptor,
        anomaly,
        warning_obj,
        forecast,
        causal,
        plan,
        t_start,
        forecast_horizon,
        causal_mode,
    )
