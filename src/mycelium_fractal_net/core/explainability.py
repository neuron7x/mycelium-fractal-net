"""Decision explainability engine.

Generates human-readable reasoning chains for every pipeline decision.
Not post-hoc — computed alongside the decision itself.

Unique capability: no other scientific computing library provides
machine-readable + human-readable decision provenance as a first-class
output of every operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DecisionExplanation:
    """Complete reasoning chain for a single pipeline decision.

    Contains: what was decided, why, how confident, what alternatives
    were considered, what would change the decision, and whether
    the decision is stable under perturbation.
    """

    decision: str
    confidence: float
    reasoning: list[str]
    evidence_ranking: list[tuple[str, float]]
    margin_to_flip: float
    nearest_alternative: str
    stability: str  # "stable" | "marginal" | "unstable"
    counterfactual: str  # what would change the decision

    def __repr__(self) -> str:
        return (
            f"Explanation({self.decision}, conf={self.confidence:.2f}, "
            f"margin={self.margin_to_flip:.3f}, {self.stability})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "evidence_ranking": [
                {"feature": k, "contribution": v} for k, v in self.evidence_ranking
            ],
            "margin_to_flip": self.margin_to_flip,
            "nearest_alternative": self.nearest_alternative,
            "stability": self.stability,
            "counterfactual": self.counterfactual,
        }

    def narrate(self) -> str:
        """Human-readable narrative of the decision."""
        lines = [f"Decision: {self.decision} (confidence: {self.confidence:.0%})"]
        lines.append("")
        lines.append("Reasoning:")
        for i, reason in enumerate(self.reasoning, 1):
            lines.append(f"  {i}. {reason}")
        lines.append("")
        lines.append("Top evidence:")
        for feature, contrib in self.evidence_ranking[:5]:
            bar = "█" * max(1, int(contrib * 20))
            lines.append(f"  {feature:30s} {bar} {contrib:.3f}")
        lines.append("")
        lines.append(f"Margin to flip: {self.margin_to_flip:.3f}")
        lines.append(f"Nearest alternative: {self.nearest_alternative}")
        lines.append(f"Stability: {self.stability}")
        lines.append(f"Counterfactual: {self.counterfactual}")
        return "\n".join(lines)


@dataclass(frozen=True)
class PipelineExplanation:
    """Complete reasoning chain for the full pipeline."""

    detection: DecisionExplanation
    regime: DecisionExplanation
    comparison: DecisionExplanation | None = None
    causal_summary: str = ""

    def __repr__(self) -> str:
        parts = [f"detection={self.detection.decision}"]
        parts.append(f"regime={self.regime.decision}")
        if self.comparison:
            parts.append(f"comparison={self.comparison.decision}")
        return f"PipelineExplanation({', '.join(parts)})"

    def narrate(self) -> str:
        sections = ["═══ Detection ═══", self.detection.narrate()]
        sections.extend(["", "═══ Regime ═══", self.regime.narrate()])
        if self.comparison:
            sections.extend(["", "═══ Comparison ═══", self.comparison.narrate()])
        if self.causal_summary:
            sections.extend(["", "═══ Causal Verification ═══", self.causal_summary])
        return "\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "detection": self.detection.to_dict(),
            "regime": self.regime.to_dict(),
            "causal_summary": self.causal_summary,
        }
        if self.comparison:
            d["comparison"] = self.comparison.to_dict()
        return d


def explain_detection(event: Any) -> DecisionExplanation:
    """Generate explanation for an anomaly detection decision."""
    evidence = dict(event.evidence)
    threshold = evidence.pop("dynamic_threshold", 0.45)
    score = event.score

    # Rank evidence by contribution
    ranked = sorted(evidence.items(), key=lambda kv: abs(kv[1]), reverse=True)

    # Compute margin to flip
    if event.label == "nominal":
        watch_threshold = max(0.30, threshold - 0.18)
        margin = watch_threshold - score
        nearest_alt = "watch"
        counterfactual = f"Score would need to increase by {margin:.3f} to trigger watch"
    elif event.label == "watch":
        margin_up = threshold - score
        margin_down = score - max(0.30, threshold - 0.18)
        if margin_up < margin_down:
            margin = margin_up
            nearest_alt = "anomalous"
            counterfactual = f"Score increase of {margin:.3f} would trigger anomalous"
        else:
            margin = margin_down
            nearest_alt = "nominal"
            counterfactual = f"Score decrease of {margin:.3f} would return to nominal"
    else:  # anomalous
        margin = score - threshold
        nearest_alt = "watch"
        counterfactual = f"Score decrease of {margin:.3f} would downgrade to watch"

    stability = "stable" if margin > 0.05 else "marginal" if margin > 0.01 else "unstable"

    # Build reasoning
    reasoning = []
    reasoning.append(f"Anomaly score {score:.3f} vs dynamic threshold {threshold:.3f}")
    if event.regime and event.regime.label == "pathological_noise":
        reasoning.append("Regime is pathological_noise → forced anomalous label")
    elif event.regime and event.regime.label == "reorganized":
        reasoning.append("Regime is reorganized → forced watch label")
    else:
        reasoning.append(
            f"Score {'above' if score >= threshold else 'below'} threshold → {event.label}"
        )

    top3 = ranked[:3]
    if top3:
        reasoning.append(f"Top drivers: {', '.join(f'{k}={v:.3f}' for k, v in top3)}")

    return DecisionExplanation(
        decision=event.label,
        confidence=event.confidence,
        reasoning=reasoning,
        evidence_ranking=ranked[:10],
        margin_to_flip=abs(margin),
        nearest_alternative=nearest_alt,
        stability=stability,
        counterfactual=counterfactual,
    )


def explain_regime(event: Any) -> DecisionExplanation:
    """Generate explanation for regime classification."""
    regime = event.regime
    evidence = dict(regime.evidence)
    ranked = sorted(evidence.items(), key=lambda kv: abs(kv[1]), reverse=True)

    reasoning = [f"Regime classified as {regime.label} with score {regime.score:.3f}"]
    if regime.label == "stable":
        reasoning.append("No strong signals for transition, criticality, or reorganization")
    elif regime.label == "critical":
        reasoning.append("High criticality pressure and/or hierarchy flattening detected")
    elif regime.label == "reorganized":
        reasoning.append(
            "Structural reorganization: high complexity gain + connectivity divergence + plasticity"
        )
    elif regime.label == "pathological_noise":
        reasoning.append(
            "High observation noise without structural complexity → noise-dominated regime"
        )
    elif regime.label == "transitional":
        reasoning.append("Moderate change signals without full criticality or reorganization")

    if ranked:
        reasoning.append(f"Dominant signal: {ranked[0][0]}={ranked[0][1]:.3f}")

    return DecisionExplanation(
        decision=regime.label,
        confidence=regime.confidence,
        reasoning=reasoning,
        evidence_ranking=ranked[:10],
        margin_to_flip=max(0, 1.0 - regime.score),
        nearest_alternative="transitional" if regime.label == "stable" else "stable",
        stability="stable" if regime.confidence > 0.7 else "marginal",
        counterfactual=f"Confidence {regime.confidence:.2f} — {'high' if regime.confidence > 0.7 else 'moderate'} certainty",
    )


def explain_comparison(comp: Any) -> DecisionExplanation:
    """Generate explanation for comparison decision."""
    reasoning = [
        f"Embedding distance: {comp.distance:.6f}",
        f"Cosine similarity: {comp.cosine_similarity:.4f}",
        f"Topology: {comp.topology_label}",
    ]

    if comp.label == "near-identical":
        reasoning.append("Extremely close embeddings → functionally identical morphologies")
    elif comp.label == "similar":
        reasoning.append("High cosine similarity → same structural family")
    elif comp.label == "related":
        reasoning.append("Moderate similarity → shared structural features but divergent details")
    else:
        reasoning.append("Low similarity → fundamentally different morphological patterns")

    return DecisionExplanation(
        decision=comp.label,
        confidence=min(1.0, comp.cosine_similarity),
        reasoning=reasoning,
        evidence_ranking=[
            ("cosine_similarity", comp.cosine_similarity),
            ("distance", comp.distance),
        ],
        margin_to_flip=0.0,
        nearest_alternative="similar" if comp.label == "near-identical" else "divergent",
        stability="stable" if comp.cosine_similarity > 0.95 else "marginal",
        counterfactual=f"Distance={comp.distance:.4f}, cosine={comp.cosine_similarity:.4f}",
    )


def explain_pipeline(
    event: Any,
    comp: Any | None = None,
    causal: Any | None = None,
) -> PipelineExplanation:
    """Generate complete reasoning chain for the full pipeline."""
    det_expl = explain_detection(event)
    reg_expl = explain_regime(event)
    comp_expl = explain_comparison(comp) if comp else None

    causal_summary = ""
    if causal is not None:
        total = len(causal.rule_results)
        passed = sum(1 for r in causal.rule_results if r.passed)
        causal_summary = (
            f"{causal.decision.value}: {passed}/{total} rules passed, "
            f"{causal.error_count} errors, {causal.warning_count} warnings"
        )

    return PipelineExplanation(
        detection=det_expl,
        regime=reg_expl,
        comparison=comp_expl,
        causal_summary=causal_summary,
    )
