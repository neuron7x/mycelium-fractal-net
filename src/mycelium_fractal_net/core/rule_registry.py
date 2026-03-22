"""Global rule registry with @rule decorator.

Each rule is a function decorated with scientific metadata.
The registry provides:
- Global lookup by rule_id
- Manifest printing (human-readable specification document)
- JSON export for machine consumption

Usage:
    @rule(
        id="SIM-002",
        claim="Membrane potential cannot fall below hyperpolarization limit",
        math="V(i,j) >= V_min = -95 mV",
        ref="Hodgkin & Huxley 1952, doi:10.1113/jphysiol.1952.sp004764",
        stage="simulate",
        severity="error",
        category="numerical",
        rationale="Below -95 mV is non-physiological; indicates numerical blow-up",
    )
    def sim_002_field_lower_bound(sequence):
        fmin = float(np.min(sequence.field))
        return fmin >= FIELD_V_MIN - 1e-10, fmin, FIELD_V_MIN
"""

from __future__ import annotations

import json
from typing import Any, Callable

from mycelium_fractal_net.types.causal import (
    CausalRuleResult,
    CausalRuleSpec,
    CausalSeverity,
    ViolationCategory,
)

_REGISTRY: dict[str, "RegisteredRule"] = {}


class RegisteredRule:
    """A rule function with attached scientific specification."""

    __slots__ = ("fn", "id", "spec", "stage", "severity", "category")

    def __init__(
        self,
        fn: Callable[..., tuple[bool, Any, Any] | tuple[bool, Any] | bool],
        rule_id: str,
        spec: CausalRuleSpec,
        stage: str,
        severity: CausalSeverity,
        category: ViolationCategory,
    ) -> None:
        self.fn = fn
        self.id = rule_id
        self.spec = spec
        self.stage = stage
        self.severity = severity
        self.category = category

    def evaluate(self, *args: Any, **kwargs: Any) -> CausalRuleResult:
        """Execute the rule and return a typed result with spec attached."""
        result = self.fn(*args, **kwargs)
        if isinstance(result, bool):
            passed, observed, expected = result, None, None
        elif len(result) == 2:
            passed, observed = result
            expected = None
        else:
            passed, observed, expected = result

        return CausalRuleResult(
            rule_id=self.id,
            stage=self.stage,
            category=self.category,
            severity=self.severity,
            passed=passed,
            message=self.spec.claim,
            spec=self.spec,
            observed=observed,
            expected=expected,
        )


def rule(
    *,
    id: str,
    claim: str,
    math: str = "",
    ref: str = "",
    stage: str,
    severity: str,
    category: str,
    rationale: str = "",
    falsifiable_by: str = "",
) -> Callable:
    """Decorator that registers a causal rule with scientific metadata."""
    sev = CausalSeverity(severity)
    cat = ViolationCategory(category)
    spec = CausalRuleSpec(
        claim=claim,
        math=math,
        reference=ref,
        falsifiable_by=falsifiable_by,
        rationale=rationale,
    )

    def decorator(fn: Callable) -> RegisteredRule:
        registered = RegisteredRule(fn, id, spec, stage, sev, cat)
        _REGISTRY[id] = registered
        return registered

    return decorator


def get_registry() -> dict[str, RegisteredRule]:
    """Return the global rule registry."""
    return dict(_REGISTRY)


def get_rule(rule_id: str) -> RegisteredRule:
    """Lookup a single rule by ID."""
    return _REGISTRY[rule_id]


def manifest_dict() -> dict[str, Any]:
    """Export the full rule manifest as a dict."""
    rules = {}
    for rid, r in sorted(_REGISTRY.items()):
        rules[rid] = {
            "stage": r.stage,
            "severity": r.severity.value,
            "category": r.category.value,
            **r.spec.to_dict(),
        }
    return {
        "schema": "mfn-causal-rule-manifest-v1",
        "total_rules": len(rules),
        "rules": rules,
    }


def print_manifest() -> None:
    """Print the living specification document to stdout."""
    reg = sorted(_REGISTRY.items())
    current_stage = ""
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  MFN Causal Rule Manifest — Living Specification Document  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  {len(reg)} rules registered\n")

    for rid, r in reg:
        if r.stage != current_stage:
            current_stage = r.stage
            print(f"  ── {current_stage.upper()} {'─' * (50 - len(current_stage))}")
            print()

        sev_tag = {"fatal": "FATAL", "error": "ERROR", "warn": " WARN", "info": " INFO"}
        print(f"  [{rid}] {sev_tag.get(r.severity.value, r.severity.value)}")
        print(f"    Claim:  {r.spec.claim}")
        if r.spec.math:
            print(f"    Math:   {r.spec.math}")
        if r.spec.reference:
            print(f"    Ref:    {r.spec.reference}")
        if r.spec.falsifiable_by:
            print(f"    Falsif: {r.spec.falsifiable_by}")
        if r.spec.rationale:
            print(f"    Why:    {r.spec.rationale}")
        print()
