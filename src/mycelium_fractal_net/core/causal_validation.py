"""Causal Validation Gate — причинно-наслідкова верифікація pipeline.

Перевіряє що кожен висновок системи випливає з даних, інваріантів
та правил, а не лише проходить типи та локальні евристики.

Не підміняє стратегію — відсікає хибні переходи, суперечності,
нестійкі евристики, приховане накопичення помилок.

Два режими:
- observe-only: фіксує порушення, не блокує
- strict: блокує при error/fatal

Schema: mfn-causal-validation-v1
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from mycelium_fractal_net.core.reaction_diffusion_config import (
    FIELD_V_MAX,
    FIELD_V_MIN,
    MAX_STABLE_DIFFUSION,
)
from mycelium_fractal_net.types.causal import (
    CausalDecision,
    CausalRuleResult,
    CausalRuleSpec,
    CausalSeverity,
    CausalValidationResult,
    ViolationCategory,
)


def _rule(
    rule_id: str,
    stage: str,
    category: ViolationCategory,
    severity: CausalSeverity,
    passed: bool,
    message: str,
    observed: Any = None,
    expected: Any = None,
    **evidence: Any,
) -> CausalRuleResult:
    return CausalRuleResult(
        rule_id=rule_id,
        stage=stage,
        category=category,
        severity=severity,
        passed=passed,
        message=message,
        spec=SPECS.get(rule_id),
        observed=observed,
        expected=expected,
        evidence=evidence,
    )


# ═══════════════════════════════════════════════════════════
# Rule specifications — each rule as a scientific claim
# ═══════════════════════════════════════════════════════════

SPECS = {
    "SIM-001": CausalRuleSpec(
        claim="Field values must be real numbers representable in IEEE 754",
        math="∀ i,j: u(i,j) ∈ ℝ, |u(i,j)| < ∞",
        reference="IEEE 754-2019",
        falsifiable_by="Any NaN or Inf in the field array",
        rationale="NaN/Inf propagates silently through all downstream computations, producing meaningless results",
    ),
    "SIM-002": CausalRuleSpec(
        claim="Membrane potential cannot fall below hyperpolarization limit",
        math="V(i,j) ≥ V_min = -95 mV",
        reference="Hodgkin & Huxley 1952, doi:10.1113/jphysiol.1952.sp004764",
        falsifiable_by="Field value below -95 mV after clamping",
        rationale="Below -95 mV is non-physiological; indicates numerical blow-up or missing clamp",
    ),
    "SIM-003": CausalRuleSpec(
        claim="Membrane potential cannot exceed action potential peak",
        math="V(i,j) ≤ V_max = +40 mV",
        reference="Hodgkin & Huxley 1952, doi:10.1113/jphysiol.1952.sp004764",
        falsifiable_by="Field value above +40 mV after clamping",
        rationale="Above +40 mV exceeds Na+ reversal potential; indicates numerical instability",
    ),
    "SIM-004": CausalRuleSpec(
        claim="History spatial dimensions must match final field",
        math="history.shape[1:] == field.shape",
        falsifiable_by="Shape mismatch between history frames and field",
        rationale="Mismatched shapes corrupt temporal feature extraction and forecasting",
    ),
    "SIM-005": CausalRuleSpec(
        claim="History must contain only finite values across all timesteps",
        math="∀ t,i,j: history(t,i,j) ∈ ℝ",
        falsifiable_by="NaN or Inf in any history frame",
        rationale="Corrupted history produces invalid temporal features and misleading Lyapunov exponents",
    ),
    "SIM-006": CausalRuleSpec(
        claim="Last history frame should approximate the final field state",
        math="‖history[-1] - field‖ < ε",
        falsifiable_by="Large discrepancy between last history frame and field",
        rationale="Divergence suggests observation noise or post-processing altered the field after recording",
    ),
    "SIM-007": CausalRuleSpec(
        claim="Diffusion coefficient must satisfy CFL stability condition",
        math="α ≤ 1/(4·dt/dx²) = 0.25 for dt=dx=1",
        reference="Courant, Friedrichs & Lewy 1928, doi:10.1007/BF01448839",
        falsifiable_by="Alpha exceeding 0.25 without substep splitting",
        rationale="CFL violation causes exponential blow-up in explicit Euler PDE integration",
    ),
    "SIM-008": CausalRuleSpec(
        claim="Receptor occupancy fractions must sum to unity (conservation of mass)",
        math="p_resting + p_active + p_desensitized = 1.0",
        reference="Colquhoun & Hawkes 1977, doi:10.1098/rspb.1977.0085",
        falsifiable_by="Occupancy sum deviating from 1.0 by more than 1e-4",
        rationale="Mass conservation violation means receptors are being created or destroyed — non-physical",
    ),
    "SIM-009": CausalRuleSpec(
        claim="Effective inhibition is non-negative (shunting conductance)",
        math="g_inh ≥ 0",
        reference="Koch 1999, Biophysics of Computation, Ch. 2",
        falsifiable_by="Negative effective inhibition value",
        rationale="Negative shunting conductance is non-physical and would amplify instead of inhibit",
    ),
    "SIM-010": CausalRuleSpec(
        claim="Simulation provenance requires attached specification for reproducibility",
        rationale="Without spec, the simulation cannot be reproduced or audited",
    ),
    "EXT-001": CausalRuleSpec(
        claim="Morphology embedding must be a finite non-empty real vector",
        math="e ∈ ℝⁿ, n > 0, ‖e‖ < ∞",
        falsifiable_by="Empty embedding or NaN/Inf in embedding components",
        rationale="Downstream distance/cosine computations produce undefined results on invalid embeddings",
    ),
    "EXT-002": CausalRuleSpec(
        claim="Descriptor must carry version for schema evolution",
        rationale="Unversioned descriptors cannot be compared across software releases",
    ),
    "EXT-003": CausalRuleSpec(
        claim="Instability index must be consistent with field coefficient of variation",
        math="instability_index ≈ σ(field) / |μ(field)|",
        falsifiable_by="Instability index diverging from direct field CV computation",
        rationale="Inconsistency indicates a bug in feature extraction or stale cached descriptor",
    ),
    "EXT-004": CausalRuleSpec(
        claim="Stability feature group must contain all required keys",
        rationale="Missing stability keys cause KeyError in detection scoring functions",
    ),
    "EXT-005": CausalRuleSpec(
        claim="Complexity feature group must contain all required keys",
        rationale="Missing complexity keys corrupt regime classification scoring",
    ),
    "EXT-006": CausalRuleSpec(
        claim="Connectivity feature group must contain all required keys",
        rationale="Connectivity features drive topology drift analysis in comparison",
    ),
    "DET-001": CausalRuleSpec(
        claim="Anomaly score is a probability-like quantity bounded to [0, 1]",
        math="0 ≤ score ≤ 1",
        falsifiable_by="Score outside unit interval",
        rationale="Unbounded scores break threshold comparisons and confidence calibration",
    ),
    "DET-002": CausalRuleSpec(
        claim="Anomaly label must be from the closed vocabulary",
        math="label ∈ {nominal, watch, anomalous}",
        falsifiable_by="Label not in allowed set",
        rationale="Unknown labels cannot be processed by downstream consumers",
    ),
    "DET-003": CausalRuleSpec(
        claim="Regime label must be from the closed vocabulary",
        math="regime ∈ {stable, transitional, critical, reorganized, pathological_noise}",
        falsifiable_by="Regime label not in allowed set",
    ),
    "DET-004": CausalRuleSpec(
        claim="Confidence is a probability bounded to [0, 1]",
        math="0 ≤ confidence ≤ 1",
        falsifiable_by="Confidence outside unit interval",
    ),
    "DET-005": CausalRuleSpec(
        claim="Contributing features must reference actual evidence keys",
        falsifiable_by="Feature name in contributing list not found in evidence dict",
        rationale="Phantom features mislead the user about what drove the decision",
    ),
    "DET-006": CausalRuleSpec(
        claim="Pathological noise regime requires observation noise evidence",
        math="regime=pathological_noise → noise_gain ≥ 0.1",
        falsifiable_by="Pathological noise classification without elevated noise signal",
        rationale="Classifying as noise-dominated without noise evidence is a causal error",
    ),
    "DET-007": CausalRuleSpec(
        claim="Reorganized regime requires plasticity evidence",
        math="regime=reorganized → plasticity_index ≥ 0.05",
        reference="Turrigiano 2008, doi:10.1016/j.cell.2008.01.001 (homeostatic plasticity)",
        falsifiable_by="Reorganization classification without plasticity signal",
        rationale="Structural reorganization without plasticity is indistinguishable from noise",
    ),
    "DET-008": CausalRuleSpec(
        claim="Watch label indicates proximity to decision boundary",
        math="|score - threshold| < margin_max",
        falsifiable_by="Watch label far from threshold boundary",
        rationale="Watch is a transitional state; if far from boundary, label should be nominal or anomalous",
    ),
    "FOR-001": CausalRuleSpec(
        claim="Forecast horizon must be at least one step",
        math="horizon ≥ 1",
        falsifiable_by="Zero or negative horizon",
    ),
    "FOR-002": CausalRuleSpec(
        claim="All predicted states must be finite real-valued arrays",
        math="∀ t: predicted(t) ∈ ℝ^(N×N)",
        falsifiable_by="NaN or Inf in any predicted frame",
        rationale="Non-finite predictions invalidate structural error and trajectory analysis",
    ),
    "FOR-003": CausalRuleSpec(
        claim="Predicted states should remain within biophysical bounds",
        math="V_min ≤ predicted(t,i,j) ≤ V_max",
        reference="Hodgkin & Huxley 1952",
        falsifiable_by="Predicted field value outside [-95, +40] mV",
        rationale="Forecast beyond biophysical bounds indicates extrapolation failure",
    ),
    "FOR-004": CausalRuleSpec(
        claim="Forecast must include uncertainty quantification",
        reference="Lakshminarayanan et al. 2017, doi:10.48550/arXiv.1612.01474",
        falsifiable_by="Empty uncertainty envelope",
        rationale="Point forecasts without uncertainty are scientifically incomplete",
    ),
    "FOR-005": CausalRuleSpec(
        claim="Benchmark metrics must include structural error and damping",
        rationale="Without these, forecast quality cannot be assessed or compared across runs",
    ),
    "FOR-006": CausalRuleSpec(
        claim="Forecast structural error should be bounded (no wild divergence)",
        math="structural_error ≤ 1.0",
        falsifiable_by="Structural error exceeding 1.0",
        rationale="Error > 1.0 means the forecast diverged more than the signal itself",
    ),
    "FOR-007": CausalRuleSpec(
        claim="Adaptive damping must be in the stable regime",
        math="0.80 ≤ damping ≤ 0.95",
        falsifiable_by="Damping outside [0.80, 0.95]",
        rationale="Damping < 0.80 causes excessive smoothing; > 0.95 causes oscillation",
    ),
    "CMP-001": CausalRuleSpec(
        claim="Embedding distance is a metric (non-negative)",
        math="d(a, b) ≥ 0, d(a, a) = 0",
        falsifiable_by="Negative distance value",
        rationale="Negative distance violates metric space axioms and corrupts similarity classification",
    ),
    "CMP-002": CausalRuleSpec(
        claim="Cosine similarity is bounded by definition",
        math="-1 ≤ cos(a, b) ≤ 1",
        falsifiable_by="Cosine outside [-1, 1]",
        rationale="Values outside bounds indicate numerical error in dot product or normalization",
    ),
    "CMP-003": CausalRuleSpec(
        claim="Comparison label must be from the closed vocabulary",
        math="label ∈ {near-identical, similar, related, divergent}",
        falsifiable_by="Unknown comparison label",
    ),
    "CMP-004": CausalRuleSpec(
        claim="Near-identical label requires low embedding distance",
        math="label=near-identical → d < 0.5",
        falsifiable_by="Near-identical classification with distance ≥ 0.5",
        rationale="Labeling distant embeddings as identical misleads downstream consumers",
    ),
    "CMP-005": CausalRuleSpec(
        claim="Divergent label requires low cosine similarity",
        math="label=divergent → cos < 0.95",
        falsifiable_by="Divergent classification with cosine ≥ 0.95",
    ),
    "CMP-006": CausalRuleSpec(
        claim="Topology and reorganization labels must be consistent mapping",
        math="nominal→stable, flattened-hierarchy→transitional, pathological-drift→pathological_noise, reorganized→reorganized",
        falsifiable_by="Topology label mapping to wrong reorganization label",
        rationale="Inconsistent labels produce contradictory interpretation in reports",
    ),
    "XST-001": CausalRuleSpec(
        claim="Stable regime should not produce anomalous detection",
        math="regime=stable ∧ label=anomalous → contradiction",
        falsifiable_by="Stable regime coexisting with anomalous label",
        rationale="If the regime is stable, the system is in a known good state — anomalous label is a false positive",
    ),
    "XST-002": CausalRuleSpec(
        claim="Disabled neuromodulation implies zero plasticity evidence",
        math="neuromod=disabled → plasticity_index < ε",
        falsifiable_by="Non-zero plasticity when neuromodulation is off",
        rationale="Plasticity without neuromodulation is a data integrity error",
    ),
    "XST-003": CausalRuleSpec(
        claim="Noise profile with reorganized regime requires structural evidence",
        math="profile=noise ∧ regime=reorganized → connectivity ≥ 0.10",
        falsifiable_by="Noise profile producing reorganized regime without connectivity signal",
        rationale="Reorganization claim from noise profile without structural evidence is a causal error",
    ),
    "PTB-001": CausalRuleSpec(
        claim="Detection label must be stable under infinitesimal perturbation",
        math="‖δ‖ = 10⁻⁶ → label(x) = label(x + δ)",
        reference="Goodfellow et al. 2015, doi:10.48550/arXiv.1412.6572 (adversarial robustness)",
        falsifiable_by="Label flip under 1e-6 additive noise",
        rationale="Label instability under negligible noise indicates the decision is on a knife-edge",
    ),
    "PTB-002": CausalRuleSpec(
        claim="Regime label must be stable under infinitesimal perturbation",
        math="‖δ‖ = 10⁻⁶ → regime(x) = regime(x + δ)",
        reference="Goodfellow et al. 2015",
        falsifiable_by="Regime flip under 1e-6 additive noise",
        rationale="Regime instability under negligible noise means the classification is not robust",
    ),
}


# ═══════════════════════════════════════════════════════════
# Stage: simulate
# ═══════════════════════════════════════════════════════════


def _check_simulation(sequence: Any) -> list[CausalRuleResult]:
    results: list[CausalRuleResult] = []
    S = "simulate"

    # SIM-001: Field finite
    is_finite = bool(np.isfinite(sequence.field).all())
    results.append(
        _rule(
            "SIM-001",
            S,
            ViolationCategory.NUMERICAL,
            CausalSeverity.FATAL,
            is_finite,
            "Field must be finite (no NaN/Inf)",
            observed=not is_finite,
            expected=True,
        )
    )

    # SIM-002: Field lower bound
    fmin = float(np.min(sequence.field))
    ok = fmin >= FIELD_V_MIN - 1e-10
    results.append(
        _rule(
            "SIM-002",
            S,
            ViolationCategory.NUMERICAL,
            CausalSeverity.ERROR,
            ok,
            f"Field min {fmin*1000:.2f} mV >= {FIELD_V_MIN*1000:.1f} mV",
            observed=fmin,
            expected=FIELD_V_MIN,
        )
    )

    # SIM-003: Field upper bound
    fmax = float(np.max(sequence.field))
    ok = fmax <= FIELD_V_MAX + 1e-10
    results.append(
        _rule(
            "SIM-003",
            S,
            ViolationCategory.NUMERICAL,
            CausalSeverity.ERROR,
            ok,
            f"Field max {fmax*1000:.2f} mV <= {FIELD_V_MAX*1000:.1f} mV",
            observed=fmax,
            expected=FIELD_V_MAX,
        )
    )

    # SIM-004: History shape consistency
    if sequence.history is not None:
        shape_ok = sequence.history.shape[1:] == sequence.field.shape
        results.append(
            _rule(
                "SIM-004",
                S,
                ViolationCategory.STRUCTURAL,
                CausalSeverity.FATAL,
                shape_ok,
                "History spatial shape must match field shape",
                observed=str(sequence.history.shape[1:]),
                expected=str(sequence.field.shape),
            )
        )
        # SIM-005: History finite
        hist_finite = bool(np.isfinite(sequence.history).all())
        results.append(
            _rule(
                "SIM-005",
                S,
                ViolationCategory.NUMERICAL,
                CausalSeverity.FATAL,
                hist_finite,
                "History must be finite",
                observed=not hist_finite,
            )
        )
        # SIM-006: Last history frame matches field
        last_frame_match = bool(np.allclose(sequence.history[-1], sequence.field, atol=0.01))
        results.append(
            _rule(
                "SIM-006",
                S,
                ViolationCategory.STRUCTURAL,
                CausalSeverity.WARN,
                last_frame_match,
                "Last history frame should approximate final field",
            )
        )

    # SIM-007: CFL stability
    if sequence.spec is not None:
        alpha_ok = sequence.spec.alpha <= MAX_STABLE_DIFFUSION
        results.append(
            _rule(
                "SIM-007",
                S,
                ViolationCategory.NUMERICAL,
                CausalSeverity.ERROR,
                alpha_ok,
                f"Alpha {sequence.spec.alpha} <= CFL limit {MAX_STABLE_DIFFUSION}",
                observed=sequence.spec.alpha,
                expected=MAX_STABLE_DIFFUSION,
            )
        )

    # SIM-008: Occupancy conservation
    ns = getattr(sequence, "neuromodulation_state", None)
    if ns is not None:
        mass = ns.occupancy_resting + ns.occupancy_active + ns.occupancy_desensitized
        mass_err = abs(mass - 1.0)
        ok = mass_err < 1e-4
        results.append(
            _rule(
                "SIM-008",
                S,
                ViolationCategory.NUMERICAL,
                CausalSeverity.ERROR,
                ok,
                f"Occupancy sum {mass:.6f} ≈ 1.0",
                observed=mass,
                expected=1.0,
            )
        )
        # SIM-009: Inhibition non-negative
        ok = ns.effective_inhibition >= 0
        results.append(
            _rule(
                "SIM-009",
                S,
                ViolationCategory.NUMERICAL,
                CausalSeverity.ERROR,
                ok,
                "Effective inhibition >= 0",
                observed=ns.effective_inhibition,
                expected=0.0,
            )
        )

    # SIM-010: Spec presence
    has_spec = sequence.spec is not None
    results.append(
        _rule(
            "SIM-010",
            S,
            ViolationCategory.PROVENANCE,
            CausalSeverity.WARN,
            has_spec,
            "SimulationSpec should be attached for traceability",
        )
    )

    return results


# ═══════════════════════════════════════════════════════════
# Stage: extract
# ═══════════════════════════════════════════════════════════


def _check_extraction(descriptor: Any, sequence: Any) -> list[CausalRuleResult]:
    results: list[CausalRuleResult] = []
    S = "extract"

    # EXT-001: Embedding finite
    emb = np.asarray(descriptor.embedding)
    ok = len(emb) > 0 and bool(np.isfinite(emb).all())
    results.append(
        _rule(
            "EXT-001",
            S,
            ViolationCategory.NUMERICAL,
            CausalSeverity.FATAL,
            ok,
            "Embedding must be non-empty and finite",
            observed=len(emb),
        )
    )

    # EXT-002: Version present
    results.append(
        _rule(
            "EXT-002",
            S,
            ViolationCategory.PROVENANCE,
            CausalSeverity.ERROR,
            bool(descriptor.version),
            "Descriptor version must be set",
            observed=descriptor.version,
        )
    )

    # EXT-003: Instability index causal consistency
    field_cv = float(np.std(sequence.field) / (abs(np.mean(sequence.field)) + 1e-12))
    desc_ii = descriptor.stability.get("instability_index", 0.0)
    drift = abs(field_cv - desc_ii)
    results.append(
        _rule(
            "EXT-003",
            S,
            ViolationCategory.CAUSAL,
            CausalSeverity.WARN,
            drift < 0.01,
            f"Instability index {desc_ii:.4f} ≈ field CV {field_cv:.4f}",
            observed=drift,
            expected=0.01,
        )
    )

    # EXT-004: Required stability keys
    required = {"instability_index", "near_transition_score", "collapse_risk_score"}
    missing = required - set(descriptor.stability.keys())
    results.append(
        _rule(
            "EXT-004",
            S,
            ViolationCategory.CONTRACT,
            CausalSeverity.ERROR,
            len(missing) == 0,
            "Stability keys complete",
            observed=sorted(missing) if missing else "all present",
        )
    )

    # EXT-005: Required complexity keys
    req_c = {"temporal_lzc", "temporal_hfd", "multiscale_entropy_short"}
    miss_c = req_c - set(descriptor.complexity.keys())
    results.append(
        _rule(
            "EXT-005",
            S,
            ViolationCategory.CONTRACT,
            CausalSeverity.ERROR,
            len(miss_c) == 0,
            "Complexity keys complete",
            observed=sorted(miss_c) if miss_c else "all present",
        )
    )

    # EXT-006: Required connectivity keys
    req_conn = {"connectivity_divergence", "hierarchy_flattening"}
    miss_conn = req_conn - set(descriptor.connectivity.keys())
    results.append(
        _rule(
            "EXT-006",
            S,
            ViolationCategory.CONTRACT,
            CausalSeverity.ERROR,
            len(miss_conn) == 0,
            "Connectivity keys complete",
            observed=sorted(miss_conn) if miss_conn else "all present",
        )
    )

    return results


# ═══════════════════════════════════════════════════════════
# Stage: detect
# ═══════════════════════════════════════════════════════════


def _check_detection(detection: Any) -> list[CausalRuleResult]:
    results: list[CausalRuleResult] = []
    S = "detect"

    valid_anomaly = {"nominal", "watch", "anomalous"}
    valid_regime = {"stable", "transitional", "critical", "reorganized", "pathological_noise"}

    # DET-001: Score bounded
    results.append(
        _rule(
            "DET-001",
            S,
            ViolationCategory.NUMERICAL,
            CausalSeverity.ERROR,
            0.0 <= detection.score <= 1.0,
            "Anomaly score in [0,1]",
            observed=detection.score,
        )
    )

    # DET-002: Label valid
    results.append(
        _rule(
            "DET-002",
            S,
            ViolationCategory.CONTRACT,
            CausalSeverity.ERROR,
            detection.label in valid_anomaly,
            "Anomaly label valid",
            observed=detection.label,
        )
    )

    # DET-003: Regime label valid
    results.append(
        _rule(
            "DET-003",
            S,
            ViolationCategory.CONTRACT,
            CausalSeverity.ERROR,
            detection.regime.label in valid_regime,
            "Regime label valid",
            observed=detection.regime.label,
        )
    )

    # DET-004: Confidence bounded
    results.append(
        _rule(
            "DET-004",
            S,
            ViolationCategory.NUMERICAL,
            CausalSeverity.ERROR,
            0.0 <= detection.confidence <= 1.0,
            "Confidence in [0,1]",
            observed=detection.confidence,
        )
    )

    # DET-005: Contributing features subset of evidence keys
    evidence_keys = set(detection.evidence.keys())
    contrib = set(detection.contributing_features)
    ok = contrib.issubset(evidence_keys)
    results.append(
        _rule(
            "DET-005",
            S,
            ViolationCategory.CONTRACT,
            CausalSeverity.WARN,
            ok,
            "Contributing features must be subset of evidence keys",
            observed=sorted(contrib - evidence_keys) if not ok else "ok",
        )
    )

    # DET-006: Pathological noise causality
    if detection.regime.label == "pathological_noise":
        noise = detection.evidence.get("observation_noise_gain", 0.0)
        ok = noise >= 0.1
        results.append(
            _rule(
                "DET-006",
                S,
                ViolationCategory.CAUSAL,
                CausalSeverity.WARN,
                ok,
                "pathological_noise requires noise evidence >= 0.1",
                observed=noise,
                expected=0.1,
            )
        )

    # DET-007: Reorganized causality
    if detection.regime.label == "reorganized":
        plast = detection.evidence.get("plasticity_index", 0.0)
        ok = plast >= 0.05
        results.append(
            _rule(
                "DET-007",
                S,
                ViolationCategory.CAUSAL,
                CausalSeverity.WARN,
                ok,
                "reorganized requires plasticity >= 0.05",
                observed=plast,
                expected=0.05,
            )
        )

    # DET-008: Watch label near threshold
    if detection.label == "watch":
        threshold = detection.evidence.get("dynamic_threshold", 0.45)
        margin = abs(detection.score - threshold)
        ok = margin < 0.25
        results.append(
            _rule(
                "DET-008",
                S,
                ViolationCategory.CAUSAL,
                CausalSeverity.INFO,
                ok,
                f"Watch label should be near threshold (margin={margin:.3f})",
                observed=margin,
                expected=0.25,
            )
        )

    return results


# ═══════════════════════════════════════════════════════════
# Stage: forecast
# ═══════════════════════════════════════════════════════════


def _check_forecast(forecast: Any) -> list[CausalRuleResult]:
    results: list[CausalRuleResult] = []
    S = "forecast"
    fd = forecast.to_dict()

    # FOR-001: Horizon positive
    results.append(
        _rule(
            "FOR-001",
            S,
            ViolationCategory.CONTRACT,
            CausalSeverity.ERROR,
            fd["horizon"] >= 1,
            "Horizon must be >= 1",
            observed=fd["horizon"],
        )
    )

    # FOR-002: Predicted states finite and bounded
    for i, frame in enumerate(fd.get("predicted_states", [])):
        arr = np.asarray(frame)
        finite = bool(np.isfinite(arr).all())
        if not finite:
            results.append(
                _rule(
                    "FOR-002",
                    S,
                    ViolationCategory.NUMERICAL,
                    CausalSeverity.ERROR,
                    False,
                    f"Predicted state {i} contains NaN/Inf",
                )
            )
            break
        bounded = float(arr.min()) >= FIELD_V_MIN - 0.01 and float(arr.max()) <= FIELD_V_MAX + 0.01
        if not bounded:
            results.append(
                _rule(
                    "FOR-003",
                    S,
                    ViolationCategory.NUMERICAL,
                    CausalSeverity.WARN,
                    False,
                    f"Predicted state {i} outside bounds [{arr.min()*1000:.1f}, {arr.max()*1000:.1f}] mV",
                    observed=float(arr.max()),
                    expected=FIELD_V_MAX,
                )
            )
            break
    else:
        if fd.get("predicted_states"):
            results.append(
                _rule(
                    "FOR-002",
                    S,
                    ViolationCategory.NUMERICAL,
                    CausalSeverity.ERROR,
                    True,
                    "All predicted states finite",
                )
            )

    # FOR-004: Uncertainty envelope present
    env = fd.get("uncertainty_envelope", {})
    results.append(
        _rule(
            "FOR-004",
            S,
            ViolationCategory.CONTRACT,
            CausalSeverity.ERROR,
            bool(env),
            "Uncertainty envelope must be non-empty",
            observed=len(env),
        )
    )

    # FOR-005: Benchmark metrics required keys
    bm = fd.get("benchmark_metrics", {})
    has_keys = "forecast_structural_error" in bm and "adaptive_damping" in bm
    results.append(
        _rule(
            "FOR-005",
            S,
            ViolationCategory.CONTRACT,
            CausalSeverity.ERROR,
            has_keys,
            "Benchmark metrics must have forecast_structural_error and adaptive_damping",
        )
    )

    # FOR-006: Structural error bounded
    se = bm.get("forecast_structural_error", 0.0)
    results.append(
        _rule(
            "FOR-006",
            S,
            ViolationCategory.CAUSAL,
            CausalSeverity.WARN,
            se <= 1.0,
            f"Forecast structural error {se:.3f} <= 1.0 (divergence check)",
            observed=se,
            expected=1.0,
        )
    )

    # FOR-007: Damping in valid range
    damping = bm.get("adaptive_damping", 0.0)
    ok = 0.80 <= damping <= 0.95
    results.append(
        _rule(
            "FOR-007",
            S,
            ViolationCategory.NUMERICAL,
            CausalSeverity.WARN,
            ok,
            f"Adaptive damping {damping:.3f} in [0.80, 0.95]",
            observed=damping,
        )
    )

    return results


# ═══════════════════════════════════════════════════════════
# Stage: compare
# ═══════════════════════════════════════════════════════════


def _check_comparison(comparison: Any) -> list[CausalRuleResult]:
    results: list[CausalRuleResult] = []
    S = "compare"

    # CMP-001: Distance non-negative
    results.append(
        _rule(
            "CMP-001",
            S,
            ViolationCategory.NUMERICAL,
            CausalSeverity.ERROR,
            comparison.distance >= 0,
            "Distance must be >= 0",
            observed=comparison.distance,
        )
    )

    # CMP-002: Cosine bounded
    ok = -1.0 - 1e-6 <= comparison.cosine_similarity <= 1.0 + 1e-6
    results.append(
        _rule(
            "CMP-002",
            S,
            ViolationCategory.NUMERICAL,
            CausalSeverity.ERROR,
            ok,
            "Cosine similarity in [-1, 1]",
            observed=comparison.cosine_similarity,
        )
    )

    # CMP-003: Label valid
    valid = {"near-identical", "similar", "related", "divergent"}
    results.append(
        _rule(
            "CMP-003",
            S,
            ViolationCategory.CONTRACT,
            CausalSeverity.ERROR,
            comparison.label in valid,
            "Comparison label valid",
            observed=comparison.label,
        )
    )

    # CMP-004: Near-identical causality
    if comparison.label == "near-identical":
        ok = comparison.distance < 0.5
        results.append(
            _rule(
                "CMP-004",
                S,
                ViolationCategory.CAUSAL,
                CausalSeverity.WARN,
                ok,
                "near-identical should have distance < 0.5",
                observed=comparison.distance,
            )
        )

    # CMP-005: Divergent causality
    if comparison.label == "divergent":
        ok = comparison.cosine_similarity < 0.95
        results.append(
            _rule(
                "CMP-005",
                S,
                ViolationCategory.CAUSAL,
                CausalSeverity.WARN,
                ok,
                "divergent should have cosine < 0.95",
                observed=comparison.cosine_similarity,
            )
        )

    # CMP-006: Topology/reorganization consistency
    topo = comparison.topology_label
    reorg = comparison.reorganization_label
    expected_map = {
        "nominal": "stable",
        "flattened-hierarchy": "transitional",
        "pathological-drift": "pathological_noise",
        "reorganized": "reorganized",
    }
    expected_reorg = expected_map.get(topo, topo)
    results.append(
        _rule(
            "CMP-006",
            S,
            ViolationCategory.CAUSAL,
            CausalSeverity.ERROR,
            reorg == expected_reorg,
            "topology_label→reorganization_label mapping consistent",
            observed=f"{topo}→{reorg}",
            expected=f"{topo}→{expected_reorg}",
        )
    )

    return results


# ═══════════════════════════════════════════════════════════
# Cross-stage causal rules
# ═══════════════════════════════════════════════════════════


def _check_cross_stage(
    sequence: Any, detection: Any | None, forecast: Any | None
) -> list[CausalRuleResult]:
    results: list[CausalRuleResult] = []
    S = "cross_stage"

    if detection is None:
        return results

    # XST-001: Stable regime should not produce anomalous
    if detection.regime.label == "stable" and detection.label == "anomalous":
        results.append(
            _rule(
                "XST-001",
                S,
                ViolationCategory.CAUSAL,
                CausalSeverity.WARN,
                False,
                "stable regime should not produce anomalous label",
            )
        )

    # XST-002: Disabled neuromod → zero plasticity
    if sequence.spec is not None:
        nm = sequence.spec.neuromodulation
        disabled = nm is None or not getattr(nm, "enabled", False)
        if disabled:
            plast = detection.evidence.get("plasticity_index", 0.0)
            ok = plast < 0.01
            results.append(
                _rule(
                    "XST-002",
                    S,
                    ViolationCategory.CAUSAL,
                    CausalSeverity.WARN,
                    ok,
                    f"Neuromod disabled but plasticity={plast:.4f}",
                    observed=plast,
                    expected=0.01,
                )
            )

    # XST-003: If noise profile, regime should not be reorganized without structural evidence
    if sequence.spec is not None and sequence.spec.neuromodulation is not None:
        profile = sequence.spec.neuromodulation.profile
        if "noise" in profile and detection.regime.label == "reorganized":
            conn = detection.evidence.get("connectivity_divergence", 0.0)
            ok = conn >= 0.10
            results.append(
                _rule(
                    "XST-003",
                    S,
                    ViolationCategory.CAUSAL,
                    CausalSeverity.WARN,
                    ok,
                    "Noise profile + reorganized requires connectivity evidence >= 0.10",
                    observed=conn,
                    expected=0.10,
                )
            )

    return results


# ═══════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════


def validate_causal_consistency(
    sequence: Any,
    descriptor: Any | None = None,
    detection: Any | None = None,
    forecast: Any | None = None,
    comparison: Any | None = None,
    *,
    mode: str = "strict",
) -> CausalValidationResult:
    """Canonical entrypoint for causal validation.

    Verifies that each conclusion follows from data, invariants and rules.
    Does not substitute strategy — cuts false transitions and contradictions.

    Parameters
    ----------
    sequence : FieldSequence
    descriptor : MorphologyDescriptor | None
    detection : AnomalyEvent | None
    forecast : ForecastResult | None
    comparison : ComparisonResult | None
    mode : str
        ``'strict'`` (default): errors set decision=FAIL.
        ``'observe'``: all violations logged, decision always PASS or DEGRADED.
        ``'permissive'``: only FATAL violations set decision=FAIL.

    Returns
    -------
    CausalValidationResult
    """
    all_results: list[CausalRuleResult] = []
    stages = 0

    all_results.extend(_check_simulation(sequence))
    stages += 1

    if descriptor is not None:
        all_results.extend(_check_extraction(descriptor, sequence))
        stages += 1

    if detection is not None:
        all_results.extend(_check_detection(detection))
        stages += 1

    if forecast is not None:
        all_results.extend(_check_forecast(forecast))
        stages += 1

    if comparison is not None:
        all_results.extend(_check_comparison(comparison))
        stages += 1

    all_results.extend(_check_cross_stage(sequence, detection, forecast))
    stages += 1

    # Perturbation stability check (if detection available)
    if detection is not None and sequence.history is not None:
        all_results.extend(_check_perturbation_stability(sequence, detection))
        stages += 1

    # Determine aggregate decision based on mode
    has_fatal = any(not r.passed and r.severity == CausalSeverity.FATAL for r in all_results)
    has_error = any(not r.passed and r.severity == CausalSeverity.ERROR for r in all_results)
    has_warn = any(not r.passed and r.severity == CausalSeverity.WARN for r in all_results)

    if mode == "observe":
        # Observe-only: never FAIL, only log
        decision = (
            CausalDecision.DEGRADED if (has_error or has_warn or has_fatal) else CausalDecision.PASS
        )
    elif mode == "permissive":
        # Permissive: only FATAL causes FAIL
        if has_fatal:
            decision = CausalDecision.FAIL
        elif has_error or has_warn:
            decision = CausalDecision.DEGRADED
        else:
            decision = CausalDecision.PASS
    else:
        # Strict (default): ERROR or FATAL causes FAIL
        if has_fatal or has_error:
            decision = CausalDecision.FAIL
        elif has_warn:
            decision = CausalDecision.DEGRADED
        else:
            decision = CausalDecision.PASS

    # Compute provenance hash: deterministic fingerprint of all rule results
    rule_payload = json.dumps(
        [{"id": r.rule_id, "passed": r.passed, "sev": r.severity.value} for r in all_results],
        sort_keys=True,
    )
    config_hash = hashlib.sha256(rule_payload.encode()).hexdigest()[:16]

    return CausalValidationResult(
        decision=decision,
        rule_results=tuple(all_results),
        stages_checked=stages,
        runtime_hash=getattr(sequence, "runtime_hash", ""),
        config_hash=config_hash,
    )


# ═══════════════════════════════════════════════════════════
# Perturbation stability
# ═══════════════════════════════════════════════════════════


def _check_perturbation_stability(sequence: Any, detection: Any) -> list[CausalRuleResult]:
    """Check that small input perturbations don't flip the detection label.

    Adds tiny noise (1e-6) to the field and re-runs detection.
    If the label changes, the decision is near a threshold boundary
    and should be flagged as unstable.
    """
    from mycelium_fractal_net.core.detect import detect_anomaly
    from mycelium_fractal_net.types.field import FieldSequence

    results: list[CausalRuleResult] = []
    S = "perturbation"
    original_label = detection.label
    original_regime = detection.regime.label

    # 3 perturbation seeds
    label_stable = True
    regime_stable = True
    for seed_offset in (1, 2, 3):
        rng = np.random.default_rng(42 + seed_offset)
        noise = rng.normal(0, 1e-6, size=sequence.field.shape)
        perturbed_field = np.clip(sequence.field + noise, FIELD_V_MIN, FIELD_V_MAX)
        perturbed = FieldSequence(
            field=perturbed_field,
            history=sequence.history,
            spec=sequence.spec,
            metadata=sequence.metadata,
        )
        perturbed_detection = detect_anomaly(perturbed)
        if perturbed_detection.label != original_label:
            label_stable = False
        if perturbed_detection.regime.label != original_regime:
            regime_stable = False

    # PTB-001: Label stability under perturbation
    results.append(
        _rule(
            "PTB-001",
            S,
            ViolationCategory.CAUSAL,
            CausalSeverity.INFO,
            label_stable,
            f"Anomaly label '{original_label}' stable under 1e-6 noise perturbation",
            observed=label_stable,
            expected=True,
        )
    )

    # PTB-002: Regime stability under perturbation
    results.append(
        _rule(
            "PTB-002",
            S,
            ViolationCategory.CAUSAL,
            CausalSeverity.INFO,
            regime_stable,
            f"Regime label '{original_regime}' stable under 1e-6 noise perturbation",
            observed=regime_stable,
            expected=True,
        )
    )

    return results
