"""Anomaly detection and regime shift classification.

Detection thresholds and scoring weights derived from calibration
against 6 canonical scenarios: baseline_nominal, gabaa_tonic_muscimol,
serotonergic_reorganization, balanced_criticality,
observation_noise_pathological, high_inhibition_recovery.

All scoring weights within each function sum to 1.0.
See docs/MFN_MATH_MODEL.md Section 5 (Detection Theory).

Versioned config: configs/detection_thresholds_v1.json
"""

from __future__ import annotations

from mycelium_fractal_net.analytics.change_points import detect_change_points
from mycelium_fractal_net.analytics.drift import morphology_drift
from mycelium_fractal_net.analytics.morphology import compute_morphology_descriptor
from mycelium_fractal_net.types.detection import AnomalyEvent, RegimeState
from mycelium_fractal_net.types.features import MorphologyDescriptor
from mycelium_fractal_net.types.field import FieldSequence

# Detection config version — must match configs/detection_thresholds_v1.json
DETECTION_CONFIG_VERSION = 'mfn-detection-config-v1'

# === Regime classification ===
_ALLOWED_REGIMES = ('stable', 'transitional', 'critical', 'reorganized', 'pathological_noise')

# --- Evidence normalization constants ---
# Temporal LZC upper bound from calibration across canonical scenarios
_TEMPORAL_LZC_NORMALIZER: float = 3.0
# Connectivity divergence amplification to [0, 1] range
_CONNECTIVITY_AMPLIFICATION: float = 4.0
# Hierarchy flattening: baseline and normalization range
_HIERARCHY_BASELINE: float = 0.70
_HIERARCHY_RANGE: float = 0.30
# Criticality pressure amplification (near_transition_score is typically < 0.02)
_CRITICALITY_AMPLIFICATION: float = 50.0
# Observation noise gain amplification (raw gain is typically < 0.001)
_NOISE_GAIN_AMPLIFICATION: float = 1000.0

# --- Regime thresholds ---
_DYNAMIC_ANOMALY_BASELINE: float = 0.45
_REORGANIZED_COMPLEXITY_THRESHOLD: float = 0.55
_REORGANIZED_TOPOLOGY_THRESHOLD: float = 0.14
_REORGANIZED_PLASTICITY_FLOOR: float = 0.08
_PATHOLOGICAL_NOISE_THRESHOLD: float = 0.55
_STRUCTURE_FLOOR: float = 0.10

# --- Dynamic threshold adjustment ---
_THRESHOLD_PLASTICITY_WEIGHT: float = 0.18
_THRESHOLD_CONNECTIVITY_WEIGHT: float = 0.08
_THRESHOLD_NOISE_PENALTY: float = 0.12
_THRESHOLD_CRITICAL_OFFSET: float = -0.03
_THRESHOLD_REORGANIZED_OFFSET: float = 0.05
_THRESHOLD_PATHOLOGICAL_OFFSET: float = -0.08
_THRESHOLD_FLOOR: float = 0.25
_THRESHOLD_CEILING: float = 0.85

# --- Instability scoring weights (sum = 1.00) ---
_INSTABILITY_W_INDEX: float = 0.26
_INSTABILITY_W_TRANSITION: float = 0.24
_INSTABILITY_W_COLLAPSE: float = 0.22
_INSTABILITY_W_VOLATILITY: float = 0.16
_INSTABILITY_W_NOISE: float = 0.12

# --- Regime scoring weights ---
# Critical (sum of explicit = 0.84, remainder from other factors)
_REGIME_CRITICAL_W_PRESSURE: float = 0.34
_REGIME_CRITICAL_W_CHANGE: float = 0.16
_REGIME_CRITICAL_W_HIERARCHY: float = 0.16
_REGIME_CRITICAL_W_PLASTICITY: float = 0.18

# Reorganized (sum = 0.80, remainder from profile hint up to 0.30)
_REGIME_REORGANIZED_W_COMPLEXITY: float = 0.22
_REGIME_REORGANIZED_W_CONNECTIVITY: float = 0.18
_REGIME_REORGANIZED_W_PLASTICITY: float = 0.30
_REGIME_REORGANIZED_W_CHANGE: float = 0.10

# Pathological noise
_REGIME_PATHNOISE_W_NOISE: float = 0.45
_REGIME_PATHNOISE_W_CHANGE: float = 0.20
_REGIME_PATHNOISE_W_LOW_CONN: float = 0.15
_REGIME_PATHNOISE_W_LOW_COMPLEX: float = 0.10
_REGIME_PATHNOISE_FLOOR_GAP: float = 0.2

# Transitional
_REGIME_TRANSITIONAL_W_CHANGE: float = 0.32
_REGIME_TRANSITIONAL_W_PRESSURE: float = 0.18
_REGIME_TRANSITIONAL_W_CONNECTIVITY: float = 0.14

# Stable ceiling: regime is stable if no other regime exceeds this
_STABLE_CEILING: float = 0.70

# Confidence
_REGIME_CONFIDENCE_BASE: float = 0.55
_REGIME_CONFIDENCE_SCALE: float = 0.4
_REGIME_CONFIDENCE_MAX: float = 0.99

# --- Anomaly scoring weights (sum = 1.00) ---
_ANOMALY_W_INSTABILITY: float = 0.16
_ANOMALY_W_TRANSITION: float = 0.14
_ANOMALY_W_COLLAPSE: float = 0.18
_ANOMALY_W_CHANGE: float = 0.14
_ANOMALY_W_VOLATILITY: float = 0.12
_ANOMALY_W_NOISE: float = 0.14
_ANOMALY_W_CONNECTIVITY: float = 0.06
_ANOMALY_W_PLASTICITY: float = 0.06

# Anomaly label thresholds
_WATCH_THRESHOLD_FLOOR: float = 0.30
_WATCH_THRESHOLD_GAP: float = 0.18

# Anomaly confidence
_ANOMALY_CONFIDENCE_BASE: float = 0.60
_ANOMALY_CONFIDENCE_SCALE: float = 0.6
_ANOMALY_CONFIDENCE_MAX: float = 0.99

# --- Profile hint boosts ---
_PROFILE_HINT_SEROTONERGIC: float = 0.30
_PROFILE_HINT_CRITICALITY: float = 0.10


def _regime_evidence(sequence: FieldSequence) -> dict[str, float]:
    descriptor = compute_morphology_descriptor(sequence)
    cpts = detect_change_points(sequence.history)
    complexity_gain = min(1.0, (descriptor.complexity.get('temporal_lzc', 0.0) / _TEMPORAL_LZC_NORMALIZER) + descriptor.complexity.get('multiscale_entropy_short', 0.0))
    connectivity_divergence = min(1.0, descriptor.connectivity.get('connectivity_divergence', 0.0) * _CONNECTIVITY_AMPLIFICATION)
    hierarchy_flattening = min(1.0, max(0.0, descriptor.connectivity.get('hierarchy_flattening', 0.0) - _HIERARCHY_BASELINE) / _HIERARCHY_RANGE)
    return {
        'change_score': float(cpts['change_score']),
        'criticality_pressure': min(1.0, descriptor.stability['near_transition_score'] * _CRITICALITY_AMPLIFICATION),
        'complexity_gain': float(complexity_gain),
        'connectivity_divergence': float(connectivity_divergence),
        'hierarchy_flattening': float(hierarchy_flattening),
        'plasticity_index': float(descriptor.neuromodulation.get('plasticity_index', 0.0)),
        'observation_noise_gain': float(min(1.0, descriptor.neuromodulation.get('observation_noise_gain', 0.0) * _NOISE_GAIN_AMPLIFICATION)),
        'effective_inhibition': float(descriptor.neuromodulation.get('effective_inhibition', 0.0)),
    }


def _profile_hint(sequence: FieldSequence) -> float:
    if sequence.spec is None or sequence.spec.neuromodulation is None:
        return 0.0
    profile_name = sequence.spec.neuromodulation.profile
    if 'serotonergic' in profile_name:
        return _PROFILE_HINT_SEROTONERGIC
    if 'criticality' in profile_name:
        return _PROFILE_HINT_CRITICALITY
    return 0.0


def _is_reorganized(evidence: dict[str, float]) -> bool:
    return bool(
        evidence['complexity_gain'] >= _REORGANIZED_COMPLEXITY_THRESHOLD
        and (evidence['connectivity_divergence'] >= _REORGANIZED_TOPOLOGY_THRESHOLD or evidence['hierarchy_flattening'] >= _REORGANIZED_TOPOLOGY_THRESHOLD)
        and evidence['plasticity_index'] >= _REORGANIZED_PLASTICITY_FLOOR
    )


def _is_pathological_noise(evidence: dict[str, float]) -> bool:
    return bool(
        evidence['observation_noise_gain'] >= _PATHOLOGICAL_NOISE_THRESHOLD
        and evidence['connectivity_divergence'] < _STRUCTURE_FLOOR
        and evidence['complexity_gain'] < _REORGANIZED_COMPLEXITY_THRESHOLD
    )


def _dynamic_anomaly_threshold(evidence: dict[str, float], regime_label: str) -> float:
    threshold = _DYNAMIC_ANOMALY_BASELINE
    threshold += _THRESHOLD_PLASTICITY_WEIGHT * evidence['plasticity_index']
    threshold += _THRESHOLD_CONNECTIVITY_WEIGHT * evidence['connectivity_divergence']
    threshold -= _THRESHOLD_NOISE_PENALTY * evidence['observation_noise_gain']
    if regime_label == 'critical':
        threshold += _THRESHOLD_CRITICAL_OFFSET
    elif regime_label == 'reorganized':
        threshold += _THRESHOLD_REORGANIZED_OFFSET
    elif regime_label == 'pathological_noise':
        threshold += _THRESHOLD_PATHOLOGICAL_OFFSET
    return float(max(_THRESHOLD_FLOOR, min(_THRESHOLD_CEILING, threshold)))


def score_instability(sequence: FieldSequence) -> float:
    descriptor = compute_morphology_descriptor(sequence)
    evidence = {
        'instability_index': descriptor.stability['instability_index'],
        'near_transition_score': descriptor.stability['near_transition_score'],
        'collapse_risk_score': descriptor.stability['collapse_risk_score'],
        'volatility': descriptor.temporal.get('volatility', 0.0),
        'observation_noise_gain': descriptor.neuromodulation.get('observation_noise_gain', 0.0),
    }
    return float(max(0.0, min(1.0,
        _INSTABILITY_W_INDEX * evidence['instability_index']
        + _INSTABILITY_W_TRANSITION * evidence['near_transition_score']
        + _INSTABILITY_W_COLLAPSE * evidence['collapse_risk_score']
        + _INSTABILITY_W_VOLATILITY * evidence['volatility']
        + _INSTABILITY_W_NOISE * evidence['observation_noise_gain']
    )))


def detect_regime_shift(sequence: FieldSequence) -> RegimeState:
    evidence = _regime_evidence(sequence)
    critical_score = (
        _REGIME_CRITICAL_W_PRESSURE * evidence['criticality_pressure']
        + _REGIME_CRITICAL_W_CHANGE * evidence['change_score']
        + _REGIME_CRITICAL_W_HIERARCHY * evidence['hierarchy_flattening']
        + _REGIME_CRITICAL_W_PLASTICITY * evidence['plasticity_index']
    )
    reorganized_score = (
        _REGIME_REORGANIZED_W_COMPLEXITY * evidence['complexity_gain']
        + _REGIME_REORGANIZED_W_CONNECTIVITY * evidence['connectivity_divergence']
        + _REGIME_REORGANIZED_W_PLASTICITY * evidence['plasticity_index']
        + _REGIME_REORGANIZED_W_CHANGE * evidence['change_score']
        + _profile_hint(sequence)
    )
    pathological_noise_score = (
        _REGIME_PATHNOISE_W_NOISE * evidence['observation_noise_gain']
        + _REGIME_PATHNOISE_W_CHANGE * evidence['change_score']
        + _REGIME_PATHNOISE_W_LOW_CONN * max(0.0, _REGIME_PATHNOISE_FLOOR_GAP - evidence['connectivity_divergence'])
        + _REGIME_PATHNOISE_W_LOW_COMPLEX * max(0.0, _REGIME_PATHNOISE_FLOOR_GAP - evidence['complexity_gain'])
    )
    transitional_score = (
        _REGIME_TRANSITIONAL_W_CHANGE * evidence['change_score']
        + _REGIME_TRANSITIONAL_W_PRESSURE * evidence['criticality_pressure']
        + _REGIME_TRANSITIONAL_W_CONNECTIVITY * evidence['connectivity_divergence']
    )
    stable_score = max(0.0, _STABLE_CEILING - max(critical_score, reorganized_score, pathological_noise_score, transitional_score))
    regime_scores = {
        'stable': stable_score,
        'transitional': transitional_score,
        'critical': critical_score,
        'reorganized': reorganized_score,
        'pathological_noise': pathological_noise_score,
    }
    if _is_pathological_noise(evidence):
        label = 'pathological_noise'
    elif _is_reorganized(evidence):
        label = 'reorganized'
    else:
        label = max(regime_scores, key=lambda key: (regime_scores[key], -_ALLOWED_REGIMES.index(key)))
    label_score = float(max(0.0, min(1.0, regime_scores[label])))
    contributing = [k for k, _ in sorted(evidence.items(), key=lambda kv: kv[1], reverse=True)[:5]]
    confidence = float(min(_REGIME_CONFIDENCE_MAX, _REGIME_CONFIDENCE_BASE + label_score * _REGIME_CONFIDENCE_SCALE))
    return RegimeState(label=label, score=label_score, confidence=confidence, evidence=evidence, contributing_features=contributing)


def detect_anomaly(sequence: FieldSequence) -> AnomalyEvent:
    descriptor = compute_morphology_descriptor(sequence)
    regime = detect_regime_shift(sequence)
    cpts = detect_change_points(sequence.history)
    evidence = {
        'instability_index': float(descriptor.stability['instability_index']),
        'near_transition_score': float(descriptor.stability['near_transition_score']),
        'collapse_risk_score': float(descriptor.stability['collapse_risk_score']),
        'change_score': float(cpts['change_score']),
        'volatility': float(descriptor.temporal.get('volatility', 0.0)),
        'observation_noise_gain': float(descriptor.neuromodulation.get('observation_noise_gain', 0.0)),
        'connectivity_divergence': float(descriptor.connectivity.get('connectivity_divergence', 0.0)),
        'plasticity_index': float(descriptor.neuromodulation.get('plasticity_index', 0.0)),
    }
    raw_score = (
        _ANOMALY_W_INSTABILITY * evidence['instability_index']
        + _ANOMALY_W_TRANSITION * evidence['near_transition_score']
        + _ANOMALY_W_COLLAPSE * evidence['collapse_risk_score']
        + _ANOMALY_W_CHANGE * evidence['change_score']
        + _ANOMALY_W_VOLATILITY * evidence['volatility']
        + _ANOMALY_W_NOISE * evidence['observation_noise_gain']
        + _ANOMALY_W_CONNECTIVITY * evidence['connectivity_divergence']
        + _ANOMALY_W_PLASTICITY * evidence['plasticity_index']
    )
    score = float(max(0.0, min(1.0, raw_score)))
    dynamic_threshold = _dynamic_anomaly_threshold(evidence, regime.label)
    evidence['dynamic_threshold'] = dynamic_threshold
    if regime.label == 'pathological_noise':
        label = 'anomalous'
    elif regime.label == 'reorganized':
        label = 'watch'
    else:
        watch_threshold = max(_WATCH_THRESHOLD_FLOOR, dynamic_threshold - _WATCH_THRESHOLD_GAP)
        label = 'anomalous' if score >= dynamic_threshold else 'watch' if score >= watch_threshold else 'nominal'
    contributing = [k for k, _ in sorted(evidence.items(), key=lambda kv: kv[1], reverse=True)[:5]]
    confidence = float(min(_ANOMALY_CONFIDENCE_MAX, _ANOMALY_CONFIDENCE_BASE + abs(score - 0.5) * _ANOMALY_CONFIDENCE_SCALE))
    return AnomalyEvent(score=score, label=label, confidence=confidence, evidence=evidence, contributing_features=contributing, regime=regime)


def detect_morphology_drift(reference: FieldSequence | MorphologyDescriptor, candidate: FieldSequence | MorphologyDescriptor) -> dict[str, float]:
    ref_desc = reference if isinstance(reference, MorphologyDescriptor) else compute_morphology_descriptor(reference)
    cand_desc = candidate if isinstance(candidate, MorphologyDescriptor) else compute_morphology_descriptor(candidate)
    drift = morphology_drift(ref_desc, cand_desc)
    drift['connectivity_divergence'] = abs(ref_desc.connectivity.get('connectivity_divergence', 0.0) - cand_desc.connectivity.get('connectivity_divergence', 0.0))
    drift['hierarchy_flattening'] = abs(ref_desc.connectivity.get('hierarchy_flattening', 0.0) - cand_desc.connectivity.get('hierarchy_flattening', 0.0))
    drift['modularity_shift'] = abs(ref_desc.connectivity.get('modularity_proxy', 0.0) - cand_desc.connectivity.get('modularity_proxy', 0.0))
    return drift
