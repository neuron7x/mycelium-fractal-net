from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mycelium_fractal_net.analytics import compute_fractal_features
from mycelium_fractal_net.core import SimulationResult

_FIELD_MIN_V = -0.095
_FIELD_MAX_V = 0.040


@dataclass(frozen=True)
class DetectionResult:
    anomaly_score: float
    anomaly_label: str
    confidence: float
    regime_label: str
    components: dict[str, float]
    dominant_signals: list[str]
    feature_summary: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_score": self.anomaly_score,
            "anomaly_label": self.anomaly_label,
            "confidence": self.confidence,
            "regime_label": self.regime_label,
            "components": self.components,
            "dominant_signals": self.dominant_signals,
            "feature_summary": self.feature_summary,
        }


@dataclass(frozen=True)
class ForecastResult:
    horizon: int
    method: str
    predicted_field_min_mV: float
    predicted_field_max_mV: float
    predicted_field_mean_mV: float
    predicted_field_std_mV: float
    predicted_features: dict[str, float]
    trajectory: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "method": self.method,
            "predicted_field_min_mV": self.predicted_field_min_mV,
            "predicted_field_max_mV": self.predicted_field_max_mV,
            "predicted_field_mean_mV": self.predicted_field_mean_mV,
            "predicted_field_std_mV": self.predicted_field_std_mV,
            "predicted_features": self.predicted_features,
            "trajectory": self.trajectory,
        }


@dataclass(frozen=True)
class ComparisonResult:
    euclidean_distance: float
    cosine_similarity: float
    similarity_label: str
    top_feature_deltas: list[dict[str, float]]
    field_delta_mean_mV: float
    field_delta_max_mV: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "euclidean_distance": self.euclidean_distance,
            "cosine_similarity": self.cosine_similarity,
            "similarity_label": self.similarity_label,
            "top_feature_deltas": self.top_feature_deltas,
            "field_delta_mean_mV": self.field_delta_mean_mV,
            "field_delta_max_mV": self.field_delta_max_mV,
        }


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _feature_dict(result: SimulationResult) -> dict[str, float]:
    return {k: float(v) for k, v in compute_fractal_features(result).values.items()}


def detect_result(result: SimulationResult) -> DetectionResult:
    features = _feature_dict(result)
    d_box = features.get("D_box", 0.0)
    f_active = features.get("f_active", 0.0)
    v_std = abs(features.get("V_std", 0.0))
    t_stable = features.get("T_stable", 0.0)

    dimension_deviation = _clamp01(abs(d_box - 1.65) / 0.45)
    activity_imbalance = _clamp01(abs(f_active - 0.22) / 0.30)
    volatility_pressure = _clamp01(max(0.0, v_std - 20.0) / 40.0)
    clamp_pressure = _clamp01(result.clamping_events / max(1.0, result.num_steps * 0.25))
    stabilization_gap = _clamp01(1.0 - min(1.0, t_stable / max(1.0, result.num_steps)))

    components = {
        "dimension_deviation": round(dimension_deviation, 6),
        "activity_imbalance": round(activity_imbalance, 6),
        "volatility_pressure": round(volatility_pressure, 6),
        "clamp_pressure": round(clamp_pressure, 6),
        "stabilization_gap": round(stabilization_gap, 6),
    }
    anomaly_score = round(
        0.30 * dimension_deviation
        + 0.20 * activity_imbalance
        + 0.15 * volatility_pressure
        + 0.20 * clamp_pressure
        + 0.15 * stabilization_gap,
        6,
    )

    if anomaly_score >= 0.70:
        anomaly_label = "anomalous"
    elif anomaly_score >= 0.40:
        anomaly_label = "watch"
    else:
        anomaly_label = "nominal"

    if clamp_pressure > 0.50 or volatility_pressure > 0.70:
        regime_label = "unstable"
    elif result.has_history and t_stable < max(3, result.num_steps // 4):
        regime_label = "transitional"
    elif d_box >= 1.40 and f_active >= 0.10:
        regime_label = "pattern-forming"
    else:
        regime_label = "quiescent"

    dominant_signals = [
        key
        for key, _ in sorted(components.items(), key=lambda item: item[1], reverse=True)
        if components[key] > 0.05
    ][:3]

    confidence = round(0.55 + 0.35 * abs(anomaly_score - 0.5), 6)
    feature_summary = {
        "D_box": round(d_box, 6),
        "f_active": round(f_active, 6),
        "V_mean": round(features.get("V_mean", 0.0), 6),
        "V_std": round(v_std, 6),
        "T_stable": round(t_stable, 6),
    }
    return DetectionResult(
        anomaly_score=anomaly_score,
        anomaly_label=anomaly_label,
        confidence=confidence,
        regime_label=regime_label,
        components=components,
        dominant_signals=dominant_signals,
        feature_summary=feature_summary,
    )


def forecast_result(result: SimulationResult, horizon: int = 8) -> ForecastResult:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    if result.history is None or result.history.shape[0] < 2:
        predicted_field = result.field.copy()
        method = "persistence"
        history_stack = np.repeat(predicted_field[None, :, :], horizon, axis=0)
    else:
        window = min(5, result.history.shape[0] - 1)
        recent = result.history[-(window + 1) :]
        avg_delta = np.mean(np.diff(recent, axis=0), axis=0)
        forecast_frames: list[NDArray[np.float64]] = []
        current = result.history[-1].astype(np.float64).copy()
        for _ in range(horizon):
            current = np.clip(current + avg_delta, _FIELD_MIN_V, _FIELD_MAX_V)
            forecast_frames.append(current.copy())
        history_stack = np.stack(forecast_frames, axis=0)
        predicted_field = history_stack[-1]
        method = "linear_delta_extrapolation"

    predicted_result = SimulationResult(field=predicted_field.astype(np.float64), history=history_stack.astype(np.float64), metadata={"forecast_horizon": horizon, "method": method})
    predicted_features = _feature_dict(predicted_result)

    trajectory: list[dict[str, float]] = []
    for idx, frame in enumerate(history_stack, start=1):
        frame_result = SimulationResult(field=frame.astype(np.float64), history=None)
        frame_features = _feature_dict(frame_result)
        trajectory.append(
            {
                "step": float(idx),
                "field_mean_mV": float(np.mean(frame) * 1000.0),
                "field_std_mV": float(np.std(frame) * 1000.0),
                "D_box": float(frame_features.get("D_box", 0.0)),
                "f_active": float(frame_features.get("f_active", 0.0)),
            }
        )

    return ForecastResult(
        horizon=horizon,
        method=method,
        predicted_field_min_mV=float(np.min(predicted_field) * 1000.0),
        predicted_field_max_mV=float(np.max(predicted_field) * 1000.0),
        predicted_field_mean_mV=float(np.mean(predicted_field) * 1000.0),
        predicted_field_std_mV=float(np.std(predicted_field) * 1000.0),
        predicted_features={k: round(v, 6) for k, v in predicted_features.items()},
        trajectory=[
            {k: round(v, 6) if isinstance(v, float) else v for k, v in entry.items()} for entry in trajectory
        ],
    )


def compare_results(left: SimulationResult, right: SimulationResult) -> ComparisonResult:
    left_features = _feature_dict(left)
    right_features = _feature_dict(right)

    keys = sorted(set(left_features) & set(right_features))
    left_vec = np.array([left_features[key] for key in keys], dtype=np.float64)
    right_vec = np.array([right_features[key] for key in keys], dtype=np.float64)
    delta = left_vec - right_vec
    euclidean_distance = float(np.linalg.norm(delta))

    denom = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
    cosine_similarity = float(np.dot(left_vec, right_vec) / denom) if denom > 0 else 1.0

    top_feature_deltas = [
        {
            "feature": key,
            "left": round(float(left_features[key]), 6),
            "right": round(float(right_features[key]), 6),
            "abs_delta": round(abs(float(left_features[key]) - float(right_features[key])), 6),
        }
        for key in sorted(keys, key=lambda item: abs(left_features[item] - right_features[item]), reverse=True)[:5]
    ]

    if cosine_similarity >= 0.995 and euclidean_distance < 5.0:
        similarity_label = "near-identical"
    elif cosine_similarity >= 0.97:
        similarity_label = "similar"
    elif cosine_similarity >= 0.90:
        similarity_label = "related"
    else:
        similarity_label = "divergent"

    field_delta = np.abs(left.field - right.field)
    return ComparisonResult(
        euclidean_distance=round(euclidean_distance, 6),
        cosine_similarity=round(cosine_similarity, 6),
        similarity_label=similarity_label,
        top_feature_deltas=top_feature_deltas,
        field_delta_mean_mV=round(float(np.mean(field_delta) * 1000.0), 6),
        field_delta_max_mV=round(float(np.max(field_delta) * 1000.0), 6),
    )
