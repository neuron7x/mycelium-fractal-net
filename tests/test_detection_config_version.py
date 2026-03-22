"""Verify detection constants match versioned config file.

If this test fails, either:
1. Constants in detect.py were changed without updating the config JSON
2. Config JSON was updated without updating detect.py constants
Both require explicit version bump and re-calibration evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_detection_config_version_matches() -> None:
    from mycelium_fractal_net.core.detect import DETECTION_CONFIG_VERSION

    config_path = ROOT / "configs" / "detection_thresholds_v1.json"
    config = json.loads(config_path.read_text())
    assert config["schema_version"] == DETECTION_CONFIG_VERSION


def test_anomaly_weights_sum_to_one() -> None:
    config_path = ROOT / "configs" / "detection_thresholds_v1.json"
    config = json.loads(config_path.read_text())
    weights = config["anomaly_weights"]
    total = sum(weights.values())
    assert total == pytest.approx(1.0, abs=1e-10), f"Anomaly weights sum to {total}, not 1.0"


def test_instability_weights_sum_to_one() -> None:
    config_path = ROOT / "configs" / "detection_thresholds_v1.json"
    config = json.loads(config_path.read_text())
    weights = config["instability_weights"]
    total = sum(weights.values())
    assert total == pytest.approx(1.0, abs=1e-10), f"Instability weights sum to {total}, not 1.0"


def test_constants_match_config_json() -> None:
    """Verify detect.py constants match the versioned config file."""
    from mycelium_fractal_net.core import detect

    config_path = ROOT / "configs" / "detection_thresholds_v1.json"
    config = json.loads(config_path.read_text())

    # Evidence normalization
    norm = config["evidence_normalization"]
    assert detect._TEMPORAL_LZC_NORMALIZER == norm["temporal_lzc_normalizer"]
    assert detect._CONNECTIVITY_AMPLIFICATION == norm["connectivity_amplification"]
    assert detect._HIERARCHY_BASELINE == norm["hierarchy_baseline"]
    assert detect._HIERARCHY_RANGE == norm["hierarchy_range"]
    assert detect._CRITICALITY_AMPLIFICATION == norm["criticality_amplification"]
    assert detect._NOISE_GAIN_AMPLIFICATION == norm["noise_gain_amplification"]

    # Regime thresholds
    rt = config["regime_thresholds"]
    assert detect._DYNAMIC_ANOMALY_BASELINE == rt["dynamic_anomaly_baseline"]
    assert detect._REORGANIZED_COMPLEXITY_THRESHOLD == rt["reorganized_complexity_threshold"]
    assert detect._REORGANIZED_TOPOLOGY_THRESHOLD == rt["reorganized_topology_threshold"]
    assert detect._REORGANIZED_PLASTICITY_FLOOR == rt["reorganized_plasticity_floor"]
    assert detect._PATHOLOGICAL_NOISE_THRESHOLD == rt["pathological_noise_threshold"]
    assert detect._STRUCTURE_FLOOR == rt["structure_floor"]
    assert detect._STABLE_CEILING == rt["stable_ceiling"]

    # Anomaly weights
    aw = config["anomaly_weights"]
    assert detect._ANOMALY_W_INSTABILITY == aw["instability"]
    assert detect._ANOMALY_W_TRANSITION == aw["transition"]
    assert detect._ANOMALY_W_COLLAPSE == aw["collapse"]
    assert detect._ANOMALY_W_CHANGE == aw["change"]
    assert detect._ANOMALY_W_VOLATILITY == aw["volatility"]
    assert detect._ANOMALY_W_NOISE == aw["noise"]
    assert detect._ANOMALY_W_CONNECTIVITY == aw["connectivity"]
    assert detect._ANOMALY_W_PLASTICITY == aw["plasticity"]

    # Profile hints
    ph = config["profile_hints"]
    assert detect._PROFILE_HINT_SEROTONERGIC == ph["serotonergic"]
    assert detect._PROFILE_HINT_CRITICALITY == ph["criticality"]


def test_compare_constants_match_config() -> None:
    """Compare thresholds in code must match versioned config."""
    from mycelium_fractal_net.core import compare

    config_path = ROOT / "configs" / "detection_thresholds_v1.json"
    config = json.loads(config_path.read_text())
    cc = config["comparison"]

    assert compare._COSINE_NEAR_IDENTICAL == cc["cosine_near_identical"]
    assert compare._COSINE_SIMILAR == cc["cosine_similar"]
    assert compare._COSINE_RELATED == cc["cosine_related"]
    assert compare._DISTANCE_NEAR_IDENTICAL == cc["distance_near_identical"]
    assert compare._NOISE_PATHOLOGICAL_HIGH == cc["noise_pathological_high"]
    assert compare._NOISE_PATHOLOGICAL_LOW == cc["noise_pathological_low"]
    assert compare._CONNECTIVITY_LOW == cc["connectivity_low"]
    assert compare._MODULARITY_LOW == cc["modularity_low"]
    assert compare._HIERARCHY_FLAT_THRESHOLD == cc["hierarchy_flat_threshold"]
    assert compare._CONNECTIVITY_FLAT_CEILING == cc["connectivity_flat_ceiling"]
    assert compare._CONNECTIVITY_REORG_THRESHOLD == cc["connectivity_reorg_threshold"]
    assert compare._MODULARITY_REORG_THRESHOLD == cc["modularity_reorg_threshold"]
    assert compare._TOP_CHANGED_FEATURES == cc["top_changed_features"]
