"""Tests for the scenarios data pipeline helpers."""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd

import mycelium_fractal_net.analytics as analytics_mod
import mycelium_fractal_net.pipelines.scenarios as scenarios
from mycelium_fractal_net.pipelines.scenarios import ScenarioConfig, _generate_param_configs


def test_generate_param_configs_uses_rng_for_shuffling() -> None:
    """Providing an RNG should change the ordering without changing seeds."""

    config = ScenarioConfig(
        name="test",
        num_samples=4,
        seeds_per_config=1,
        alpha_values=[0.1, 0.2],
    )

    rng = np.random.default_rng(123)
    configs = _generate_param_configs(config, rng=rng)

    sim_ids = [cfg["sim_id"] for cfg in configs]
    seeds = [cfg["random_seed"] for cfg in configs]

    # All simulations should still be present but in a shuffled order
    assert set(sim_ids) == {0, 1, 2, 3}
    assert sim_ids != sorted(sim_ids)

    # The seeds remain tied to sim_id even after shuffling
    assert seeds == [config.base_seed + sim_id for sim_id in sim_ids]


def test_run_scenario_uses_package_analytics(monkeypatch, tmp_path) -> None:
    """Ensure the data pipeline relies on the namespaced analytics module."""

    class FailingModule(types.ModuleType):
        def __getattr__(self, name: str):  # pragma: no cover - defensive
            raise AssertionError("bare analytics import should not be used")

    # Insert a sentinel module that would explode if imported via the bare name
    monkeypatch.setitem(sys.modules, "analytics", FailingModule("analytics"))

    # Stub analytics feature extraction to avoid heavy computation
    class DummyFeatureVector:
        def __init__(self, value: float = 0.0) -> None:
            self.value = value

        def to_dict(self) -> dict[str, float]:
            return {"dummy_feature": self.value}

        @classmethod
        def feature_names(cls) -> list[str]:
            return ["dummy_feature"]

    def fake_compute_features(history: np.ndarray, config: object) -> DummyFeatureVector:
        return DummyFeatureVector(1.23)

    monkeypatch.setattr(analytics_mod, "compute_features", fake_compute_features)
    monkeypatch.setattr(analytics_mod, "FeatureVector", DummyFeatureVector)

    # Bypass the expensive simulator with a deterministic placeholder
    def fake_run_single_simulation(params: dict[str, object]):
        history = np.zeros(
            (params["steps"], params["grid_size"], params["grid_size"]), dtype=np.float64
        )
        metadata = {
            "growth_events": 0,
            "turing_activations": 0,
            "clamping_events": 0,
        }
        return history, metadata

    monkeypatch.setattr(scenarios, "_run_single_simulation", fake_run_single_simulation)

    config = ScenarioConfig(
        name="namespaced",
        grid_size=8,
        steps=1,
        num_samples=1,
        seeds_per_config=1,
        alpha_values=[0.1],
        output_format="csv",
        output_dir="tmp",
    )

    meta = scenarios.run_scenario(config, data_root=tmp_path)

    assert meta.output_path.exists()
    df = pd.read_csv(meta.output_path)
    assert "dummy_feature" in df.columns
    assert df["dummy_feature"].iloc[0] == 1.23
    assert meta.feature_names == ["dummy_feature"]
