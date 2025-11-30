"""
Tests for the RunRegistry module.

Verifies that the run registry correctly creates directories, stores configs,
metadata, and metrics, and can list/retrieve runs.
"""

import json
from pathlib import Path

import pytest

from mycelium_fractal_net import SimulationConfig
from mycelium_fractal_net.infra.run_registry import (
    RunRegistry,
    get_registry,
    reset_global_registry,
)


class TestRunRegistry:
    """Test RunRegistry functionality."""

    @pytest.fixture
    def tmp_registry(self, tmp_path: Path) -> RunRegistry:
        """Create a registry with a temporary directory."""
        return RunRegistry(root_dir=tmp_path, enabled=True)

    def test_start_run_creates_directory_structure(
        self, tmp_registry: RunRegistry, tmp_path: Path
    ) -> None:
        """start_run should create the run directory with required files."""
        config = SimulationConfig(grid_size=32, steps=20, seed=42)
        run = tmp_registry.start_run(config, run_type="simulation", seed=42)

        # Run directory should exist
        assert run.run_dir.exists()
        assert run.run_dir.is_dir()

        # config.json should exist
        config_path = run.run_dir / "config.json"
        assert config_path.exists()

        # meta.json should exist
        meta_path = run.run_dir / "meta.json"
        assert meta_path.exists()

    def test_config_json_contains_correct_fields(
        self, tmp_registry: RunRegistry
    ) -> None:
        """config.json should contain all config fields."""
        config = SimulationConfig(
            grid_size=64, steps=100, seed=123, turing_enabled=True
        )
        run = tmp_registry.start_run(config, run_type="simulation")

        config_path = run.run_dir / "config.json"
        with open(config_path) as f:
            saved_config = json.load(f)

        assert saved_config["grid_size"] == 64
        assert saved_config["steps"] == 100
        assert saved_config["seed"] == 123
        assert saved_config["turing_enabled"] is True

    def test_meta_json_contains_required_fields(
        self, tmp_registry: RunRegistry
    ) -> None:
        """meta.json should contain all required metadata fields."""
        config = {"seed": 42, "name": "test"}
        run = tmp_registry.start_run(config, run_type="validation", seed=42)

        meta_path = run.run_dir / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        # Required fields
        assert "run_id" in meta
        assert "run_type" in meta
        assert "timestamp" in meta
        assert "status" in meta
        assert "seed" in meta

        # Values
        assert meta["run_type"] == "validation"
        assert meta["status"] == "running"
        assert meta["seed"] == 42

    def test_log_metrics_creates_metrics_file(
        self, tmp_registry: RunRegistry
    ) -> None:
        """log_metrics should create metrics.json."""
        config = {"seed": 42}
        run = tmp_registry.start_run(config, run_type="benchmark")

        tmp_registry.log_metrics(run, {"loss": 0.123, "accuracy": 0.95})

        metrics_path = run.run_dir / "metrics.json"
        assert metrics_path.exists()

        with open(metrics_path) as f:
            metrics = json.load(f)

        assert metrics["loss"] == 0.123
        assert metrics["accuracy"] == 0.95

    def test_log_metrics_appends_by_default(
        self, tmp_registry: RunRegistry
    ) -> None:
        """log_metrics should append to existing metrics."""
        config = {"seed": 42}
        run = tmp_registry.start_run(config, run_type="experiment")

        tmp_registry.log_metrics(run, {"loss": 0.5})
        tmp_registry.log_metrics(run, {"accuracy": 0.8})

        metrics_path = run.run_dir / "metrics.json"
        with open(metrics_path) as f:
            metrics = json.load(f)

        assert metrics["loss"] == 0.5
        assert metrics["accuracy"] == 0.8

    def test_end_run_updates_status(
        self, tmp_registry: RunRegistry
    ) -> None:
        """end_run should update the status in meta.json."""
        config = {"seed": 42}
        run = tmp_registry.start_run(config, run_type="simulation")

        tmp_registry.end_run(run, status="success")

        meta_path = run.run_dir / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["status"] == "success"
        assert "end_timestamp" in meta
        assert "duration_seconds" in meta

    def test_end_run_with_final_metrics(
        self, tmp_registry: RunRegistry
    ) -> None:
        """end_run should log final metrics if provided."""
        config = {"seed": 42}
        run = tmp_registry.start_run(config, run_type="simulation")

        tmp_registry.end_run(run, status="success", final_metrics={"final_loss": 0.01})

        metrics_path = run.run_dir / "metrics.json"
        with open(metrics_path) as f:
            metrics = json.load(f)

        assert metrics["final_loss"] == 0.01

    def test_disabled_registry_does_not_create_files(
        self, tmp_path: Path
    ) -> None:
        """Disabled registry should not create any files."""
        registry = RunRegistry(root_dir=tmp_path, enabled=False)
        config = {"seed": 42}

        run = registry.start_run(config, run_type="simulation")
        registry.log_metrics(run, {"loss": 0.1})
        registry.end_run(run, status="success")

        # No files should be created
        assert not run.run_dir.exists()

    def test_list_runs_returns_run_ids(
        self, tmp_registry: RunRegistry
    ) -> None:
        """list_runs should return run IDs."""
        config1 = {"seed": 42}
        config2 = {"seed": 99}

        run1 = tmp_registry.start_run(config1, run_type="simulation")
        tmp_registry.end_run(run1, status="success")

        run2 = tmp_registry.start_run(config2, run_type="validation")
        tmp_registry.end_run(run2, status="failed")

        runs = tmp_registry.list_runs()
        assert len(runs) == 2
        assert run1.run_id in runs
        assert run2.run_id in runs

    def test_list_runs_with_filter(
        self, tmp_registry: RunRegistry
    ) -> None:
        """list_runs should support filtering by type and status."""
        tmp_registry.start_run({"seed": 1}, run_type="simulation")
        run2 = tmp_registry.start_run({"seed": 2}, run_type="validation")
        tmp_registry.end_run(run2, status="success")

        # Filter by type
        sim_runs = tmp_registry.list_runs(run_type="simulation")
        assert len(sim_runs) == 1

        val_runs = tmp_registry.list_runs(run_type="validation")
        assert len(val_runs) == 1

    def test_get_run_loads_existing_run(
        self, tmp_registry: RunRegistry
    ) -> None:
        """get_run should load an existing run by ID."""
        config = {"seed": 42, "grid_size": 64}
        original_run = tmp_registry.start_run(config, run_type="simulation", seed=42)
        tmp_registry.end_run(original_run, status="success")

        # Load the run
        loaded_run = tmp_registry.get_run(original_run.run_id)

        assert loaded_run is not None
        assert loaded_run.run_id == original_run.run_id
        assert loaded_run.meta.run_type == "simulation"
        assert loaded_run.config["seed"] == 42

    def test_get_run_returns_none_for_nonexistent(
        self, tmp_registry: RunRegistry
    ) -> None:
        """get_run should return None for nonexistent run."""
        result = tmp_registry.get_run("nonexistent_run_id")
        assert result is None

    def test_dict_config_serialization(
        self, tmp_registry: RunRegistry
    ) -> None:
        """Registry should handle plain dict configs."""
        config = {
            "grid_size": 32,
            "steps": 50,
            "seed": 42,
            "alpha": 0.15,
            "nested": {"key": "value"},
        }

        run = tmp_registry.start_run(config, run_type="simulation")

        config_path = run.run_dir / "config.json"
        with open(config_path) as f:
            saved = json.load(f)

        assert saved["grid_size"] == 32
        assert saved["nested"]["key"] == "value"


class TestGlobalRegistry:
    """Test global registry functions."""

    def test_get_registry_returns_singleton(self) -> None:
        """get_registry should return the same instance."""
        reset_global_registry()

        r1 = get_registry()
        r2 = get_registry()

        assert r1 is r2

    def test_reset_global_registry(self) -> None:
        """reset_global_registry should create a new instance."""
        reset_global_registry()
        r1 = get_registry()

        reset_global_registry()
        r2 = get_registry()

        assert r1 is not r2


class TestEnvironmentVariable:
    """Test environment variable configuration."""

    def test_registry_respects_env_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Registry should be disabled when MFN_RUN_REGISTRY_ENABLED=false."""
        monkeypatch.setenv("MFN_RUN_REGISTRY_ENABLED", "false")

        registry = RunRegistry(root_dir=tmp_path)
        assert registry.enabled is False

    def test_registry_default_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Registry should be enabled by default."""
        monkeypatch.delenv("MFN_RUN_REGISTRY_ENABLED", raising=False)

        registry = RunRegistry(root_dir=tmp_path)
        assert registry.enabled is True
