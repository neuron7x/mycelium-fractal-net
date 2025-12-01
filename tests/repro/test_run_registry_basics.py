"""
Tests for RunRegistry functionality.

These tests verify the run registry creates proper directory structures
and files with expected content.
"""

import json
import tempfile

from mycelium_fractal_net.infra.run_registry import RunRegistry


class TestRunRegistry:
    """Tests for RunRegistry class."""

    def test_registry_creates_run_directory(self) -> None:
        """start_run creates run directory with expected structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RunRegistry(runs_dir=tmpdir, enabled=True)

            config = {"grid_size": 64, "steps": 100, "seed": 42}
            run = registry.start_run(config, run_type="test")

            assert run.run_dir.exists()
            assert (run.run_dir / "config.json").exists()
            assert (run.run_dir / "meta.json").exists()

    def test_config_json_content(self) -> None:
        """config.json contains the provided configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RunRegistry(runs_dir=tmpdir, enabled=True)

            config = {"grid_size": 64, "steps": 100, "alpha": 0.18, "seed": 42}
            run = registry.start_run(config, run_type="test")

            config_path = run.run_dir / "config.json"
            with open(config_path) as f:
                saved_config = json.load(f)

            assert saved_config["grid_size"] == 64
            assert saved_config["steps"] == 100
            assert saved_config["alpha"] == 0.18
            assert saved_config["seed"] == 42

    def test_meta_json_content(self) -> None:
        """meta.json contains expected metadata fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RunRegistry(runs_dir=tmpdir, enabled=True)

            config = {"seed": 42}
            run = registry.start_run(
                config,
                run_type="validation",
                seed=42,
                command="pytest",
                env="test",
            )

            meta_path = run.run_dir / "meta.json"
            with open(meta_path) as f:
                meta = json.load(f)

            # Required fields
            assert "run_id" in meta
            assert meta["run_type"] == "validation"
            assert "timestamp" in meta
            assert "git_commit" in meta
            assert meta["command"] == "pytest"
            assert meta["env"] == "test"
            assert meta["seed"] == 42
            assert meta["status"] == "running"

    def test_log_metrics(self) -> None:
        """log_metrics creates metrics.json with provided values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RunRegistry(runs_dir=tmpdir, enabled=True)

            config = {"seed": 42}
            run = registry.start_run(config, run_type="test")

            metrics = {"loss": 0.5, "accuracy": 0.95, "duration_s": 10.5}
            registry.log_metrics(run, metrics)

            metrics_path = run.run_dir / "metrics.json"
            with open(metrics_path) as f:
                saved_metrics = json.load(f)

            assert saved_metrics["loss"] == 0.5
            assert saved_metrics["accuracy"] == 0.95
            assert saved_metrics["duration_s"] == 10.5

    def test_log_metrics_append(self) -> None:
        """log_metrics appends to existing metrics by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RunRegistry(runs_dir=tmpdir, enabled=True)

            config = {"seed": 42}
            run = registry.start_run(config, run_type="test")

            registry.log_metrics(run, {"loss": 0.5})
            registry.log_metrics(run, {"accuracy": 0.95})

            metrics_path = run.run_dir / "metrics.json"
            with open(metrics_path) as f:
                saved_metrics = json.load(f)

            assert saved_metrics["loss"] == 0.5
            assert saved_metrics["accuracy"] == 0.95

    def test_end_run_updates_status(self) -> None:
        """end_run updates meta.json with final status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RunRegistry(runs_dir=tmpdir, enabled=True)

            config = {"seed": 42}
            run = registry.start_run(config, run_type="test")
            registry.end_run(run, status="success")

            meta_path = run.run_dir / "meta.json"
            with open(meta_path) as f:
                meta = json.load(f)

            assert meta["status"] == "success"
            assert "end_time" in meta

    def test_end_run_failed_status(self) -> None:
        """end_run can set failed status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RunRegistry(runs_dir=tmpdir, enabled=True)

            config = {"seed": 42}
            run = registry.start_run(config, run_type="test")
            registry.end_run(run, status="failed")

            meta_path = run.run_dir / "meta.json"
            with open(meta_path) as f:
                meta = json.load(f)

            assert meta["status"] == "failed"

    def test_get_run(self) -> None:
        """get_run retrieves run data by ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RunRegistry(runs_dir=tmpdir, enabled=True)

            config = {"grid_size": 64, "seed": 42}
            run = registry.start_run(config, run_type="test")
            registry.log_metrics(run, {"loss": 0.5})
            registry.end_run(run, status="success")

            data = registry.get_run(run.run_id)

            assert data is not None
            assert data["run_id"] == run.run_id
            assert "config" in data
            assert "meta" in data
            assert "metrics" in data
            assert data["config"]["grid_size"] == 64
            assert data["metrics"]["loss"] == 0.5
            assert data["meta"]["status"] == "success"

    def test_get_run_not_found(self) -> None:
        """get_run returns None for non-existent run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RunRegistry(runs_dir=tmpdir, enabled=True)
            data = registry.get_run("nonexistent_run_id")
            assert data is None

    def test_list_runs(self) -> None:
        """list_runs returns recent runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RunRegistry(runs_dir=tmpdir, enabled=True)

            # Create multiple runs
            run1 = registry.start_run({"seed": 1}, run_type="test")
            registry.end_run(run1, status="success")

            run2 = registry.start_run({"seed": 2}, run_type="validation")
            registry.end_run(run2, status="success")

            runs = registry.list_runs()

            assert len(runs) == 2
            # Most recent first
            assert runs[0]["run_id"] == run2.run_id
            assert runs[1]["run_id"] == run1.run_id

    def test_list_runs_filter_by_type(self) -> None:
        """list_runs can filter by run_type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RunRegistry(runs_dir=tmpdir, enabled=True)

            run1 = registry.start_run({"seed": 1}, run_type="test")
            registry.end_run(run1, status="success")

            run2 = registry.start_run({"seed": 2}, run_type="validation")
            registry.end_run(run2, status="success")

            validation_runs = registry.list_runs(run_type="validation")

            assert len(validation_runs) == 1
            assert validation_runs[0]["run_id"] == run2.run_id

    def test_disabled_registry_no_files(self) -> None:
        """Disabled registry doesn't create files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RunRegistry(runs_dir=tmpdir, enabled=False)

            config = {"seed": 42}
            run = registry.start_run(config, run_type="test")
            registry.log_metrics(run, {"loss": 0.5})
            registry.end_run(run, status="success")

            # No files should be created
            assert not run.run_dir.exists()

    def test_run_id_format(self) -> None:
        """Run ID follows expected format YYYYMMDD_HHMMSS_suffix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RunRegistry(runs_dir=tmpdir, enabled=True)

            config = {"seed": 42}
            run = registry.start_run(config, run_type="test")

            # Format: YYYYMMDD_HHMMSS_suffix
            parts = run.run_id.split("_")
            assert len(parts) == 3
            assert len(parts[0]) == 8  # YYYYMMDD
            assert len(parts[1]) == 6  # HHMMSS
            assert len(parts[2]) == 6  # suffix

    def test_seed_consistency(self) -> None:
        """Seed in RunHandle matches seed in meta.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RunRegistry(runs_dir=tmpdir, enabled=True)

            config = {"grid_size": 64, "seed": 42}
            run = registry.start_run(config, run_type="test", seed=42)

            assert run.seed == 42

            meta_path = run.run_dir / "meta.json"
            with open(meta_path) as f:
                meta = json.load(f)

            assert meta["seed"] == 42

    def test_config_seed_extraction(self) -> None:
        """Registry extracts seed from config when not explicitly provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RunRegistry(runs_dir=tmpdir, enabled=True)

            config = {"grid_size": 64, "seed": 123}
            run = registry.start_run(config, run_type="test")

            assert run.seed == 123
