"""
Run Registry for MyceliumFractalNet.

Provides a lightweight, file-based registry for tracking experiment runs.
Each run creates a directory with configuration, metadata, and metrics files.

The registry is designed for:
- Local development and experiments
- CI/CD pipeline artifact generation
- Reproducibility audit trails

Usage:
    >>> from mycelium_fractal_net.infra.run_registry import RunRegistry
    >>> registry = RunRegistry()
    >>> run = registry.start_run(config=my_config, run_type="validation")
    >>> # ... run experiment ...
    >>> registry.log_metrics(run, {"loss": 0.5, "accuracy": 0.95})
    >>> registry.end_run(run, status="success")

Environment Variables:
    MFN_RUN_REGISTRY_ENABLED: Set to "false" or "0" to disable registry.
    MFN_RUN_REGISTRY_DIR: Override default runs directory.

Reference: docs/MFN_REPRODUCIBILITY.md
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class RunStatus(str, Enum):
    """Status of a run."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RunHandle:
    """
    Handle to an active run.

    Holds the run identifier and directory path for logging.

    Attributes:
        run_id: Unique identifier for this run (YYYYMMDD_HHMMSS_shortid).
        run_dir: Path to the run's directory.
        run_type: Type of run (e.g., "validation", "benchmark", "experiment").
        start_time: ISO format timestamp when run started.
        config: Configuration dictionary for this run.
        seed: Random seed used for this run.
    """

    run_id: str
    run_dir: Path
    run_type: str
    start_time: str
    config: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None


class RunRegistry:
    """
    File-based run registry for tracking experiment metadata.

    Creates a directory structure:
        runs/
        └── YYYYMMDD_HHMMSS_<shortid>/
            ├── config.json     # Full configuration
            ├── meta.json       # Run metadata
            └── metrics.json    # Key metrics

    The registry can be disabled via environment variable for CI or production.

    Attributes:
        runs_dir: Root directory for run storage.
        enabled: Whether registry is active.
    """

    DEFAULT_RUNS_DIR = "runs"

    def __init__(
        self,
        runs_dir: Path | str | None = None,
        enabled: bool | None = None,
    ) -> None:
        """
        Initialize run registry.

        Args:
            runs_dir: Directory for storing runs. Defaults to 'runs/' in
                     project root or MFN_RUN_REGISTRY_DIR env var.
            enabled: Enable/disable registry. Defaults to True unless
                    MFN_RUN_REGISTRY_ENABLED is set to 'false' or '0'.
        """
        # Determine enabled state
        if enabled is None:
            env_enabled = os.environ.get("MFN_RUN_REGISTRY_ENABLED", "true")
            self.enabled = env_enabled.lower() not in ("false", "0", "no", "off")
        else:
            self.enabled = enabled

        # Determine runs directory
        if runs_dir is not None:
            self.runs_dir = Path(runs_dir)
        else:
            env_dir = os.environ.get("MFN_RUN_REGISTRY_DIR")
            if env_dir:
                self.runs_dir = Path(env_dir)
            else:
                self.runs_dir = Path(self.DEFAULT_RUNS_DIR)

    def _get_git_commit(self) -> str:
        """Get current git commit hash, or 'unknown' if not in a git repo."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            pass
        return "unknown"

    def _generate_run_id(self) -> str:
        """Generate unique run ID with timestamp and short random suffix."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        import random

        suffix = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
        return f"{timestamp}_{suffix}"

    def start_run(
        self,
        config: dict[str, Any] | Any,
        run_type: str = "experiment",
        seed: int | None = None,
        command: str | None = None,
        env: str = "dev",
    ) -> RunHandle:
        """
        Start a new run and create its directory structure.

        Args:
            config: Configuration dictionary or dataclass with to_dict() method.
            run_type: Type of run (e.g., "validation", "benchmark", "experiment").
            seed: Random seed used for this run.
            command: Optional command string that started this run.
            env: Environment name (dev/stage/prod).

        Returns:
            RunHandle for logging metrics and ending the run.
        """
        # Convert config to dict if it's a dataclass
        config_dict: dict[str, Any]
        if hasattr(config, "to_dict"):
            config_dict = config.to_dict()
        elif hasattr(config, "__dataclass_fields__"):
            config_dict = asdict(config)  # type: ignore[arg-type]
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = {"raw": str(config)}

        run_id = self._generate_run_id()
        start_time = datetime.now().isoformat()

        # Extract seed from config if not provided
        if seed is None:
            seed = config_dict.get("seed") or config_dict.get("base_seed")

        handle = RunHandle(
            run_id=run_id,
            run_dir=self.runs_dir / run_id,
            run_type=run_type,
            start_time=start_time,
            config=config_dict,
            seed=seed,
        )

        if not self.enabled:
            return handle

        # Create run directory
        handle.run_dir.mkdir(parents=True, exist_ok=True)

        # Write config.json
        config_path = handle.run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

        # Write meta.json
        meta = {
            "run_id": run_id,
            "run_type": run_type,
            "timestamp": start_time,
            "git_commit": self._get_git_commit(),
            "command": command or "",
            "env": env,
            "seed": seed,
            "status": RunStatus.RUNNING.value,
        }
        meta_path = handle.run_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return handle

    def log_metrics(
        self,
        run: RunHandle,
        metrics: dict[str, Any],
        *,
        append: bool = True,
    ) -> None:
        """
        Log metrics for a run.

        Args:
            run: RunHandle from start_run().
            metrics: Dictionary of metric names to values.
            append: If True, merge with existing metrics. If False, overwrite.
        """
        if not self.enabled:
            return

        metrics_path = run.run_dir / "metrics.json"

        existing: dict[str, Any] = {}
        if append and metrics_path.exists():
            with open(metrics_path) as f:
                existing = json.load(f)

        existing.update(metrics)

        with open(metrics_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)

    def end_run(
        self,
        run: RunHandle,
        status: Literal["success", "failed", "cancelled"] = "success",
    ) -> None:
        """
        End a run and update its final status.

        Args:
            run: RunHandle from start_run().
            status: Final status of the run.
        """
        if not self.enabled:
            return

        meta_path = run.run_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}

        meta["status"] = status
        meta["end_time"] = datetime.now().isoformat()

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """
        Get run data by ID.

        Args:
            run_id: Run identifier.

        Returns:
            Dictionary with config, meta, and metrics, or None if not found.
        """
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            return None

        result: dict[str, Any] = {"run_id": run_id}

        config_path = run_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                result["config"] = json.load(f)

        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                result["meta"] = json.load(f)

        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                result["metrics"] = json.load(f)

        return result

    def list_runs(
        self,
        run_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        List recent runs.

        Args:
            run_type: Optional filter by run type.
            limit: Maximum number of runs to return.

        Returns:
            List of run data dictionaries, most recent first.
        """
        if not self.runs_dir.exists():
            return []

        runs = []
        for run_dir in sorted(self.runs_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue

            meta_path = run_dir / "meta.json"
            if not meta_path.exists():
                continue

            with open(meta_path) as f:
                meta = json.load(f)

            if run_type and meta.get("run_type") != run_type:
                continue

            runs.append({
                "run_id": run_dir.name,
                "meta": meta,
            })

            if len(runs) >= limit:
                break

        return runs


__all__ = [
    "RunStatus",
    "RunHandle",
    "RunRegistry",
]
