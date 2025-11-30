"""
Run Registry for MyceliumFractalNet.

This module provides a lightweight, file-based run registry for tracking
experimental runs, simulations, validations, and benchmarks. Each run
creates a directory with config, metadata, and metrics files.

Features:
- File-based storage (no database required)
- Git commit tracking
- Configurable via environment variables
- Support for CI/CD and local development

Usage:
    >>> from mycelium_fractal_net.infra.run_registry import RunRegistry
    >>> registry = RunRegistry()
    >>> run = registry.start_run(config=my_config, run_type="simulation")
    >>> # ... execute run ...
    >>> registry.log_metrics(run, {"loss": 0.1, "accuracy": 0.95})
    >>> registry.end_run(run, status="success")

Environment Variables:
    MFN_RUN_REGISTRY_ENABLED: Set to "false" to disable registry (default: "true")
    MFN_RUN_REGISTRY_DIR: Override default runs directory

Reference:
    - docs/MFN_REPRODUCIBILITY.md — Reproducibility documentation
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Protocol, Union

# Type alias for run status
RunStatus = Literal["running", "success", "failed", "cancelled"]


class ConfigLike(Protocol):
    """Protocol for objects that can be serialized to dict."""

    def to_dict(self) -> Dict[str, Any]:
        ...


@dataclass
class RunMeta:
    """
    Metadata for a single run.

    Attributes
    ----------
    run_id : str
        Unique identifier for the run (UUID-based).
    run_type : str
        Type of run (e.g., "simulation", "validation", "benchmark", "experiment").
    timestamp : str
        ISO format timestamp when the run started.
    git_commit : str | None
        Git commit hash at the time of the run.
    command : str | None
        Command line that started the run.
    env : str
        Environment (dev, staging, prod).
    seed : int | None
        Random seed used for the run.
    status : RunStatus
        Current status of the run.
    end_timestamp : str | None
        ISO format timestamp when the run ended.
    duration_seconds : float | None
        Total duration of the run in seconds.
    """

    run_id: str
    run_type: str
    timestamp: str
    git_commit: Optional[str] = None
    command: Optional[str] = None
    env: str = "dev"
    seed: Optional[int] = None
    status: RunStatus = "running"
    end_timestamp: Optional[str] = None
    duration_seconds: Optional[float] = None


@dataclass
class RunHandle:
    """
    Handle to an active or completed run.

    This object is returned by `RunRegistry.start_run()` and should be passed
    to `log_metrics()` and `end_run()`.

    Attributes
    ----------
    run_id : str
        Unique identifier for the run.
    run_dir : Path
        Directory where run artifacts are stored.
    meta : RunMeta
        Metadata about the run.
    config : Dict[str, Any]
        Configuration used for the run.
    _start_time : datetime
        Internal timestamp for duration calculation.
    """

    run_id: str
    run_dir: Path
    meta: RunMeta
    config: Dict[str, Any]
    _start_time: datetime = field(default_factory=datetime.now)


class RunRegistry:
    """
    File-based registry for tracking experimental runs.

    Creates a directory structure:
        runs/
        └── YYYYMMDD_HHMMSS_<shortid>/
            ├── config.json     # Full configuration
            ├── meta.json       # Run metadata
            └── metrics.json    # Logged metrics (optional)

    Parameters
    ----------
    root_dir : Path | str | None
        Root directory for storing runs. Defaults to "runs/" in project root.
    enabled : bool | None
        Whether the registry is enabled. If None, reads from
        MFN_RUN_REGISTRY_ENABLED environment variable.

    Example
    -------
    >>> registry = RunRegistry(root_dir=Path("./runs"))
    >>> run = registry.start_run({"seed": 42, "grid_size": 64}, run_type="simulation")
    >>> registry.log_metrics(run, {"loss": 0.123})
    >>> registry.end_run(run, status="success")
    """

    def __init__(
        self,
        root_dir: Optional[Union[Path, str]] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        # Check environment variable for enabled status
        if enabled is None:
            env_enabled = os.environ.get("MFN_RUN_REGISTRY_ENABLED", "true").lower()
            enabled = env_enabled not in ("false", "0", "no", "off")

        self.enabled = enabled

        # Determine root directory
        if root_dir is None:
            env_dir = os.environ.get("MFN_RUN_REGISTRY_DIR")
            if env_dir:
                root_dir = Path(env_dir)
            else:
                # Default to runs/ in project root
                root_dir = Path("runs")

        self.root_dir = Path(root_dir)

    def _get_git_commit(self) -> Optional[str]:
        """Get the current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return None

    def _get_git_commit_full(self) -> Optional[str]:
        """Get the full current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return None

    def _config_to_dict(self, config: Any) -> Dict[str, Any]:
        """Convert a config object to a dictionary."""
        if config is None:
            return {}

        if isinstance(config, dict):
            return config

        # Check for to_dict method (ConfigLike protocol)
        if hasattr(config, "to_dict") and callable(config.to_dict):
            result = config.to_dict()
            return result if isinstance(result, dict) else {}

        # Check for model_dump (Pydantic v2)
        if hasattr(config, "model_dump") and callable(config.model_dump):
            result = config.model_dump()
            return result if isinstance(result, dict) else {}

        # Check for dict method (Pydantic v1)
        if hasattr(config, "dict") and callable(config.dict):
            result = config.dict()
            return result if isinstance(result, dict) else {}

        # Handle dataclass
        if is_dataclass(config) and not isinstance(config, type):
            return asdict(config)

        # Last resort: try __dict__
        if hasattr(config, "__dict__"):
            return dict(config.__dict__)

        raise TypeError(f"Cannot convert {type(config).__name__} to dict")

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert an object to JSON-serializable form."""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        if isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {str(k): self._make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, "__dict__"):
            return self._make_json_serializable(obj.__dict__)
        return str(obj)

    def start_run(
        self,
        config: Any,
        run_type: str = "simulation",
        seed: Optional[int] = None,
        command: Optional[str] = None,
    ) -> RunHandle:
        """
        Start a new run and create the run directory.

        Parameters
        ----------
        config : Any
            Configuration for the run. Can be a dict, dataclass, or
            object with to_dict()/model_dump() method.
        run_type : str
            Type of run: "simulation", "validation", "benchmark", "experiment", etc.
        seed : int | None
            Random seed used for the run (extracted from config if not provided).
        command : str | None
            Command line that started the run.

        Returns
        -------
        RunHandle
            Handle to the started run.
        """
        # Generate unique run ID
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        run_id = f"{timestamp_str}_{short_id}"

        # Convert config to dict
        config_dict = self._config_to_dict(config)
        config_dict = self._make_json_serializable(config_dict)

        # Extract seed from config if not provided
        if seed is None:
            seed = config_dict.get("seed") or config_dict.get("base_seed")

        # Determine environment
        env = os.environ.get("MFN_ENV", "dev")

        # Create metadata
        meta = RunMeta(
            run_id=run_id,
            run_type=run_type,
            timestamp=timestamp.isoformat(),
            git_commit=self._get_git_commit_full(),
            command=command,
            env=env,
            seed=seed,
            status="running",
        )

        # Create run directory and files if enabled
        run_dir = self.root_dir / run_id

        if self.enabled:
            run_dir.mkdir(parents=True, exist_ok=True)

            # Save config
            config_path = run_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)

            # Save metadata
            meta_path = run_dir / "meta.json"
            with open(meta_path, "w") as f:
                json.dump(asdict(meta), f, indent=2)

        return RunHandle(
            run_id=run_id,
            run_dir=run_dir,
            meta=meta,
            config=config_dict,
            _start_time=timestamp,
        )

    def log_metrics(
        self,
        run: RunHandle,
        metrics: Dict[str, Any],
        append: bool = True,
    ) -> None:
        """
        Log metrics for a run.

        Parameters
        ----------
        run : RunHandle
            Handle to the run.
        metrics : Dict[str, Any]
            Metrics to log (key-value pairs).
        append : bool
            If True, append to existing metrics. If False, overwrite.
        """
        if not self.enabled:
            return

        metrics_path = run.run_dir / "metrics.json"

        # Load existing metrics if appending
        existing_metrics: Dict[str, Any] = {}
        if append and metrics_path.exists():
            with open(metrics_path, "r") as f:
                existing_metrics = json.load(f)

        # Merge metrics
        existing_metrics.update(self._make_json_serializable(metrics))

        # Save metrics
        with open(metrics_path, "w") as f:
            json.dump(existing_metrics, f, indent=2)

    def end_run(
        self,
        run: RunHandle,
        status: RunStatus = "success",
        final_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        End a run and update its status.

        Parameters
        ----------
        run : RunHandle
            Handle to the run.
        status : RunStatus
            Final status: "success", "failed", or "cancelled".
        final_metrics : Dict[str, Any] | None
            Optional final metrics to log.
        """
        if not self.enabled:
            return

        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - run._start_time).total_seconds()

        # Update metadata
        run.meta.status = status
        run.meta.end_timestamp = end_time.isoformat()
        run.meta.duration_seconds = duration

        # Save updated metadata
        meta_path = run.run_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(asdict(run.meta), f, indent=2)

        # Log final metrics if provided
        if final_metrics:
            self.log_metrics(run, final_metrics)

    def get_run(self, run_id: str) -> Optional[RunHandle]:
        """
        Load an existing run by ID.

        Parameters
        ----------
        run_id : str
            The run ID to load.

        Returns
        -------
        RunHandle | None
            The run handle if found, None otherwise.
        """
        run_dir = self.root_dir / run_id
        if not run_dir.exists():
            return None

        # Load metadata
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            return None

        with open(meta_path, "r") as f:
            meta_dict = json.load(f)

        meta = RunMeta(**meta_dict)

        # Load config
        config_path = run_dir / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)

        return RunHandle(
            run_id=run_id,
            run_dir=run_dir,
            meta=meta,
            config=config,
        )

    def list_runs(
        self,
        run_type: Optional[str] = None,
        status: Optional[RunStatus] = None,
        limit: int = 100,
    ) -> list[str]:
        """
        List run IDs matching the given filters.

        Parameters
        ----------
        run_type : str | None
            Filter by run type.
        status : RunStatus | None
            Filter by status.
        limit : int
            Maximum number of runs to return.

        Returns
        -------
        list[str]
            List of run IDs (most recent first).
        """
        if not self.root_dir.exists():
            return []

        runs: list[tuple[str, datetime]] = []

        for run_dir in self.root_dir.iterdir():
            if not run_dir.is_dir():
                continue

            meta_path = run_dir / "meta.json"
            if not meta_path.exists():
                continue

            try:
                with open(meta_path, "r") as f:
                    meta_dict = json.load(f)

                # Apply filters
                if run_type and meta_dict.get("run_type") != run_type:
                    continue
                if status and meta_dict.get("status") != status:
                    continue

                # Parse timestamp for sorting
                ts = datetime.fromisoformat(meta_dict["timestamp"])
                runs.append((run_dir.name, ts))

            except (json.JSONDecodeError, KeyError, ValueError):
                continue

        # Sort by timestamp (most recent first) and limit
        runs.sort(key=lambda x: x[1], reverse=True)
        return [run_id for run_id, _ in runs[:limit]]


# Global registry instance (lazy initialization)
_global_registry: Optional[RunRegistry] = None


def get_registry() -> RunRegistry:
    """
    Get the global run registry instance.

    Returns
    -------
    RunRegistry
        The global registry instance.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = RunRegistry()
    return _global_registry


def reset_global_registry() -> None:
    """Reset the global registry instance (useful for testing)."""
    global _global_registry
    _global_registry = None


__all__ = [
    "RunRegistry",
    "RunHandle",
    "RunMeta",
    "RunStatus",
    "get_registry",
    "reset_global_registry",
]
