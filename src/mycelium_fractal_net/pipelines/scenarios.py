"""
Scenario abstraction for MyceliumFractalNet data pipelines.

This module defines the ScenarioConfig dataclass and run_scenario function
that form the core of the MFN data generation pipeline.

Scenarios are domain-agnostic configurations that define:
- Input parameters for simulations
- Output format and location
- Scenario type (scientific, features, benchmark)

Reference: docs/MFN_DATA_PIPELINES.md
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

# Guard: ensure pandas is available with clear error message
try:
    import pandas as pd
except ModuleNotFoundError:
    raise RuntimeError(
        "pandas is required for dataset generation. "
        "Install with: pip install -r requirements.txt"
    )

from mycelium_fractal_net.core import ReactionDiffusionConfig, ReactionDiffusionEngine

logger = logging.getLogger(__name__)


class ScenarioType(str, Enum):
    """Type of scenario for data generation."""

    SCIENTIFIC = "scientific"
    FEATURES = "features"
    BENCHMARK = "benchmark"


@dataclass
class ScenarioConfig:
    """
    Configuration for a data generation scenario.

    Attributes
    ----------
    name : str
        Unique name for this scenario (e.g., 'scientific_baseline').
    scenario_type : ScenarioType
        Type of scenario: scientific, features, or benchmark.
    grid_size : int
        Size of simulation grid (NxN). Range: [8, 256].
    steps : int
        Number of simulation steps. Range: [1, 1000].
    num_samples : int
        Number of samples to generate in this scenario.
    seeds_per_config : int
        Number of random seeds per parameter configuration.
    base_seed : int
        Base random seed for reproducibility.
    alpha_values : list[float]
        Diffusion coefficients to sweep (CFL: < 0.25).
    turing_enabled : bool
        Enable Turing morphogenesis patterns.
    output_format : Literal['parquet', 'csv']
        Output file format.
    output_dir : str
        Output directory relative to data/.
    description : str
        Human-readable description of this scenario.
    """

    name: str
    scenario_type: ScenarioType = ScenarioType.FEATURES
    grid_size: int = 64
    steps: int = 100
    num_samples: int = 100
    seeds_per_config: int = 3
    base_seed: int = 42
    alpha_values: list[float] = field(default_factory=lambda: [0.10, 0.15, 0.20])
    turing_enabled: bool = True
    output_format: Literal["parquet", "csv"] = "parquet"
    output_dir: str = "scenarios"
    description: str = ""

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.grid_size < 8 or self.grid_size > 256:
            raise ValueError(f"grid_size must be in [8, 256], got {self.grid_size}")
        if self.steps < 1 or self.steps > 1000:
            raise ValueError(f"steps must be in [1, 1000], got {self.steps}")
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {self.num_samples}")
        if self.seeds_per_config < 1:
            raise ValueError(
                f"seeds_per_config must be >= 1, got {self.seeds_per_config}"
            )
        for alpha in self.alpha_values:
            if alpha <= 0 or alpha >= 0.25:
                raise ValueError(
                    f"alpha must be in (0, 0.25) for CFL stability, got {alpha}"
                )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize configuration to a dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the configuration.
        """
        return {
            "name": self.name,
            "scenario_type": self.scenario_type.value,
            "grid_size": self.grid_size,
            "steps": self.steps,
            "num_samples": self.num_samples,
            "seeds_per_config": self.seeds_per_config,
            "base_seed": self.base_seed,
            "alpha_values": self.alpha_values,
            "turing_enabled": self.turing_enabled,
            "output_format": self.output_format,
            "output_dir": self.output_dir,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioConfig":
        """
        Create configuration from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary with configuration values.

        Returns
        -------
        ScenarioConfig
            New configuration instance.
        """
        scenario_type = data.get("scenario_type", "features")
        if isinstance(scenario_type, str):
            scenario_type = ScenarioType(scenario_type)

        return cls(
            name=data["name"],
            scenario_type=scenario_type,
            grid_size=data.get("grid_size", 64),
            steps=data.get("steps", 100),
            num_samples=data.get("num_samples", 100),
            seeds_per_config=data.get("seeds_per_config", 3),
            base_seed=data.get("base_seed", 42),
            alpha_values=data.get("alpha_values", [0.10, 0.15, 0.20]),
            turing_enabled=data.get("turing_enabled", True),
            output_format=data.get("output_format", "parquet"),
            output_dir=data.get("output_dir", "scenarios"),
            description=data.get("description", ""),
        )


@dataclass
class DatasetMeta:
    """
    Metadata about a generated dataset.

    Attributes
    ----------
    scenario_name : str
        Name of the scenario that generated this dataset.
    output_path : Path
        Full path to the output file.
    num_rows : int
        Number of rows in the dataset.
    num_columns : int
        Number of columns in the dataset.
    elapsed_seconds : float
        Time taken to generate the dataset.
    timestamp : str
        ISO format timestamp of generation.
    feature_names : list[str]
        List of feature column names.
    """

    scenario_name: str
    output_path: Path
    num_rows: int
    num_columns: int
    elapsed_seconds: float
    timestamp: str
    feature_names: list[str]


# =============================================================================
# Preset Configurations
# =============================================================================


def _make_small_scientific() -> ScenarioConfig:
    """Create small scientific scenario for quick validation tests."""
    return ScenarioConfig(
        name="scientific_small",
        scenario_type=ScenarioType.SCIENTIFIC,
        grid_size=32,
        steps=50,
        num_samples=10,
        seeds_per_config=2,
        base_seed=42,
        alpha_values=[0.15],
        turing_enabled=True,
        output_format="parquet",
        output_dir="scenarios/scientific_small",
        description="Small scientific scenario for quick validation (Nernst/Turing/STDP).",
    )


def _make_medium_features() -> ScenarioConfig:
    """Create medium feature-generation scenario."""
    return ScenarioConfig(
        name="features_medium",
        scenario_type=ScenarioType.FEATURES,
        grid_size=64,
        steps=100,
        num_samples=100,
        seeds_per_config=3,
        base_seed=42,
        alpha_values=[0.10, 0.15, 0.20],
        turing_enabled=True,
        output_format="parquet",
        output_dir="scenarios/features_medium",
        description="Medium feature-generation scenario for ML training.",
    )


def _make_large_features() -> ScenarioConfig:
    """Create large feature-generation scenario."""
    return ScenarioConfig(
        name="features_large",
        scenario_type=ScenarioType.FEATURES,
        grid_size=128,
        steps=200,
        num_samples=500,
        seeds_per_config=5,
        base_seed=42,
        alpha_values=[0.08, 0.12, 0.16, 0.20],
        turing_enabled=True,
        output_format="parquet",
        output_dir="scenarios/features_large",
        description="Large feature-generation scenario for production ML.",
    )


def _make_benchmark_small() -> ScenarioConfig:
    """Create small benchmark scenario."""
    return ScenarioConfig(
        name="benchmark_small",
        scenario_type=ScenarioType.BENCHMARK,
        grid_size=32,
        steps=32,
        num_samples=5,
        seeds_per_config=1,
        base_seed=42,
        alpha_values=[0.15],
        turing_enabled=True,
        output_format="parquet",
        output_dir="scenarios/benchmark_small",
        description="Small benchmark scenario for performance testing.",
    )


# Preset registry
_PRESETS: dict[str, ScenarioConfig] = {}


def _init_presets() -> None:
    """Initialize preset configurations."""
    global _PRESETS
    _PRESETS = {
        "small": _make_small_scientific(),
        "medium": _make_medium_features(),
        "large": _make_large_features(),
        "benchmark": _make_benchmark_small(),
    }


_init_presets()


def get_preset_config(preset_name: str) -> ScenarioConfig:
    """
    Get a preset scenario configuration by name.

    Parameters
    ----------
    preset_name : str
        Name of the preset: 'small', 'medium', 'large', or 'benchmark'.

    Returns
    -------
    ScenarioConfig
        A copy of the preset configuration.

    Raises
    ------
    ValueError
        If preset_name is not recognized.
    """
    if preset_name not in _PRESETS:
        available = list(_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

    # Return a fresh copy to avoid mutation
    original = _PRESETS[preset_name]
    return ScenarioConfig(
        name=original.name,
        scenario_type=original.scenario_type,
        grid_size=original.grid_size,
        steps=original.steps,
        num_samples=original.num_samples,
        seeds_per_config=original.seeds_per_config,
        base_seed=original.base_seed,
        alpha_values=list(original.alpha_values),
        turing_enabled=original.turing_enabled,
        output_format=original.output_format,
        output_dir=original.output_dir,
        description=original.description,
    )


def list_presets() -> list[str]:
    """
    List available preset names.

    Returns
    -------
    list[str]
        List of preset names.
    """
    return list(_PRESETS.keys())


# =============================================================================
# Scenario Execution
# =============================================================================


def _generate_param_configs(config: ScenarioConfig) -> list[dict[str, Any]]:
    """Generate all parameter combinations for the scenario."""
    configs: list[dict[str, Any]] = []
    sim_id = 0

    # Generate combinations up to num_samples
    while len(configs) < config.num_samples:
        for alpha in config.alpha_values:
            for seed_offset in range(config.seeds_per_config):
                if len(configs) >= config.num_samples:
                    break
                seed = config.base_seed + sim_id
                configs.append(
                    {
                        "sim_id": sim_id,
                        "scenario_name": config.name,
                        "grid_size": config.grid_size,
                        "steps": config.steps,
                        "alpha": alpha,
                        "turing_enabled": config.turing_enabled,
                        "random_seed": seed,
                    }
                )
                sim_id += 1
            if len(configs) >= config.num_samples:
                break

    return configs


def _run_single_simulation(
    params: dict[str, Any],
) -> tuple[NDArray[Any], dict[str, Any]] | None:
    """Run a single simulation with given parameters."""
    try:
        rd_config = ReactionDiffusionConfig(
            grid_size=params["grid_size"],
            alpha=params["alpha"],
            random_seed=params["random_seed"],
        )
        engine = ReactionDiffusionEngine(rd_config)

        history, metrics = engine.simulate(
            steps=params["steps"],
            turing_enabled=params["turing_enabled"],
            return_history=True,
        )

        metadata = {
            "growth_events": metrics.growth_events,
            "turing_activations": metrics.turing_activations,
            "clamping_events": metrics.clamping_events,
        }

        return history, metadata

    except Exception as e:
        logger.warning(f"Simulation failed for params {params}: {e}")
        return None


def _atomic_write(df: Any, output_path: Path, output_format: str) -> None:
    """
    Write dataframe atomically to avoid corrupt files.

    Writes to a temporary file first, then renames to final destination.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory for atomic rename
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=output_path.parent,
        suffix=f".tmp.{output_format}",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        if output_format == "parquet":
            df.to_parquet(tmp_path, index=False)
        else:  # csv
            df.to_csv(tmp_path, index=False)

        # Atomic rename
        shutil.move(str(tmp_path), str(output_path))
        logger.info(f"Dataset saved to {output_path}")

    except Exception:
        # Clean up temp file on failure
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def run_scenario(
    config: ScenarioConfig,
    data_root: Path | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> DatasetMeta:
    """
    Run a scenario and generate a dataset.

    Parameters
    ----------
    config : ScenarioConfig
        Scenario configuration.
    data_root : Path | None
        Root directory for data output. Defaults to 'data/' in project root.
    rng : np.random.Generator | None
        Optional random generator for reproducibility.

    Returns
    -------
    DatasetMeta
        Metadata about the generated dataset.
    """
    # Import here to avoid circular imports
    from analytics import FeatureConfig, FeatureVector, compute_features

    # Determine output path
    if data_root is None:
        data_root = Path("data")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = data_root / config.output_dir / timestamp
    output_filename = f"dataset.{config.output_format}"
    output_path = output_dir / output_filename

    # Generate parameter configurations
    param_configs = _generate_param_configs(config)
    n_configs = len(param_configs)

    logger.info(f"Starting scenario '{config.name}' with {n_configs} configurations")

    # Feature extraction config
    feature_config = FeatureConfig()

    # Storage for results
    all_rows: list[dict[str, Any]] = []
    n_success = 0
    n_failed = 0

    start_time = time.time()

    for i, params in enumerate(param_configs):
        if (i + 1) % max(1, n_configs // 10) == 0 or i == 0:
            logger.info(f"Processing {i + 1}/{n_configs}...")

        result = _run_single_simulation(params)
        if result is None:
            n_failed += 1
            continue

        history, sim_meta = result

        # Extract features
        try:
            features = compute_features(history, feature_config)
        except Exception as e:
            logger.warning(
                f"Feature extraction failed for sim_id={params['sim_id']}: {e}"
            )
            n_failed += 1
            continue

        # Build row
        row = {
            **params,
            **features.to_dict(),
            **sim_meta,
        }
        all_rows.append(row)
        n_success += 1

    elapsed = time.time() - start_time

    # Create and save dataset
    if not all_rows:
        raise RuntimeError(
            f"No successful simulations for scenario '{config.name}' "
            f"({n_failed}/{n_configs} failed)"
        )

    df = pd.DataFrame(all_rows)
    _atomic_write(df, output_path, config.output_format)

    # Get feature names
    feature_names = FeatureVector.feature_names()

    logger.info(
        f"Scenario '{config.name}' complete: "
        f"{n_success}/{n_configs} successful ({elapsed:.1f}s)"
    )

    return DatasetMeta(
        scenario_name=config.name,
        output_path=output_path,
        num_rows=len(df),
        num_columns=len(df.columns),
        elapsed_seconds=elapsed,
        timestamp=timestamp,
        feature_names=feature_names,
    )
