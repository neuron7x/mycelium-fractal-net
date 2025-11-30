"""
Dataset Generation Pipeline for MyceliumFractalNet.

Generates experimental datasets by running simulations with parameter sweeps
and extracting features as defined in FEATURE_SCHEMA.md.

Usage (scenario-based - recommended):
    python -m experiments.generate_dataset --preset small
    python -m experiments.generate_dataset --preset medium
    python -m experiments.generate_dataset --preset large

Usage (legacy sweep mode):
    python -m experiments.generate_dataset --output data/mycelium_dataset.parquet --sweep default

Features:
- Scenario-based data generation with small/medium/large presets
- Parameter sweeps across stable ranges from MFN_MATH_MODEL.md
- Reproducible via random_seed control
- Atomic file writes for data integrity
- Logging of metrics and failures
- Output in Parquet format

Reference: docs/MFN_DATA_PIPELINES.md
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analytics import FeatureConfig, FeatureVector, compute_features
from mycelium_fractal_net.core import (
    ReactionDiffusionConfig,
    ReactionDiffusionEngine,
)
from mycelium_fractal_net.pipelines import (
    get_preset_config,
    list_presets,
    run_scenario,
)

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# === Parameter Sweep Configuration ===
@dataclass
class SweepConfig:
    """Configuration for parameter sweep."""

    # Grid sizes to test
    grid_sizes: list[int] | None = None
    # Simulation steps
    steps_list: list[int] | None = None
    # Diffusion coefficients (must be < 0.25 for CFL stability)
    alpha_values: list[float] | None = None
    # Turing enabled/disabled
    turing_values: list[bool] | None = None
    # Number of random seeds per configuration
    seeds_per_config: int = 3
    # Base random seed for reproducibility
    base_seed: int = 42

    def __post_init__(self) -> None:
        if self.grid_sizes is None:
            self.grid_sizes = [32, 64]
        if self.steps_list is None:
            self.steps_list = [50, 100]
        if self.alpha_values is None:
            self.alpha_values = [0.10, 0.15, 0.20]
        if self.turing_values is None:
            self.turing_values = [True, False]


def _create_default_sweep() -> SweepConfig:
    """Create default sweep configuration."""
    return SweepConfig(
        grid_sizes=[32, 64],
        steps_list=[50, 100],
        alpha_values=[0.10, 0.15, 0.20],
        turing_values=[True, False],
        seeds_per_config=3,
        base_seed=42,
    )


def _create_minimal_sweep() -> SweepConfig:
    """Create minimal sweep for quick testing."""
    return SweepConfig(
        grid_sizes=[32],
        steps_list=[50],
        alpha_values=[0.15],
        turing_values=[True],
        seeds_per_config=2,
        base_seed=42,
    )


def _create_extended_sweep() -> SweepConfig:
    """Create extended sweep for comprehensive dataset."""
    return SweepConfig(
        grid_sizes=[32, 64, 128],
        steps_list=[50, 100, 200],
        alpha_values=[0.08, 0.12, 0.16, 0.20],
        turing_values=[True, False],
        seeds_per_config=5,
        base_seed=42,
    )


def generate_parameter_configs(
    sweep: SweepConfig,
) -> list[dict[str, Any]]:
    """
    Generate all parameter configurations for sweep.

    Parameters
    ----------
    sweep : SweepConfig
        Sweep configuration.

    Returns
    -------
    list[dict]
        List of parameter dictionaries.
    """
    configs = []
    sim_id = 0

    for grid_size in sweep.grid_sizes:
        for steps in sweep.steps_list:
            for alpha in sweep.alpha_values:
                for turing in sweep.turing_values:
                    for seed_offset in range(sweep.seeds_per_config):
                        seed = sweep.base_seed + sim_id
                        configs.append(
                            {
                                "sim_id": sim_id,
                                "grid_size": grid_size,
                                "steps": steps,
                                "alpha": alpha,
                                "turing_enabled": turing,
                                "random_seed": seed,
                            }
                        )
                        sim_id += 1

    return configs


def run_simulation(
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """
    Run a single simulation with given parameters.

    Parameters
    ----------
    params : dict
        Simulation parameters.

    Returns
    -------
    tuple | None
        (field_history, metadata) or None if failed.
    """
    try:
        config = ReactionDiffusionConfig(
            grid_size=params["grid_size"],
            alpha=params["alpha"],
            random_seed=params["random_seed"],
        )
        engine = ReactionDiffusionEngine(config)

        # Run simulation with history
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


def generate_dataset(
    sweep: SweepConfig,
    output_path: Path | None = None,
    feature_config: FeatureConfig | None = None,
) -> dict[str, Any]:
    """
    Generate complete dataset with parameter sweep.

    Parameters
    ----------
    sweep : SweepConfig
        Sweep configuration.
    output_path : Path | None
        Output path for parquet file.
    feature_config : FeatureConfig | None
        Feature extraction configuration.

    Returns
    -------
    dict
        Dataset statistics and metadata.
    """
    if feature_config is None:
        feature_config = FeatureConfig()

    configs = generate_parameter_configs(sweep)
    n_configs = len(configs)

    logger.info(f"Starting dataset generation with {n_configs} configurations")

    # Storage for results
    all_rows: list[dict[str, Any]] = []
    n_success = 0
    n_failed = 0

    start_time = time.time()

    for i, params in enumerate(configs):
        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"Processing {i + 1}/{n_configs}...")

        result = run_simulation(params)
        if result is None:
            n_failed += 1
            continue

        history, sim_meta = result

        # Extract features
        try:
            features = compute_features(history, feature_config)
        except Exception as e:
            logger.warning(f"Feature extraction failed for sim_id={params['sim_id']}: {e}")
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

    # Create dataset
    if all_rows:
        try:
            import pandas as pd

            df = pd.DataFrame(all_rows)

            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(output_path, index=False)
                logger.info(f"Dataset saved to {output_path}")

        except ImportError:
            logger.warning("pandas not installed, saving as npz instead")
            if output_path is not None and all_rows:
                npz_path = output_path.with_suffix(".npz")
                np.savez(
                    npz_path,
                    data=np.array([list(r.values()) for r in all_rows]),
                    columns=list(all_rows[0].keys()),
                )
                logger.info(f"Dataset saved to {npz_path}")

    # Compute statistics
    stats = {
        "total_configs": n_configs,
        "successful": n_success,
        "failed": n_failed,
        "success_rate": n_success / n_configs if n_configs > 0 else 0.0,
        "elapsed_seconds": elapsed,
        "configs_per_second": n_configs / elapsed if elapsed > 0 else 0.0,
    }

    # Feature statistics
    if all_rows:
        feature_names = FeatureVector.feature_names()
        for fname in feature_names:
            values = [r.get(fname, np.nan) for r in all_rows]
            values = [v for v in values if not np.isnan(v)]
            if values:
                stats[f"{fname}_min"] = float(np.min(values))
                stats[f"{fname}_max"] = float(np.max(values))
                stats[f"{fname}_mean"] = float(np.mean(values))

    logger.info(f"Dataset generation complete: {n_success}/{n_configs} successful")
    logger.info(f"Time elapsed: {elapsed:.1f}s ({stats['configs_per_second']:.2f} configs/s)")

    return stats


def main() -> None:
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate MyceliumFractalNet experimental dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with a preset (recommended)
  python -m experiments.generate_dataset --preset small
  python -m experiments.generate_dataset --preset medium

  # Legacy sweep mode
  python -m experiments.generate_dataset --sweep default --output data/my_dataset.parquet

Available presets: small, medium, large, benchmark
        """,
    )

    # Scenario-based arguments
    parser.add_argument(
        "--preset",
        "-p",
        type=str,
        choices=list_presets(),
        help="Scenario preset: small (quick test), medium (standard), large (production)",
    )

    # Legacy sweep arguments
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/mycelium_dataset.parquet"),
        help="Output path for dataset (legacy sweep mode)",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        choices=["minimal", "default", "extended"],
        help="Sweep configuration preset (legacy mode, use --preset instead)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets and exit",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle --list-presets
    if args.list_presets:
        print("Available presets:")
        for preset_name in list_presets():
            config = get_preset_config(preset_name)
            print(f"  {preset_name}: {config.description}")
        return

    # Scenario-based mode (preferred)
    if args.preset:
        config = get_preset_config(args.preset)
        config.base_seed = args.seed

        print(f"\n=== Running Scenario: {config.name} ===")
        print(f"Description: {config.description}")
        print(f"Grid: {config.grid_size}x{config.grid_size}, Steps: {config.steps}")
        print(f"Samples: {config.num_samples}")
        print()

        meta = run_scenario(config, data_root=Path("data"))

        print("\n=== Scenario Complete ===")
        print(f"Output: {meta.output_path}")
        print(f"Rows: {meta.num_rows}")
        print(f"Columns: {meta.num_columns}")
        print(f"Time: {meta.elapsed_seconds:.1f}s")
        return

    # Legacy sweep mode
    if args.sweep:
        if args.sweep == "minimal":
            sweep = _create_minimal_sweep()
        elif args.sweep == "extended":
            sweep = _create_extended_sweep()
        else:
            sweep = _create_default_sweep()

        sweep.base_seed = args.seed

        # Generate dataset
        stats = generate_dataset(sweep, args.output)

        # Print summary
        print("\n=== Dataset Generation Summary ===")
        print(f"Total configurations: {stats['total_configs']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Time: {stats['elapsed_seconds']:.1f}s")

        if stats["successful"] > 0:
            print("\n=== Feature Ranges ===")
            for key in ["D_box", "V_mean", "f_active"]:
                if f"{key}_mean" in stats:
                    print(
                        f"{key}: [{stats[f'{key}_min']:.3f}, {stats[f'{key}_max']:.3f}] "
                        f"(mean: {stats[f'{key}_mean']:.3f})"
                    )
        return

    # Default: show help
    parser.print_help()
    print("\nTip: Use --preset small for a quick test run.")


if __name__ == "__main__":
    main()
