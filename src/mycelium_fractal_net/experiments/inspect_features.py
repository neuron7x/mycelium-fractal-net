"""
Feature Inspection and Exploratory Analysis Script.

Loads a generated dataset and provides summary statistics, correlations,
and basic visualizations.

Usage:
    python -m mycelium_fractal_net.experiments.inspect_features --input data/mycelium_dataset.parquet

Features:
- Descriptive statistics for all features
- Correlation analysis
- Sanity checks for feature validity
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ..analytics import FeatureVector

logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> tuple[Any, list[str]]:
    """
    Load dataset from parquet or npz format.

    Returns
    -------
    tuple
        (data array/dataframe, column names)
    """
    if path.suffix == ".parquet":
        try:
            import pandas as pd

            df = pd.read_parquet(path)
            return df, list(df.columns)
        except ImportError:
            raise RuntimeError("pandas required for parquet files")
    elif path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return data["data"], list(data["columns"])
    else:
        raise ValueError(f"Unknown file format: {path.suffix}")


def compute_descriptive_stats(df: Any, feature_names: list[str]) -> dict[str, dict[str, float]]:
    """Compute descriptive statistics for all features."""
    stats: dict[str, dict[str, float]] = {}

    for fname in feature_names:
        if fname not in df.columns:
            logger.warning(f"Expected feature '{fname}' not found in dataset")
            continue

        values = df[fname].values
        values = values[~np.isnan(values)]

        if len(values) == 0:
            continue

        stats[fname] = {
            "count": len(values),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
        }

    return stats


def compute_correlations(df: Any, feature_names: list[str]) -> dict[tuple[str, str], float]:
    """Compute pairwise correlations between features."""
    correlations: dict[tuple[str, str], float] = {}

    valid_features = [f for f in feature_names if f in df.columns]

    for i, f1 in enumerate(valid_features):
        for f2 in valid_features[i + 1 :]:
            v1 = df[f1].values.astype(float)
            v2 = df[f2].values.astype(float)

            # Remove NaN pairs
            mask = ~(np.isnan(v1) | np.isnan(v2))
            v1, v2 = v1[mask], v2[mask]

            if len(v1) < 2:
                continue

            corr = np.corrcoef(v1, v2)[0, 1]
            correlations[(f1, f2)] = float(corr)

    return correlations


def run_sanity_checks(df: Any, feature_names: list[str]) -> list[str]:
    """Run sanity checks on feature values."""
    warnings: list[str] = []

    # Check fractal dimension range
    if "D_box" in df.columns:
        d_min, d_max = df["D_box"].min(), df["D_box"].max()
        if d_min < 0 or d_max > 2.5:
            warnings.append(f"D_box out of range: [{d_min:.3f}, {d_max:.3f}]")

    # Check voltage ranges
    if "V_min" in df.columns:
        v_min = df["V_min"].min()
        if v_min < -100:
            warnings.append(f"V_min too low: {v_min:.1f} mV")

    if "V_max" in df.columns:
        v_max = df["V_max"].max()
        if v_max > 50:
            warnings.append(f"V_max too high: {v_max:.1f} mV")

    # Check for NaN in features
    for fname in feature_names:
        if fname in df.columns:
            nan_count = df[fname].isna().sum()
            if nan_count > 0:
                warnings.append(f"{fname}: {nan_count} NaN values")

    # Check active fraction
    if "f_active" in df.columns:
        f_min, f_max = df["f_active"].min(), df["f_active"].max()
        if f_min < 0 or f_max > 1:
            warnings.append(f"f_active out of [0,1]: [{f_min:.3f}, {f_max:.3f}]")

    # Check for highly correlated features (redundancy)
    # This is informational, not an error

    return warnings


def print_summary(
    stats: dict[str, dict[str, float]],
    correlations: dict[tuple[str, str], float],
    warnings: list[str],
    n_samples: int,
) -> None:
    """Print formatted summary to console."""
    print("\n" + "=" * 60)
    print("MYCELIUM FRACTAL FEATURES - DATASET INSPECTION")
    print("=" * 60)

    print(f"\nTotal samples: {n_samples}")

    print("\n--- Descriptive Statistics ---")
    print(f"{'Feature':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("-" * 60)

    for fname, s in stats.items():
        print(
            f"{fname:<20} {s['min']:>10.3f} {s['max']:>10.3f} "
            f"{s['mean']:>10.3f} {s['std']:>10.3f}"
        )

    # Top correlations
    if correlations:
        print("\n--- Top Correlations (|r| > 0.5) ---")
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        for (f1, f2), r in sorted_corr[:10]:
            if abs(r) > 0.5:
                print(f"  {f1} <-> {f2}: {r:.3f}")

    # Warnings
    if warnings:
        print("\n--- Sanity Check Warnings ---")
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print("\n✓ All sanity checks passed")

    print("\n" + "=" * 60)


def main() -> None:
    """Main entry point for feature inspection."""
    parser = argparse.ArgumentParser(
        description="Inspect MyceliumFractalNet dataset features"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/mycelium_dataset.parquet"),
        help="Input dataset path",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all correlations (not just top)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Dataset not found at {args.input}")
        print("Run `python -m mycelium_fractal_net.experiments.generate_dataset` first")
        sys.exit(1)

    # Load dataset
    print(f"Loading dataset from {args.input}...")
    df, columns = load_dataset(args.input)

    # Get feature names
    feature_names = FeatureVector.feature_names()

    # Compute statistics
    stats = compute_descriptive_stats(df, feature_names)
    correlations = compute_correlations(df, feature_names)
    warnings = run_sanity_checks(df, feature_names)

    # Print summary
    print_summary(stats, correlations, warnings, len(df))


if __name__ == "__main__":
    main()
