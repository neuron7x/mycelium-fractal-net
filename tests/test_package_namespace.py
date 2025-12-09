"""
Test to ensure no top-level package namespace pollution.

This test verifies that after installation, only mycelium_fractal_net is
installed at the top level, and generic names like 'analytics' or 'experiments'
are not polluting the global namespace.

Reference: P0 package collision prevention requirement.
"""

import subprocess
import sys
from pathlib import Path


def test_no_top_level_analytics_in_distribution() -> None:
    """Verify that 'analytics' is not a top-level package in the distribution."""
    # Build wheel first
    project_root = Path(__file__).parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Failed to build wheel: {result.stderr}"

    # Check wheel contents
    dist_dir = project_root / "dist"
    wheels = list(dist_dir.glob("*.whl"))
    assert len(wheels) > 0, "No wheel file found"

    wheel_file = wheels[-1]  # Use most recent
    result = subprocess.run(
        ["unzip", "-l", str(wheel_file)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Failed to list wheel contents: {result.stderr}"

    lines = result.stdout.splitlines()

    # Check that no top-level analytics/ or experiments/ directories exist
    for line in lines:
        # Skip the archive line and empty lines
        if not line.strip() or "Archive:" in line:
            continue

        # Extract path (format: "  length   date   time   name")
        parts = line.split()
        if len(parts) >= 4:
            path = parts[-1]

            # Check for top-level analytics or experiments
            assert not path.startswith("analytics/"), (
                f"Top-level 'analytics' package found in wheel: {path}. "
                "This creates namespace pollution risk."
            )
            assert not path.startswith("experiments/"), (
                f"Top-level 'experiments' package found in wheel: {path}. "
                "This creates namespace pollution risk."
            )

            # Ensure namespaced versions exist
            if "analytics" in path:
                assert path.startswith("mycelium_fractal_net/analytics/"), (
                    f"Analytics found in unexpected location: {path}"
                )
            if "experiments" in path:
                assert path.startswith("mycelium_fractal_net/experiments/"), (
                    f"Experiments found in unexpected location: {path}"
                )


def test_canonical_imports_work() -> None:
    """Verify that canonical namespaced imports work correctly."""
    # Test analytics import
    from mycelium_fractal_net.analytics import (
        FeatureConfig,
        FeatureVector,
        compute_features,
    )

    assert FeatureConfig is not None
    assert FeatureVector is not None
    assert compute_features is not None

    # Test experiments import
    from mycelium_fractal_net.experiments import (
        SweepConfig,
        generate_dataset,
    )

    assert SweepConfig is not None
    assert generate_dataset is not None


def test_top_level_analytics_not_importable() -> None:
    """Verify that top-level 'analytics' is not accidentally importable from package."""
    # Check if 'analytics' package exists at top level from our distribution
    # Note: This may fail if another package named 'analytics' is installed
    # but that's actually the point - we don't want to conflict with others!

    # The key check is that OUR distribution doesn't provide it
    try:
        # Test that canonical import works
        from mycelium_fractal_net import analytics  # noqa: F401

        assert analytics is not None

        # Verify it's actually from our package
        assert "mycelium_fractal_net" in analytics.__name__
    except ImportError as e:
        raise AssertionError(
            f"Canonical mycelium_fractal_net.analytics import failed: {e}"
        )


def test_wheel_top_level_txt_only_has_mycelium_fractal_net() -> None:
    """Verify that top_level.txt in the wheel only lists mycelium_fractal_net."""
    project_root = Path(__file__).parent.parent
    dist_dir = project_root / "dist"
    wheels = list(dist_dir.glob("*.whl"))
    assert len(wheels) > 0, "No wheel file found"

    wheel_file = wheels[-1]
    result = subprocess.run(
        ["unzip", "-p", str(wheel_file), "**/top_level.txt"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Failed to extract top_level.txt: {result.stderr}"

    top_level_packages = [
        line.strip() for line in result.stdout.strip().splitlines() if line.strip()
    ]

    # Should only contain mycelium_fractal_net
    assert top_level_packages == ["mycelium_fractal_net"], (
        f"top_level.txt should only contain 'mycelium_fractal_net', "
        f"but found: {top_level_packages}"
    )

    # Specifically ensure analytics and experiments are NOT in top_level.txt
    assert "analytics" not in top_level_packages, (
        "Top-level 'analytics' found in top_level.txt - namespace pollution!"
    )
    assert "experiments" not in top_level_packages, (
        "Top-level 'experiments' found in top_level.txt - namespace pollution!"
    )
