#!/usr/bin/env python3
"""
Validation script for P0 package namespace collision fix.

This script validates that:
1. The wheel only contains mycelium_fractal_net at top-level
2. Canonical imports work correctly
3. No top-level analytics or experiments packages are installed

Run this after building the distribution to verify the fix.
"""

import subprocess
import sys
import tempfile
from pathlib import Path


def validate_wheel_contents():
    """Validate that wheel only contains namespaced packages."""
    print("=" * 70)
    print("VALIDATION 1: Checking wheel contents")
    print("=" * 70)
    
    dist_dir = Path(__file__).parent.parent / "dist"
    wheels = list(dist_dir.glob("*.whl"))
    
    if not wheels:
        print("❌ FAILED: No wheel file found in dist/")
        return False
    
    wheel_file = wheels[-1]
    print(f"✓ Found wheel: {wheel_file.name}")
    
    result = subprocess.run(
        ["unzip", "-l", str(wheel_file)],
        capture_output=True,
        text=True,
    )
    
    has_top_level_analytics = False
    has_top_level_experiments = False
    has_namespaced_analytics = False
    has_namespaced_experiments = False
    
    for line in result.stdout.splitlines():
        if not line.strip() or "Archive:" in line:
            continue
        
        parts = line.split()
        if len(parts) >= 4:
            path = parts[-1]
            
            if path.startswith("analytics/") and not path.startswith("mycelium_fractal_net/"):
                has_top_level_analytics = True
                print(f"❌ Found top-level analytics: {path}")
            
            if path.startswith("experiments/") and not path.startswith("mycelium_fractal_net/"):
                has_top_level_experiments = True
                print(f"❌ Found top-level experiments: {path}")
            
            if "mycelium_fractal_net/analytics/" in path:
                has_namespaced_analytics = True
            
            if "mycelium_fractal_net/experiments/" in path:
                has_namespaced_experiments = True
    
    success = True
    
    if has_top_level_analytics:
        print("❌ FAILED: Top-level 'analytics' package found in wheel")
        success = False
    else:
        print("✓ No top-level 'analytics' package (good)")
    
    if has_top_level_experiments:
        print("❌ FAILED: Top-level 'experiments' package found in wheel")
        success = False
    else:
        print("✓ No top-level 'experiments' package (good)")
    
    if has_namespaced_analytics:
        print("✓ Namespaced 'mycelium_fractal_net.analytics' found (good)")
    else:
        print("❌ FAILED: Namespaced analytics not found")
        success = False
    
    if has_namespaced_experiments:
        print("✓ Namespaced 'mycelium_fractal_net.experiments' found (good)")
    else:
        print("❌ FAILED: Namespaced experiments not found")
        success = False
    
    return success


def validate_top_level_txt():
    """Validate top_level.txt only lists mycelium_fractal_net."""
    print("\n" + "=" * 70)
    print("VALIDATION 2: Checking top_level.txt")
    print("=" * 70)
    
    dist_dir = Path(__file__).parent.parent / "dist"
    wheels = list(dist_dir.glob("*.whl"))
    wheel_file = wheels[-1]
    
    result = subprocess.run(
        ["unzip", "-p", str(wheel_file), "**/top_level.txt"],
        capture_output=True,
        text=True,
    )
    
    top_level_packages = [
        line.strip() for line in result.stdout.strip().splitlines() if line.strip()
    ]
    
    print(f"Top-level packages: {top_level_packages}")
    
    if top_level_packages == ["mycelium_fractal_net"]:
        print("✓ Only 'mycelium_fractal_net' in top_level.txt (good)")
        return True
    else:
        print(f"❌ FAILED: Expected ['mycelium_fractal_net'], got {top_level_packages}")
        return False


def validate_imports():
    """Validate that canonical imports work."""
    print("\n" + "=" * 70)
    print("VALIDATION 3: Testing canonical imports")
    print("=" * 70)
    
    try:
        # Test analytics imports
        from mycelium_fractal_net.analytics import (  # noqa: F401
            FeatureConfig,  # noqa: F401
            FeatureVector,  # noqa: F401
            compute_features,  # noqa: F401
            compute_fractal_features,  # noqa: F401
        )
        print("✓ mycelium_fractal_net.analytics imports work")
        
        # Test experiments imports
        from mycelium_fractal_net.experiments import (  # noqa: F401
            SweepConfig,  # noqa: F401
            generate_dataset,  # noqa: F401
        )
        print("✓ mycelium_fractal_net.experiments imports work")
        
        # Test submodule access via main package
        from mycelium_fractal_net import analytics, experiments  # noqa: F401
        print("✓ Submodules accessible from main package")
        
        return True
    except ImportError as e:
        print(f"❌ FAILED: Import error: {e}")
        return False


def main():
    """Run all validations."""
    print("\n" + "=" * 70)
    print("MyceliumFractalNet Package Namespace Validation")
    print("P0 Fix: Remove top-level analytics/experiments packages")
    print("=" * 70 + "\n")
    
    results = []
    
    # Run validations
    results.append(("Wheel Contents", validate_wheel_contents()))
    results.append(("Top-Level Packages", validate_top_level_txt()))
    results.append(("Canonical Imports", validate_imports()))
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("\nThe package namespace collision has been successfully fixed:")
        print("- No top-level 'analytics' or 'experiments' packages")
        print("- Only 'mycelium_fractal_net' at top level")
        print("- Canonical imports work correctly")
        print("\nSafe to release! ✨")
        return 0
    else:
        print("\n⚠️  SOME VALIDATIONS FAILED")
        print("Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
