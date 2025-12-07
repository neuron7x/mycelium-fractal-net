#!/usr/bin/env python3
"""
Verification script to test that all mycelium_fractal_net subpackages are included
in the built wheel and can be imported successfully.
"""
import sys
import pkgutil
import importlib


def verify_package_completeness():
    """Verify that all nested packages under mycelium_fractal_net can be imported."""
    print("=" * 70)
    print("Package Import Verification")
    print("=" * 70)
    
    # Import the main package
    try:
        import mycelium_fractal_net
        print(f"✓ Main package imported: mycelium_fractal_net")
        print(f"  Package path: {mycelium_fractal_net.__file__}")
    except ImportError as e:
        print(f"✗ Failed to import mycelium_fractal_net: {e}")
        return False
    
    # Expected subpackages based on the source structure
    expected_subpackages = [
        "mycelium_fractal_net.analytics",
        "mycelium_fractal_net.core",
        "mycelium_fractal_net.crypto",
        "mycelium_fractal_net.experiments",
        "mycelium_fractal_net.integration",
        "mycelium_fractal_net.numerics",
        "mycelium_fractal_net.pipelines",
        "mycelium_fractal_net.security",
        "mycelium_fractal_net.types",
    ]
    
    print("\nVerifying expected subpackages:")
    all_passed = True
    for subpkg in expected_subpackages:
        try:
            importlib.import_module(subpkg)
            print(f"  ✓ {subpkg}")
        except ImportError as e:
            print(f"  ✗ {subpkg}: {e}")
            all_passed = False
    
    # Recursively discover and import all packages
    print("\nRecursively discovering all packages:")
    discovered_modules = []
    try:
        for importer, modname, ispkg in pkgutil.walk_packages(
            mycelium_fractal_net.__path__,
            mycelium_fractal_net.__name__ + "."
        ):
            discovered_modules.append((modname, ispkg))
            try:
                importlib.import_module(modname)
                pkg_type = "package" if ispkg else "module"
                print(f"  ✓ {modname} ({pkg_type})")
            except Exception as e:
                print(f"  ✗ {modname}: {e}")
                all_passed = False
    except Exception as e:
        print(f"✗ Error during package discovery: {e}")
        return False
    
    print(f"\nTotal discovered: {len(discovered_modules)} modules/packages")
    print("=" * 70)
    
    if all_passed:
        print("✓ All imports succeeded!")
        return True
    else:
        print("✗ Some imports failed!")
        return False


if __name__ == "__main__":
    success = verify_package_completeness()
    sys.exit(0 if success else 1)
