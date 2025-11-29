"""
Smoke tests for mycelium_fractal_net package imports.

Verifies that the package can be imported and all public API symbols are accessible.
"""


def test_package_import() -> None:
    """Test that mycelium_fractal_net package imports successfully."""
    import mycelium_fractal_net as mfn

    assert mfn is not None


def test_package_version() -> None:
    """Test that package version is accessible."""
    import mycelium_fractal_net as mfn

    assert hasattr(mfn, "__version__")
    assert isinstance(mfn.__version__, str)
    assert mfn.__version__ == "4.1.0"


def test_simulation_types_import() -> None:
    """Test that simulation types are importable."""
    from mycelium_fractal_net import (
        FeatureVector,
        MyceliumField,
        SimulationConfig,
        SimulationResult,
    )

    assert SimulationConfig is not None
    assert SimulationResult is not None
    assert MyceliumField is not None
    assert FeatureVector is not None


def test_all_exports() -> None:
    """Test that all __all__ exports are importable."""
    import mycelium_fractal_net as mfn

    assert hasattr(mfn, "__all__")
    for name in mfn.__all__:
        assert hasattr(mfn, name), f"Missing export: {name}"


def test_submodules_exist() -> None:
    """Test that expected submodules exist."""
    from mycelium_fractal_net import analytics, core, experiments, numerics

    assert core is not None
    assert analytics is not None
    assert numerics is not None
    assert experiments is not None
