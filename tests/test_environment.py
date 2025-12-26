"""
Environment validation tests.

These tests verify that critical dependencies are installed correctly.
They run fast and fail early if the environment is misconfigured.
"""


def test_must_have_pandas() -> None:
    """Verify pandas is installed for dataset generation."""
    try:
        import pandas  # noqa: F401
    except ModuleNotFoundError:
        raise RuntimeError(
            "pandas is required for dataset generation. "
            "Install with: pip install -r requirements.txt"
        )


def test_must_have_numpy() -> None:
    """Verify numpy is installed for numerical computations."""
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        raise RuntimeError(
            "numpy is required for numerical computations. "
            "Install with: pip install -r requirements.txt"
        )


def test_must_have_torch() -> None:
    """Verify torch is installed for neural network operations."""
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        raise RuntimeError(
            "torch is required for neural network operations. "
            "Install with: pip install -r requirements.txt"
        )


def test_pandas_version_compatibility() -> None:
    """Verify pandas version is within supported range."""
    import pandas as pd
    from packaging import version

    min_version = "1.5.3"
    max_version = "3.0.0"

    current = version.parse(pd.__version__)
    min_v = version.parse(min_version)
    max_v = version.parse(max_version)

    assert current >= min_v, f"pandas version {pd.__version__} is too old (minimum: {min_version})"
    assert current < max_v, f"pandas version {pd.__version__} is too new (maximum: <{max_version})"


def test_numpy_typing_available() -> None:
    """Verify numpy.typing module is available for type hints."""
    try:
        from numpy.typing import NDArray  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "numpy.typing is required. Ensure numpy >= 1.20 is installed. "
            "Install with: pip install -r requirements.txt"
        )
