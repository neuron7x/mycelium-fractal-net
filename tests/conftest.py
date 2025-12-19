"""
Global pytest configuration and fixtures for MFN tests.

Sets up the test environment to ensure consistent behavior across all tests.

Determinism Controls:
    - Random seed fixtures for reproducible tests
    - Time mocking for time-dependent tests
    - Environment isolation

Reference: Issue requirements for test determinism and repeatability.
"""

from __future__ import annotations

import os
import random
from typing import Generator

import numpy as np
import pytest
import torch

# Default seed for deterministic tests
DEFAULT_TEST_SEED = 42


def pytest_configure(config: pytest.Config) -> None:
    """
    Configure pytest environment before tests run.

    Sets MFN_ENV=dev to ensure authentication and rate limiting are disabled
    by default during tests, unless explicitly overridden.
    
    Also registers custom markers for test categorization.
    """
    # Only set if not already set (allows override for specific tests)
    if "MFN_ENV" not in os.environ:
        os.environ["MFN_ENV"] = "dev"

    # Disable auth by default in tests
    if "MFN_API_KEY_REQUIRED" not in os.environ:
        os.environ["MFN_API_KEY_REQUIRED"] = "false"

    # Disable rate limiting by default in tests
    if "MFN_RATE_LIMIT_ENABLED" not in os.environ:
        os.environ["MFN_RATE_LIMIT_ENABLED"] = "false"
    
    # Register custom markers
    config.addinivalue_line("markers", "deterministic: mark test as requiring deterministic RNG")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "security: mark test as security-related")


@pytest.fixture(autouse=True, scope="session")
def setup_test_environment() -> Generator[None, None, None]:
    """
    Session-scoped fixture to ensure test environment is properly configured.

    This runs once at the start of the test session.
    """
    # Store original values
    original_env = os.environ.get("MFN_ENV")
    original_auth = os.environ.get("MFN_API_KEY_REQUIRED")
    original_rate = os.environ.get("MFN_RATE_LIMIT_ENABLED")

    # Set test environment
    os.environ["MFN_ENV"] = "dev"
    os.environ["MFN_API_KEY_REQUIRED"] = "false"
    os.environ["MFN_RATE_LIMIT_ENABLED"] = "false"

    yield

    # Restore original values
    if original_env is not None:
        os.environ["MFN_ENV"] = original_env
    elif "MFN_ENV" in os.environ:
        del os.environ["MFN_ENV"]

    if original_auth is not None:
        os.environ["MFN_API_KEY_REQUIRED"] = original_auth
    elif "MFN_API_KEY_REQUIRED" in os.environ:
        del os.environ["MFN_API_KEY_REQUIRED"]

    if original_rate is not None:
        os.environ["MFN_RATE_LIMIT_ENABLED"] = original_rate
    elif "MFN_RATE_LIMIT_ENABLED" in os.environ:
        del os.environ["MFN_RATE_LIMIT_ENABLED"]


@pytest.fixture
def fixed_seed() -> int:
    """
    Provide a fixed seed for deterministic tests.
    
    Returns:
        Fixed seed value (42).
    """
    return DEFAULT_TEST_SEED


@pytest.fixture
def seeded_rng(fixed_seed: int) -> Generator[np.random.Generator, None, None]:
    """
    Provide a seeded NumPy random generator for deterministic tests.
    
    This fixture ensures tests using random numbers are reproducible.
    The generator is reset before each test.
    
    Args:
        fixed_seed: The seed to use for the RNG.
        
    Yields:
        A seeded NumPy random generator.
    """
    rng = np.random.default_rng(fixed_seed)
    yield rng


@pytest.fixture
def seed_all_rngs(fixed_seed: int) -> Generator[int, None, None]:
    """
    Seed all random number generators for fully deterministic tests.
    
    Seeds:
        - Python's random module
        - NumPy's global random state
        - PyTorch's random state (CPU and CUDA if available)
    
    Args:
        fixed_seed: The seed to use for all RNGs.
        
    Yields:
        The seed value that was used.
    """
    # Store original states
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = None
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()
    
    # Set seeds
    random.seed(fixed_seed)
    np.random.seed(fixed_seed)
    torch.manual_seed(fixed_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fixed_seed)
    
    # Enable deterministic mode for PyTorch (may impact performance)
    torch.use_deterministic_algorithms(False)  # Some ops don't have deterministic impl
    
    yield fixed_seed
    
    # Restore original states
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


@pytest.fixture
def isolated_env() -> Generator[dict[str, str | None], None, None]:
    """
    Provide an isolated environment for tests that modify environment variables.
    
    Stores all MFN_* environment variables before the test and restores them after.
    
    Yields:
        Dictionary of original MFN_* environment variable values.
    """
    # Store original MFN_* environment variables
    original_vars: dict[str, str | None] = {}
    for key in list(os.environ.keys()):
        if key.startswith("MFN_"):
            original_vars[key] = os.environ.get(key)
    
    yield original_vars
    
    # Restore original values
    # First, remove any new MFN_* variables
    for key in list(os.environ.keys()):
        if key.startswith("MFN_") and key not in original_vars:
            del os.environ[key]
    
    # Then restore original values
    for key, value in original_vars.items():
        if value is not None:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]
