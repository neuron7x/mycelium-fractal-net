"""
Global pytest configuration and fixtures for MFN tests.

Sets up the test environment to ensure consistent behavior across all tests.
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """
    Configure pytest environment before tests run.

    Sets MFN_ENV=dev to ensure authentication and rate limiting are disabled
    by default during tests, unless explicitly overridden.
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


@pytest.fixture(autouse=True, scope="session")
def setup_test_environment():
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
