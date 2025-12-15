"""Tests for API configuration alignment across components."""

from __future__ import annotations

import os

from mycelium_fractal_net.integration.api_config import APIConfig, reset_config


def _clear_env_vars() -> None:
    for var in ["MFN_ENV", "MFN_METRICS_ENABLED", "MFN_METRICS_INCLUDE_IN_AUTH"]:
        os.environ.pop(var, None)


def test_metrics_public_by_default(monkeypatch) -> None:
    """Metrics endpoint should remain public when auth is not requested."""

    _clear_env_vars()
    reset_config()

    config = APIConfig.from_env()

    assert config.metrics.include_in_auth is False
    assert "/metrics" in config.auth.public_endpoints


def test_metrics_can_be_protected(monkeypatch) -> None:
    """Enabling metrics auth should remove the endpoint from public list."""

    _clear_env_vars()
    monkeypatch.setenv("MFN_METRICS_INCLUDE_IN_AUTH", "true")
    reset_config()

    config = APIConfig.from_env()

    assert config.metrics.include_in_auth is True
    assert "/metrics" not in config.auth.public_endpoints
