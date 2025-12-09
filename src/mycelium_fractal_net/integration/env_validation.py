"""
Environment validation utilities for production safety.

Centralizes environment variable validation to prevent misconfigurations
and ensure production safety.

Usage:
    >>> from mycelium_fractal_net.integration.env_validation import validate_production_env
    >>> validate_production_env()  # Raises ConfigurationError if unsafe

Reference: Technical debt item P2-C2
"""

from __future__ import annotations

import os
import warnings


class ConfigurationError(Exception):
    """Raised when configuration is invalid or unsafe for production."""

    pass


def validate_production_env(raise_on_error: bool = True) -> list[str]:
    """
    Validate production environment configuration.

    Checks for common misconfigurations that could lead to security issues
    or operational problems in production.

    Args:
        raise_on_error: Whether to raise on critical errors (default: True).
                       If False, returns list of error messages.

    Returns:
        List of warning/error messages (empty if all validations pass).

    Raises:
        ConfigurationError: If critical misconfigurations detected and raise_on_error=True.
    """
    env = os.getenv("MFN_ENV", "prod").lower()
    issues: list[str] = []

    # Only validate if in production
    if env not in ("prod", "production"):
        return issues

    # Check API key configuration
    api_key = os.getenv("MFN_API_KEY", "")
    api_key_required = os.getenv("MFN_API_KEY_REQUIRED", "true").lower() in (
        "true",
        "1",
        "yes",
    )

    if api_key_required:
        if not api_key:
            issues.append(
                "CRITICAL: MFN_API_KEY_REQUIRED=true but MFN_API_KEY is not set. "
                "API will reject all requests."
            )
        elif len(api_key) < 16:
            issues.append(
                f"WARNING: MFN_API_KEY is short ({len(api_key)} chars). "
                "Use strong, randomly generated keys (min 32 chars recommended)."
            )
        elif api_key in (
            "dev-key-for-testing",
            "test",
            "placeholder",
            "changeme",
            "secret",
        ):
            issues.append(
                f"CRITICAL: MFN_API_KEY contains placeholder value '{api_key}'. "
                "This is a severe security risk in production."
            )

    # Check CORS configuration
    cors_origins = os.getenv("MFN_CORS_ORIGINS", "")
    if not cors_origins:
        issues.append(
            "WARNING: MFN_CORS_ORIGINS not set in production. "
            "CORS will be disabled. Set to allow specific origins if needed."
        )
    elif "*" in cors_origins:
        issues.append(
            "CRITICAL: MFN_CORS_ORIGINS contains wildcard '*'. "
            "This allows any origin and is a security risk in production."
        )

    # Check rate limiting
    rate_limit_enabled = os.getenv("MFN_RATE_LIMIT_ENABLED", "true").lower() in (
        "true",
        "1",
        "yes",
    )
    if not rate_limit_enabled:
        issues.append(
            "WARNING: Rate limiting is disabled (MFN_RATE_LIMIT_ENABLED=false). "
            "This may allow abuse. Enable rate limiting or use external protection."
        )

    # Check proxy header trust
    trust_proxy = os.getenv("MFN_TRUST_PROXY_HEADERS", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    if not trust_proxy:
        issues.append(
            "INFO: MFN_TRUST_PROXY_HEADERS=false (default). If behind a trusted reverse "
            "proxy, set to 'true' for accurate client IP detection in rate limiting."
        )

    # Check for critical issues
    critical_issues = [issue for issue in issues if issue.startswith("CRITICAL:")]

    if critical_issues and raise_on_error:
        raise ConfigurationError(
            "Production environment validation failed:\n"
            + "\n".join(critical_issues)
        )

    # Emit warnings for non-critical issues
    for issue in issues:
        if not issue.startswith("CRITICAL:"):
            warnings.warn(issue, UserWarning, stacklevel=2)

    return issues


__all__ = [
    "ConfigurationError",
    "validate_production_env",
]
