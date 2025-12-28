"""Shared runtime configuration assembly for API and CLI."""

from __future__ import annotations

import os
from typing import Any, Mapping

from mycelium_fractal_net.config_profiles import (
    ConfigProfile,
    ConfigValidationError,
    load_config_profile,
)
from mycelium_fractal_net.model import ValidationConfig
from mycelium_fractal_net.integration.schemas import ValidateRequest

DEFAULT_PROFILE_ENV = "MFN_CONFIG_PROFILE"
DEFAULT_PROFILE_NAME = "dev"


def _load_profile(profile_name: str | None) -> ConfigProfile | None:
    """Load a config profile, returning None when the profile is missing."""
    if not profile_name:
        return None
    try:
        return load_config_profile(profile_name)
    except (FileNotFoundError, ConfigValidationError):
        return None


def assemble_validation_config(
    request: ValidateRequest | Mapping[str, Any] | None = None,
    profile_name: str | None = None,
) -> ValidationConfig:
    """
    Build ValidationConfig using a deterministic precedence:
    defaults → profile → env overrides (applied via load_config_profile) → request overrides.
    """
    base_config = ValidationConfig()
    base = {key: value for key, value in vars(base_config).items()}
    profile = _load_profile(profile_name or os.getenv(DEFAULT_PROFILE_ENV, DEFAULT_PROFILE_NAME))
    if profile:
        base.update(profile.to_dict().get("validation", {}))

    if request:
        request_data = (
            request.model_dump()
            if callable(getattr(request, "model_dump", None))
            else dict(request)
        )
        base.update(request_data)

    return ValidationConfig(**base)


__all__ = ["assemble_validation_config"]
