"""
Cryptographic configuration for MyceliumFractalNet API.

Provides configuration management for cryptographic operations including
algorithm selection, key management, and audit logging settings.

Environment Variables:
    MFN_CRYPTO_ENABLED           - Enable/disable crypto module (default: true)
    MFN_CRYPTO_SYMMETRIC_ALGO    - Symmetric algorithm (default: aes-256-gcm)
    MFN_CRYPTO_SYMMETRIC_KEY_BITS - Key length in bits (default: 256)
    MFN_CRYPTO_KEY_EXCHANGE      - Key exchange algorithm (default: x25519)
    MFN_CRYPTO_SIGNATURE         - Signature algorithm (default: ed25519)
    MFN_CRYPTO_MAX_PAYLOAD       - Max payload size in bytes (default: 10MB)
    MFN_CRYPTO_AUDIT_ENABLED     - Enable audit logging (default: true)

Reference: docs/MFN_CRYPTOGRAPHY.md
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Check for yaml availability (optional dependency)
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class SymmetricAlgorithm(str, Enum):
    """Supported symmetric encryption algorithms."""

    AES_256_GCM = "aes-256-gcm"


class KeyExchangeAlgorithm(str, Enum):
    """Supported key exchange algorithms."""

    X25519 = "x25519"


class SignatureAlgorithm(str, Enum):
    """Supported digital signature algorithms."""

    ED25519 = "ed25519"


@dataclass
class AlgorithmConfig:
    """
    Algorithm configuration for cryptographic operations.

    Attributes:
        symmetric: Symmetric encryption algorithm.
        symmetric_key_bits: Key length for symmetric encryption.
        key_exchange: Key exchange algorithm.
        signature: Digital signature algorithm.
    """

    symmetric: SymmetricAlgorithm = SymmetricAlgorithm.AES_256_GCM
    symmetric_key_bits: int = 256
    key_exchange: KeyExchangeAlgorithm = KeyExchangeAlgorithm.X25519
    signature: SignatureAlgorithm = SignatureAlgorithm.ED25519


@dataclass
class KeyManagementConfig:
    """
    Key management configuration.

    Attributes:
        max_stored_keys: Maximum number of keys stored per type.
        key_expiration_seconds: Key expiration time (0 = never).
        auto_rotate: Enable automatic key rotation.
        rotation_interval_seconds: Key rotation interval.
    """

    max_stored_keys: int = 100
    key_expiration_seconds: int = 86400  # 24 hours
    auto_rotate: bool = False
    rotation_interval_seconds: int = 604800  # 7 days


@dataclass
class CryptoAPIConfig:
    """
    API-specific cryptographic configuration.

    Attributes:
        max_payload_bytes: Maximum payload size for encryption/decryption.
        require_base64_input: Require base64 encoded input.
        base64_output: Return base64 encoded output.
        rate_limit_per_minute: Rate limit for crypto endpoints.
    """

    max_payload_bytes: int = 10_485_760  # 10 MB
    require_base64_input: bool = True
    base64_output: bool = True
    rate_limit_per_minute: int = 100


@dataclass
class CryptoAuditConfig:
    """
    Audit logging configuration for cryptographic operations.

    Attributes:
        enabled: Enable audit logging.
        log_level: Log level for crypto operations.
        include_fields: Fields to include in audit logs.
    """

    enabled: bool = True
    log_level: str = "INFO"
    include_fields: List[str] = field(
        default_factory=lambda: [
            "operation",
            "key_id",
            "algorithm",
            "timestamp",
            "request_id",
            "success",
            "error_code",
        ]
    )


@dataclass
class CryptoConfig:
    """
    Complete cryptographic configuration.

    Aggregates all crypto-related configuration including algorithm selection,
    key management, API settings, and audit logging.

    Attributes:
        enabled: Master switch for crypto module.
        algorithms: Algorithm configuration.
        key_management: Key management configuration.
        api: API-specific configuration.
        audit: Audit logging configuration.
    """

    enabled: bool = True
    algorithms: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    key_management: KeyManagementConfig = field(default_factory=KeyManagementConfig)
    api: CryptoAPIConfig = field(default_factory=CryptoAPIConfig)
    audit: CryptoAuditConfig = field(default_factory=CryptoAuditConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CryptoConfig":
        """
        Create CryptoConfig from dictionary (e.g., parsed YAML).

        Args:
            data: Configuration dictionary.

        Returns:
            CryptoConfig: Configured crypto settings.
        """
        enabled = data.get("enabled", True)

        # Parse algorithms
        algo_data = data.get("algorithms", {})
        algorithms = AlgorithmConfig(
            symmetric=SymmetricAlgorithm(algo_data.get("symmetric", "aes-256-gcm")),
            symmetric_key_bits=algo_data.get("symmetric_key_bits", 256),
            key_exchange=KeyExchangeAlgorithm(algo_data.get("key_exchange", "x25519")),
            signature=SignatureAlgorithm(algo_data.get("signature", "ed25519")),
        )

        # Parse key management
        km_data = data.get("key_management", {})
        key_management = KeyManagementConfig(
            max_stored_keys=km_data.get("max_stored_keys", 100),
            key_expiration_seconds=km_data.get("key_expiration_seconds", 86400),
            auto_rotate=km_data.get("auto_rotate", False),
            rotation_interval_seconds=km_data.get("rotation_interval_seconds", 604800),
        )

        # Parse API config
        api_data = data.get("api", {})
        api = CryptoAPIConfig(
            max_payload_bytes=api_data.get("max_payload_bytes", 10_485_760),
            require_base64_input=api_data.get("require_base64_input", True),
            base64_output=api_data.get("base64_output", True),
            rate_limit_per_minute=api_data.get("rate_limit_per_minute", 100),
        )

        # Parse audit config
        audit_data = data.get("audit", {})
        audit = CryptoAuditConfig(
            enabled=audit_data.get("enabled", True),
            log_level=audit_data.get("log_level", "INFO"),
            include_fields=audit_data.get(
                "include_fields",
                [
                    "operation",
                    "key_id",
                    "algorithm",
                    "timestamp",
                    "request_id",
                    "success",
                    "error_code",
                ],
            ),
        )

        return cls(
            enabled=enabled,
            algorithms=algorithms,
            key_management=key_management,
            api=api,
            audit=audit,
        )

    @classmethod
    def from_env(cls) -> "CryptoConfig":
        """
        Create CryptoConfig from environment variables.

        Returns:
            CryptoConfig: Configured crypto settings from environment.
        """
        # Parse enabled flag
        enabled_str = os.getenv("MFN_CRYPTO_ENABLED", "true").lower()
        enabled = enabled_str in ("true", "1", "yes")

        # Parse algorithm settings
        symmetric_algo = os.getenv("MFN_CRYPTO_SYMMETRIC_ALGO", "aes-256-gcm")
        symmetric_key_bits = int(os.getenv("MFN_CRYPTO_SYMMETRIC_KEY_BITS", "256"))
        key_exchange = os.getenv("MFN_CRYPTO_KEY_EXCHANGE", "x25519")
        signature = os.getenv("MFN_CRYPTO_SIGNATURE", "ed25519")

        algorithms = AlgorithmConfig(
            symmetric=SymmetricAlgorithm(symmetric_algo),
            symmetric_key_bits=symmetric_key_bits,
            key_exchange=KeyExchangeAlgorithm(key_exchange),
            signature=SignatureAlgorithm(signature),
        )

        # Parse key management settings
        max_stored_keys = int(os.getenv("MFN_CRYPTO_MAX_STORED_KEYS", "100"))
        key_expiration = int(os.getenv("MFN_CRYPTO_KEY_EXPIRATION", "86400"))
        auto_rotate_str = os.getenv("MFN_CRYPTO_AUTO_ROTATE", "false").lower()
        auto_rotate = auto_rotate_str in ("true", "1", "yes")
        rotation_interval = int(os.getenv("MFN_CRYPTO_ROTATION_INTERVAL", "604800"))

        key_management = KeyManagementConfig(
            max_stored_keys=max_stored_keys,
            key_expiration_seconds=key_expiration,
            auto_rotate=auto_rotate,
            rotation_interval_seconds=rotation_interval,
        )

        # Parse API settings
        max_payload = int(os.getenv("MFN_CRYPTO_MAX_PAYLOAD", "10485760"))
        require_base64_str = os.getenv("MFN_CRYPTO_REQUIRE_BASE64", "true").lower()
        require_base64 = require_base64_str in ("true", "1", "yes")
        rate_limit = int(os.getenv("MFN_CRYPTO_RATE_LIMIT", "100"))

        api = CryptoAPIConfig(
            max_payload_bytes=max_payload,
            require_base64_input=require_base64,
            base64_output=True,  # Always true for binary data
            rate_limit_per_minute=rate_limit,
        )

        # Parse audit settings
        audit_enabled_str = os.getenv("MFN_CRYPTO_AUDIT_ENABLED", "true").lower()
        audit_enabled = audit_enabled_str in ("true", "1", "yes")
        audit_level = os.getenv("MFN_CRYPTO_AUDIT_LEVEL", "INFO")

        audit = CryptoAuditConfig(
            enabled=audit_enabled,
            log_level=audit_level,
        )

        return cls(
            enabled=enabled,
            algorithms=algorithms,
            key_management=key_management,
            api=api,
            audit=audit,
        )

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "CryptoConfig":
        """
        Load configuration from file and environment.

        Environment variables override file settings.

        Args:
            config_path: Path to crypto.yaml file. If None, looks in standard locations.

        Returns:
            CryptoConfig: Merged configuration.
        """
        config_data: Dict[str, Any] = {}

        # Try to load from file
        if config_path is None:
            # Check standard locations
            possible_paths = [
                Path("config/crypto.yaml"),
                Path("configs/crypto.yaml"),
                Path("/etc/mfn/crypto.yaml"),
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break

        if config_path is not None and config_path.exists() and YAML_AVAILABLE:
            with open(config_path) as f:
                config_data = yaml.safe_load(f) or {}

        # Start with file config or defaults
        if config_data:
            config = cls.from_dict(config_data)
        else:
            config = cls()

        # Override with environment variables
        env_config = cls.from_env()

        # Only override if explicitly set in environment
        if os.getenv("MFN_CRYPTO_ENABLED"):
            config.enabled = env_config.enabled
        if os.getenv("MFN_CRYPTO_SYMMETRIC_ALGO"):
            config.algorithms.symmetric = env_config.algorithms.symmetric
        if os.getenv("MFN_CRYPTO_SYMMETRIC_KEY_BITS"):
            config.algorithms.symmetric_key_bits = env_config.algorithms.symmetric_key_bits
        if os.getenv("MFN_CRYPTO_KEY_EXCHANGE"):
            config.algorithms.key_exchange = env_config.algorithms.key_exchange
        if os.getenv("MFN_CRYPTO_SIGNATURE"):
            config.algorithms.signature = env_config.algorithms.signature
        if os.getenv("MFN_CRYPTO_MAX_PAYLOAD"):
            config.api.max_payload_bytes = env_config.api.max_payload_bytes
        if os.getenv("MFN_CRYPTO_AUDIT_ENABLED"):
            config.audit.enabled = env_config.audit.enabled

        return config


# Singleton instance
_crypto_config: Optional[CryptoConfig] = None


def get_crypto_config() -> CryptoConfig:
    """
    Get the crypto configuration singleton.

    Returns:
        CryptoConfig: Current crypto configuration.
    """
    global _crypto_config
    if _crypto_config is None:
        _crypto_config = CryptoConfig.load()
    return _crypto_config


def reset_crypto_config() -> None:
    """Reset the crypto configuration singleton (useful for testing)."""
    global _crypto_config
    _crypto_config = None


def is_crypto_enabled() -> bool:
    """
    Check if cryptographic operations are enabled.

    Returns:
        bool: True if crypto module is enabled.
    """
    return get_crypto_config().enabled


__all__ = [
    "SymmetricAlgorithm",
    "KeyExchangeAlgorithm",
    "SignatureAlgorithm",
    "AlgorithmConfig",
    "KeyManagementConfig",
    "CryptoAPIConfig",
    "CryptoAuditConfig",
    "CryptoConfig",
    "get_crypto_config",
    "reset_crypto_config",
    "is_crypto_enabled",
]
