"""
Cryptographic configuration for MyceliumFractalNet.

Provides configuration management for cryptographic operations including:
- Enable/disable crypto layer via MFN_CRYPTO_ENABLED
- Cipher suite selection (AES-256-GCM)
- Key exchange algorithm (X25519)
- Signature algorithm (Ed25519)
- Key derivation settings
- Audit logging configuration

Configuration is sourced from environment variables and optional config files.

Environment Variables:
    MFN_CRYPTO_ENABLED         - Enable crypto layer (default: true)
    MFN_CRYPTO_CIPHER_SUITE    - Cipher suite (default: AES-256-GCM)
    MFN_CRYPTO_KEY_BITS        - Key size in bits (default: 256)
    MFN_CRYPTO_AUDIT_ENABLED   - Enable audit logging (default: true)
    MFN_CRYPTO_RATE_LIMIT      - Rate limit per minute (default: 100)
    MFN_CRYPTO_MAX_PAYLOAD_SIZE - Max payload size in bytes (default: 1MB)

Reference: docs/MFN_CRYPTOGRAPHY.md
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml


class CipherSuite(str, Enum):
    """Supported cipher suites for symmetric encryption."""

    AES_256_GCM = "AES-256-GCM"
    CHACHA20_POLY1305 = "ChaCha20-Poly1305"


class KeyExchangeAlgorithm(str, Enum):
    """Supported key exchange algorithms."""

    X25519 = "X25519"


class SignatureAlgorithm(str, Enum):
    """Supported signature algorithms."""

    ED25519 = "Ed25519"


class KDFAlgorithm(str, Enum):
    """Supported key derivation algorithms."""

    PBKDF2_SHA256 = "PBKDF2-SHA256"
    SCRYPT = "scrypt"


@dataclass
class EncryptionConfig:
    """
    Encryption configuration.

    Attributes:
        cipher_suite: Cipher suite for symmetric encryption.
        key_bits: Key size in bits (128, 192, 256).
    """

    cipher_suite: CipherSuite = CipherSuite.AES_256_GCM
    key_bits: int = 256

    @classmethod
    def from_dict(cls, data: dict) -> "EncryptionConfig":
        """Create config from dictionary."""
        cipher_str = data.get("cipher_suite", "AES-256-GCM")
        try:
            cipher_suite = CipherSuite(cipher_str)
        except ValueError:
            cipher_suite = CipherSuite.AES_256_GCM

        return cls(
            cipher_suite=cipher_suite,
            key_bits=data.get("key_bits", 256),
        )


@dataclass
class KeyExchangeConfig:
    """Key exchange configuration."""

    algorithm: KeyExchangeAlgorithm = KeyExchangeAlgorithm.X25519

    @classmethod
    def from_dict(cls, data: dict) -> "KeyExchangeConfig":
        """Create config from dictionary."""
        alg_str = data.get("algorithm", "X25519")
        try:
            algorithm = KeyExchangeAlgorithm(alg_str)
        except ValueError:
            algorithm = KeyExchangeAlgorithm.X25519

        return cls(algorithm=algorithm)


@dataclass
class SignatureConfig:
    """Digital signature configuration."""

    algorithm: SignatureAlgorithm = SignatureAlgorithm.ED25519

    @classmethod
    def from_dict(cls, data: dict) -> "SignatureConfig":
        """Create config from dictionary."""
        alg_str = data.get("algorithm", "Ed25519")
        try:
            algorithm = SignatureAlgorithm(alg_str)
        except ValueError:
            algorithm = SignatureAlgorithm.ED25519

        return cls(algorithm=algorithm)


@dataclass
class KDFConfig:
    """Key derivation function configuration."""

    algorithm: KDFAlgorithm = KDFAlgorithm.SCRYPT
    pbkdf2_iterations: int = 100_000
    scrypt_n: int = 16384
    scrypt_r: int = 8
    scrypt_p: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "KDFConfig":
        """Create config from dictionary."""
        alg_str = data.get("algorithm", "scrypt")
        try:
            algorithm = KDFAlgorithm(alg_str)
        except ValueError:
            algorithm = KDFAlgorithm.SCRYPT

        return cls(
            algorithm=algorithm,
            pbkdf2_iterations=data.get("pbkdf2_iterations", 100_000),
            scrypt_n=data.get("scrypt_n", 16384),
            scrypt_r=data.get("scrypt_r", 8),
            scrypt_p=data.get("scrypt_p", 1),
        )


@dataclass
class CryptoAPIConfig:
    """Crypto API configuration."""

    rate_limit: int = 100  # requests per minute
    max_payload_size: int = 1_048_576  # 1 MB

    @classmethod
    def from_dict(cls, data: dict) -> "CryptoAPIConfig":
        """Create config from dictionary."""
        return cls(
            rate_limit=data.get("rate_limit", 100),
            max_payload_size=data.get("max_payload_size", 1_048_576),
        )


@dataclass
class AuditConfig:
    """Audit logging configuration for crypto operations."""

    enabled: bool = True
    level: str = "INFO"
    include_key_id: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "AuditConfig":
        """Create config from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            level=data.get("level", "INFO"),
            include_key_id=data.get("include_key_id", True),
        )


@dataclass
class TLSConfig:
    """TLS configuration."""

    required: bool = True
    min_version: str = "TLS1.2"

    @classmethod
    def from_dict(cls, data: dict) -> "TLSConfig":
        """Create config from dictionary."""
        return cls(
            required=data.get("required", True),
            min_version=data.get("min_version", "TLS1.2"),
        )


@dataclass
class CryptoConfig:
    """
    Complete cryptographic configuration.

    Aggregates all crypto-related configuration including encryption,
    key exchange, signatures, and audit logging.

    Attributes:
        enabled: Whether the crypto layer is enabled.
        encryption: Encryption configuration.
        key_exchange: Key exchange configuration.
        signatures: Signature configuration.
        kdf: Key derivation configuration.
        api: API configuration for crypto endpoints.
        audit: Audit logging configuration.
        tls: TLS configuration.
    """

    enabled: bool = True
    encryption: EncryptionConfig = field(default_factory=EncryptionConfig)
    key_exchange: KeyExchangeConfig = field(default_factory=KeyExchangeConfig)
    signatures: SignatureConfig = field(default_factory=SignatureConfig)
    kdf: KDFConfig = field(default_factory=KDFConfig)
    api: CryptoAPIConfig = field(default_factory=CryptoAPIConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    tls: TLSConfig = field(default_factory=TLSConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "CryptoConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            CryptoConfig: Loaded configuration.
        """
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "CryptoConfig":
        """Create config from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            encryption=EncryptionConfig.from_dict(data.get("encryption", {})),
            key_exchange=KeyExchangeConfig.from_dict(data.get("key_exchange", {})),
            signatures=SignatureConfig.from_dict(data.get("signatures", {})),
            kdf=KDFConfig.from_dict(data.get("key_derivation", {})),
            api=CryptoAPIConfig.from_dict(data.get("api", {})),
            audit=AuditConfig.from_dict(data.get("audit", {})),
            tls=TLSConfig.from_dict(data.get("tls", {})),
        )

    @classmethod
    def from_env(cls) -> "CryptoConfig":
        """
        Create configuration from environment variables.

        Environment variables override YAML configuration values.

        Returns:
            CryptoConfig: Configured settings.
        """
        import logging

        logger = logging.getLogger("mfn.crypto.config")

        # Try to load from YAML first
        # Support configurable path via environment variable
        yaml_path_str = os.getenv("MFN_CRYPTO_CONFIG_PATH")
        if yaml_path_str:
            yaml_path = Path(yaml_path_str)
        else:
            # Default: look in configs/ relative to project root
            # Use package location to find project root
            yaml_path = Path(__file__).parent.parent.parent.parent.parent / "configs" / "crypto.yaml"

        if yaml_path.exists():
            try:
                config = cls.from_yaml(yaml_path)
                logger.debug(f"Loaded crypto config from {yaml_path}")
            except (FileNotFoundError, yaml.YAMLError, PermissionError) as e:
                logger.warning(f"Failed to load crypto config from {yaml_path}: {e}")
                config = cls()
        else:
            config = cls()

        # Override with environment variables
        enabled_env = os.getenv("MFN_CRYPTO_ENABLED", "").lower()
        if enabled_env:
            config.enabled = enabled_env in ("true", "1", "yes")

        cipher_env = os.getenv("MFN_CRYPTO_CIPHER_SUITE")
        if cipher_env:
            try:
                config.encryption.cipher_suite = CipherSuite(cipher_env)
            except ValueError:
                pass

        key_bits_env = os.getenv("MFN_CRYPTO_KEY_BITS")
        if key_bits_env:
            try:
                config.encryption.key_bits = int(key_bits_env)
            except ValueError:
                pass

        audit_enabled_env = os.getenv("MFN_CRYPTO_AUDIT_ENABLED", "").lower()
        if audit_enabled_env:
            config.audit.enabled = audit_enabled_env in ("true", "1", "yes")

        rate_limit_env = os.getenv("MFN_CRYPTO_RATE_LIMIT")
        if rate_limit_env:
            try:
                config.api.rate_limit = int(rate_limit_env)
            except ValueError:
                pass

        max_payload_env = os.getenv("MFN_CRYPTO_MAX_PAYLOAD_SIZE")
        if max_payload_env:
            try:
                config.api.max_payload_size = int(max_payload_env)
            except ValueError:
                pass

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
        _crypto_config = CryptoConfig.from_env()
    return _crypto_config


def reset_crypto_config() -> None:
    """Reset the crypto configuration singleton (useful for testing)."""
    global _crypto_config
    _crypto_config = None


def is_crypto_enabled() -> bool:
    """
    Check if the crypto layer is enabled.

    This function checks the MFN_CRYPTO_ENABLED environment variable
    and the configuration file to determine if cryptographic operations
    should be performed.

    Returns:
        bool: True if crypto is enabled, False otherwise.
    """
    return get_crypto_config().enabled


def generate_key_id(key_bytes: bytes) -> str:
    """
    Generate a non-reversible key identifier for logging.

    Uses SHA-256 truncated to first 8 characters for safe logging
    of key identifiers without exposing actual key material.

    Args:
        key_bytes: The key bytes to generate an ID for.

    Returns:
        str: 8-character hex key identifier.
    """
    return hashlib.sha256(key_bytes).hexdigest()[:8]


__all__ = [
    "CipherSuite",
    "KeyExchangeAlgorithm",
    "SignatureAlgorithm",
    "KDFAlgorithm",
    "EncryptionConfig",
    "KeyExchangeConfig",
    "SignatureConfig",
    "KDFConfig",
    "CryptoAPIConfig",
    "AuditConfig",
    "TLSConfig",
    "CryptoConfig",
    "get_crypto_config",
    "reset_crypto_config",
    "is_crypto_enabled",
    "generate_key_id",
]
