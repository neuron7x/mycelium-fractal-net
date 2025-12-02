"""
Secure key store for cryptographic operations.

Provides in-memory storage for cryptographic keys with support for
key generation, retrieval, and automatic expiration.

Security Note:
    Keys are stored in memory and are lost on server restart.
    For production use, integrate with a Key Management System (KMS).

Reference: docs/MFN_CRYPTOGRAPHY.md
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple

from ..crypto import (
    ECDHKeyPair,
    SignatureKeyPair,
    generate_aes_key,
    generate_ecdh_keypair,
    generate_signature_keypair,
)
from .crypto_config import get_crypto_config
from .logging_config import get_logger

logger = get_logger("crypto.keystore")


@dataclass
class StoredKey:
    """
    Metadata for a stored cryptographic key.

    Attributes:
        key_id: Unique identifier for the key.
        key_type: Type of key (aes, ecdh, signing).
        created_at: Timestamp when key was created.
        expires_at: Timestamp when key expires (0 = never).
        algorithm: Algorithm associated with the key.
    """

    key_id: str
    key_type: str
    created_at: float
    expires_at: float
    algorithm: str


@dataclass
class KeyStore:
    """
    Secure in-memory key store.

    Provides thread-safe storage for cryptographic keys with automatic
    expiration and audit logging.

    Note: This is an in-memory implementation. For production, integrate
    with a proper Key Management System (KMS).

    Attributes:
        _aes_keys: Dictionary of AES symmetric keys.
        _ecdh_keys: Dictionary of ECDH key pairs.
        _signing_keys: Dictionary of Ed25519 signing key pairs.
        _key_metadata: Metadata for all stored keys.
        _lock: Thread lock for concurrent access.
    """

    _aes_keys: Dict[str, bytes] = field(default_factory=dict)
    _ecdh_keys: Dict[str, ECDHKeyPair] = field(default_factory=dict)
    _signing_keys: Dict[str, SignatureKeyPair] = field(default_factory=dict)
    _key_metadata: Dict[str, StoredKey] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _default_aes_key_id: Optional[str] = None
    _default_signing_key_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Initialize default keys on startup."""
        # Generate default AES key for encryption
        self._default_aes_key_id = self.generate_aes_key("default-aes")
        # Generate default signing key
        self._default_signing_key_id = self.generate_signing_keypair("default-signing")

    def _generate_key_id(self, prefix: str = "") -> str:
        """Generate a unique key identifier."""
        timestamp = int(time.time() * 1000)
        random_part = secrets.token_hex(8)
        if prefix:
            return f"{prefix}-{timestamp}-{random_part}"
        return f"{timestamp}-{random_part}"

    def _is_expired(self, key_id: str) -> bool:
        """Check if a key has expired."""
        metadata = self._key_metadata.get(key_id)
        if metadata is None:
            return True
        if metadata.expires_at == 0:
            return False
        return time.time() > metadata.expires_at

    def _cleanup_expired(self) -> None:
        """Remove expired keys from storage."""
        with self._lock:
            expired = [k for k in self._key_metadata if self._is_expired(k)]
            for key_id in expired:
                self._remove_key(key_id)

    def _remove_key(self, key_id: str) -> None:
        """Remove a key from all stores."""
        self._aes_keys.pop(key_id, None)
        self._ecdh_keys.pop(key_id, None)
        self._signing_keys.pop(key_id, None)
        self._key_metadata.pop(key_id, None)

    def _log_operation(
        self,
        operation: str,
        key_id: str,
        algorithm: str,
        success: bool = True,
        error_code: Optional[str] = None,
    ) -> None:
        """Log a cryptographic operation for auditing."""
        config = get_crypto_config()
        if not config.audit.enabled:
            return

        log_level = getattr(logging, config.audit.log_level.upper(), logging.INFO)

        log_data = {}
        if "operation" in config.audit.include_fields:
            log_data["operation"] = operation
        if "key_id" in config.audit.include_fields:
            log_data["key_id"] = key_id
        if "algorithm" in config.audit.include_fields:
            log_data["algorithm"] = algorithm
        if "timestamp" in config.audit.include_fields:
            log_data["timestamp"] = datetime.utcnow().isoformat() + "Z"
        if "success" in config.audit.include_fields:
            log_data["success"] = success
        if "error_code" in config.audit.include_fields and error_code:
            log_data["error_code"] = error_code

        logger.log(log_level, f"Crypto operation: {operation}", extra=log_data)

    # ==========================================================================
    # AES Key Management
    # ==========================================================================

    def generate_aes_key(
        self,
        key_id: Optional[str] = None,
        expiration_seconds: Optional[int] = None,
    ) -> str:
        """
        Generate and store a new AES-256 key.

        Args:
            key_id: Optional custom key identifier.
            expiration_seconds: Key expiration time (uses config default if None).

        Returns:
            str: The key identifier.
        """
        config = get_crypto_config()

        with self._lock:
            # Check key limit
            if len(self._aes_keys) >= config.key_management.max_stored_keys:
                self._cleanup_expired()
                if len(self._aes_keys) >= config.key_management.max_stored_keys:
                    raise RuntimeError("Maximum number of AES keys reached")

            # Generate key
            if key_id is None:
                key_id = self._generate_key_id("aes")

            key = generate_aes_key()
            self._aes_keys[key_id] = key

            # Store metadata
            exp_seconds = expiration_seconds or config.key_management.key_expiration_seconds
            expires_at = 0.0 if exp_seconds == 0 else time.time() + exp_seconds

            self._key_metadata[key_id] = StoredKey(
                key_id=key_id,
                key_type="aes",
                created_at=time.time(),
                expires_at=expires_at,
                algorithm="aes-256-gcm",
            )

            self._log_operation("generate_aes_key", key_id, "aes-256-gcm")

        return key_id

    def get_aes_key(self, key_id: Optional[str] = None) -> Tuple[str, bytes]:
        """
        Get an AES key by ID.

        Args:
            key_id: Key identifier. If None, returns default key.

        Returns:
            Tuple of (key_id, key_bytes).

        Raises:
            KeyError: If key not found or expired.
        """
        with self._lock:
            if key_id is None:
                key_id = self._default_aes_key_id

            if key_id is None:
                raise KeyError("No default AES key available")

            if self._is_expired(key_id):
                self._remove_key(key_id)
                raise KeyError(f"Key {key_id} has expired")

            key = self._aes_keys.get(key_id)
            if key is None:
                raise KeyError(f"AES key {key_id} not found")

            return key_id, key

    # ==========================================================================
    # ECDH Key Management
    # ==========================================================================

    def generate_ecdh_keypair(
        self,
        key_id: Optional[str] = None,
        expiration_seconds: Optional[int] = None,
    ) -> Tuple[str, bytes]:
        """
        Generate and store a new ECDH key pair.

        Args:
            key_id: Optional custom key identifier.
            expiration_seconds: Key expiration time (uses config default if None).

        Returns:
            Tuple of (key_id, public_key_bytes).
        """
        config = get_crypto_config()

        with self._lock:
            # Check key limit
            if len(self._ecdh_keys) >= config.key_management.max_stored_keys:
                self._cleanup_expired()
                if len(self._ecdh_keys) >= config.key_management.max_stored_keys:
                    raise RuntimeError("Maximum number of ECDH keys reached")

            # Generate keypair
            if key_id is None:
                key_id = self._generate_key_id("ecdh")

            keypair = generate_ecdh_keypair()
            self._ecdh_keys[key_id] = keypair

            # Store metadata
            exp_seconds = expiration_seconds or config.key_management.key_expiration_seconds
            expires_at = 0.0 if exp_seconds == 0 else time.time() + exp_seconds

            self._key_metadata[key_id] = StoredKey(
                key_id=key_id,
                key_type="ecdh",
                created_at=time.time(),
                expires_at=expires_at,
                algorithm="x25519",
            )

            self._log_operation("generate_ecdh_keypair", key_id, "x25519")

        return key_id, keypair.public_key

    def get_ecdh_keypair(self, key_id: str) -> ECDHKeyPair:
        """
        Get an ECDH key pair by ID.

        Args:
            key_id: Key identifier.

        Returns:
            ECDHKeyPair: The key pair.

        Raises:
            KeyError: If key not found or expired.
        """
        with self._lock:
            if self._is_expired(key_id):
                self._remove_key(key_id)
                raise KeyError(f"Key {key_id} has expired")

            keypair = self._ecdh_keys.get(key_id)
            if keypair is None:
                raise KeyError(f"ECDH key {key_id} not found")

            return keypair

    # ==========================================================================
    # Signing Key Management
    # ==========================================================================

    def generate_signing_keypair(
        self,
        key_id: Optional[str] = None,
        expiration_seconds: Optional[int] = None,
    ) -> str:
        """
        Generate and store a new Ed25519 signing key pair.

        Args:
            key_id: Optional custom key identifier.
            expiration_seconds: Key expiration time (uses config default if None).

        Returns:
            str: The key identifier.
        """
        config = get_crypto_config()

        with self._lock:
            # Check key limit
            if len(self._signing_keys) >= config.key_management.max_stored_keys:
                self._cleanup_expired()
                if len(self._signing_keys) >= config.key_management.max_stored_keys:
                    raise RuntimeError("Maximum number of signing keys reached")

            # Generate keypair
            if key_id is None:
                key_id = self._generate_key_id("sign")

            keypair = generate_signature_keypair()
            self._signing_keys[key_id] = keypair

            # Store metadata
            exp_seconds = expiration_seconds or config.key_management.key_expiration_seconds
            expires_at = 0.0 if exp_seconds == 0 else time.time() + exp_seconds

            self._key_metadata[key_id] = StoredKey(
                key_id=key_id,
                key_type="signing",
                created_at=time.time(),
                expires_at=expires_at,
                algorithm="ed25519",
            )

            self._log_operation("generate_signing_keypair", key_id, "ed25519")

        return key_id

    def get_signing_keypair(
        self, key_id: Optional[str] = None
    ) -> Tuple[str, SignatureKeyPair]:
        """
        Get a signing key pair by ID.

        Args:
            key_id: Key identifier. If None, returns default key.

        Returns:
            Tuple of (key_id, SignatureKeyPair).

        Raises:
            KeyError: If key not found or expired.
        """
        with self._lock:
            if key_id is None:
                key_id = self._default_signing_key_id

            if key_id is None:
                raise KeyError("No default signing key available")

            if self._is_expired(key_id):
                self._remove_key(key_id)
                raise KeyError(f"Key {key_id} has expired")

            keypair = self._signing_keys.get(key_id)
            if keypair is None:
                raise KeyError(f"Signing key {key_id} not found")

            return key_id, keypair

    # ==========================================================================
    # General Operations
    # ==========================================================================

    def get_key_count(self) -> Dict[str, int]:
        """
        Get the count of stored keys by type.

        Returns:
            Dict mapping key type to count.
        """
        with self._lock:
            self._cleanup_expired()
            return {
                "aes": len(self._aes_keys),
                "ecdh": len(self._ecdh_keys),
                "signing": len(self._signing_keys),
            }

    def delete_key(self, key_id: str) -> bool:
        """
        Delete a key from storage.

        Args:
            key_id: Key identifier to delete.

        Returns:
            bool: True if key was deleted, False if not found.
        """
        with self._lock:
            if key_id in self._key_metadata:
                key_type = self._key_metadata[key_id].key_type
                algorithm = self._key_metadata[key_id].algorithm
                self._remove_key(key_id)
                self._log_operation("delete_key", key_id, algorithm)
                return True
            return False

    def clear(self) -> None:
        """Clear all stored keys (for testing)."""
        with self._lock:
            self._aes_keys.clear()
            self._ecdh_keys.clear()
            self._signing_keys.clear()
            self._key_metadata.clear()
            self._default_aes_key_id = None
            self._default_signing_key_id = None


# Singleton instance
_key_store: Optional[KeyStore] = None


def get_key_store() -> KeyStore:
    """
    Get the key store singleton.

    Returns:
        KeyStore: The key store instance.
    """
    global _key_store
    if _key_store is None:
        _key_store = KeyStore()
    return _key_store


def reset_key_store() -> None:
    """Reset the key store singleton (for testing)."""
    global _key_store
    if _key_store is not None:
        _key_store.clear()
    _key_store = None


__all__ = [
    "StoredKey",
    "KeyStore",
    "get_key_store",
    "reset_key_store",
]
