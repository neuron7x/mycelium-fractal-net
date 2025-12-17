"""
Data encryption utilities for MyceliumFractalNet.

Implements authenticated encryption (AES-256-GCM) with URL-safe
encoding, suitable for protecting secrets and configuration data
at rest. The implementation prioritizes modern, vetted primitives
and explicit failure modes to support production-grade deployments.

Security Properties:
    - AES-256-GCM via ``cryptography`` with 96-bit nonces
    - Explicit key-size validation (256-bit keys only)
    - Authenticated decryption (fails closed on tampering or key mismatch)
    - Payload size guard to avoid memory exhaustion attacks
    - URL-safe base64 encoding for transport/storage

Usage:
    >>> from mycelium_fractal_net.security.encryption import (
    ...     encrypt_data,
    ...     decrypt_data,
    ...     generate_key,
    ... )
    >>> key = generate_key()
    >>> ciphertext = encrypt_data("sensitive data", key)
    >>> plaintext = decrypt_data(ciphertext, key)

Reference: docs/MFN_SECURITY.md, docs/SECURITY_READINESS.md
"""

from __future__ import annotations

import base64
import os
import secrets
from dataclasses import dataclass
from typing import Optional, Union

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class EncryptionError(Exception):
    """Raised when encryption or decryption fails."""

    pass


def generate_key() -> bytes:
    """
    Generate a cryptographically secure encryption key.

    Uses ``secrets.token_bytes`` for secure random generation.

    Returns:
        bytes: 32-byte encryption key (AES-256).

    Example:
        >>> key = generate_key()
        >>> len(key) == 32
        True
    """
    return secrets.token_bytes(32)


def encrypt_data(
    data: Union[str, bytes],
    key: bytes,
    encoding: str = "utf-8",
) -> str:
    """
    Encrypt data using AES-256-GCM with URL-safe base64 output.

    Args:
        data: Data to encrypt (string or bytes).
        key: 32-byte encryption key.
        encoding: String encoding (default: utf-8).

    Returns:
        str: Base64-encoded ciphertext containing nonce + encrypted payload.

    Raises:
        EncryptionError: If encryption fails.

    Example:
        >>> key = generate_key()
        >>> ciphertext = encrypt_data("secret", key)
        >>> isinstance(ciphertext, str)
        True
    """
    # Enforce key size and payload limits before any crypto operations
    if len(key) != 32:
        raise EncryptionError("Invalid key length: expected 32 bytes for AES-256-GCM")

    try:
        # Convert string to bytes if needed
        if isinstance(data, str):
            data = data.encode(encoding)

        if len(data) > 1_048_576:  # 1 MiB safety guard
            raise EncryptionError("Plaintext too large; refused to encrypt")

        nonce = os.urandom(12)  # 96-bit nonce per RFC 5116
        cipher = AESGCM(key)
        ciphertext = cipher.encrypt(nonce, data, associated_data=None)

        result = nonce + ciphertext
        return base64.urlsafe_b64encode(result).decode("ascii")

    except EncryptionError:
        raise
    except Exception as e:
        raise EncryptionError(f"Encryption failed: {e}") from e


def decrypt_data(
    ciphertext: str,
    key: bytes,
    encoding: str = "utf-8",
) -> str:
    """
    Decrypt data that was encrypted with encrypt_data.

    Verifies HMAC before decryption to prevent tampering.

    Args:
        ciphertext: Base64-encoded ciphertext.
        key: 32-byte encryption key (same as used for encryption).
        encoding: String encoding for result (default: utf-8).

    Returns:
        str: Decrypted plaintext.

    Raises:
        EncryptionError: If decryption fails or HMAC verification fails.

    Example:
        >>> key = generate_key()
        >>> ciphertext = encrypt_data("secret", key)
        >>> decrypt_data(ciphertext, key)
        'secret'
    """
    if len(key) != 32:
        raise EncryptionError("Invalid key length: expected 32 bytes for AES-256-GCM")

    try:
        # Decode base64
        decoded = base64.urlsafe_b64decode(ciphertext.encode("ascii"))

        # Minimum size: nonce(12) + auth tag(16)
        if len(decoded) < 28:
            raise EncryptionError("Invalid ciphertext: too short")

        nonce = decoded[:12]
        encrypted = decoded[12:]

        cipher = AESGCM(key)
        plaintext_bytes = cipher.decrypt(nonce, encrypted, associated_data=None)

        return plaintext_bytes.decode(encoding)

    except InvalidTag as exc:
        raise EncryptionError("Authentication failed: data tampered or wrong key") from exc
    except EncryptionError:
        raise
    except Exception as e:
        raise EncryptionError(f"Decryption failed: {e}") from e


@dataclass
class DataEncryptor:
    """
    Stateful encryptor for managing encryption operations.

    Provides a convenient wrapper around encryption functions
    with key management.

    Attributes:
        key: Encryption key (auto-generated if not provided).

    Example:
        >>> encryptor = DataEncryptor()
        >>> ciphertext = encryptor.encrypt("secret")
        >>> encryptor.decrypt(ciphertext)
        'secret'
    """

    key: bytes

    def __init__(self, key: Optional[bytes] = None) -> None:
        """
        Initialize encryptor.

        Args:
            key: Encryption key. If None, generates a new key.
        """
        self.key = key if key is not None else generate_key()

    def encrypt(self, data: Union[str, bytes]) -> str:
        """
        Encrypt data.

        Args:
            data: Data to encrypt.

        Returns:
            str: Encrypted ciphertext.
        """
        return encrypt_data(data, self.key)

    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt data.

        Args:
            ciphertext: Encrypted data.

        Returns:
            str: Decrypted plaintext.
        """
        return decrypt_data(ciphertext, self.key)


__all__ = [
    "EncryptionError",
    "generate_key",
    "encrypt_data",
    "decrypt_data",
    "DataEncryptor",
]
