"""
Data encryption utilities for MyceliumFractalNet.

Provides symmetric encryption for sensitive data at rest using a
key derivation and XOR-based cipher with HMAC authentication.
Suitable for encrypting configuration secrets and API keys.

NOTE: For high-security production use cases requiring regulatory
compliance (PCI-DSS, HIPAA), consider using the `cryptography`
library with Fernet or AES-GCM encryption.

Security Properties:
    - PBKDF2 key derivation with 100,000 iterations
    - SHA256-based cipher key generation
    - HMAC-SHA256 for authentication and tamper detection
    - URL-safe base64 encoding

Usage:
    >>> from mycelium_fractal_net.security.encryption import (
    ...     encrypt_data,
    ...     decrypt_data,
    ...     generate_key,
    ... )
    >>> key = generate_key()
    >>> ciphertext = encrypt_data("sensitive data", key)
    >>> plaintext = decrypt_data(ciphertext, key)

Reference: docs/MFN_SECURITY.md
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from typing import Optional, Union


class EncryptionError(Exception):
    """Raised when encryption or decryption fails."""

    pass


def generate_key() -> bytes:
    """
    Generate a cryptographically secure encryption key.

    Uses os.urandom for secure random generation.

    Returns:
        bytes: 32-byte encryption key suitable for Fernet.

    Example:
        >>> key = generate_key()
        >>> len(key) == 32
        True
    """
    return secrets.token_bytes(32)


def _validate_key(key: bytes | bytearray) -> bytes:
    """
    Validate that the encryption key is sufficiently strong.

    Args:
        key: Candidate key material.

    Returns:
        bytes: Normalized key bytes.

    Raises:
        EncryptionError: If the key is not bytes-like or too short.
    """
    if not isinstance(key, (bytes, bytearray)):
        raise EncryptionError("Encryption key must be bytes")
    normalized = bytes(key)
    if len(normalized) < 32:
        raise EncryptionError("Encryption key must be at least 32 bytes")
    return normalized


def _derive_key(key: bytes, salt: bytes) -> bytes:
    """
    Derive an encryption key from the master key using PBKDF2.

    Args:
        key: Master encryption key.
        salt: Random salt for key derivation.

    Returns:
        bytes: Derived 32-byte key.
    """
    return hashlib.pbkdf2_hmac("sha256", key, salt, iterations=100000, dklen=32)


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    """
    XOR data with key (repeating key if necessary).

    This is a simple encryption primitive used as part of the
    encryption scheme. NOT secure on its own.

    Args:
        data: Data to XOR.
        key: Key bytes.

    Returns:
        bytes: XORed result.
    """
    key_len = len(key)
    return bytes(d ^ key[i % key_len] for i, d in enumerate(data))


def encrypt_data(
    data: Union[str, bytes],
    key: bytes,
    encoding: str = "utf-8",
) -> str:
    """
    Encrypt data using symmetric encryption.

    Uses a simplified encryption scheme based on:
    - Random 16-byte salt for key derivation
    - Random 16-byte IV
    - XOR encryption with derived key
    - HMAC-SHA256 for authentication

    Args:
        data: Data to encrypt (string or bytes).
        key: 32-byte encryption key.
        encoding: String encoding (default: utf-8).

    Returns:
        str: Base64-encoded ciphertext with salt, IV, and HMAC.

    Raises:
        EncryptionError: If encryption fails.

    Example:
        >>> key = generate_key()
        >>> ciphertext = encrypt_data("secret", key)
        >>> isinstance(ciphertext, str)
        True
    """
    key = _validate_key(key)
    try:
        # Convert string to bytes if needed
        if isinstance(data, str):
            data = data.encode(encoding)

        # Generate random salt and IV
        salt = os.urandom(16)
        iv = os.urandom(16)

        # Derive encryption key
        derived_key = _derive_key(key, salt)

        # Create cipher stream by combining IV with derived key
        cipher_key = hashlib.sha256(iv + derived_key).digest()

        # Encrypt data using XOR with cipher key
        encrypted = _xor_bytes(data, cipher_key)

        # Create HMAC for authentication
        mac = hmac.new(derived_key, salt + iv + encrypted, hashlib.sha256).digest()

        # Combine: salt + iv + encrypted + mac
        result = salt + iv + encrypted + mac

        # Return base64-encoded result
        return base64.urlsafe_b64encode(result).decode("ascii")

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
    key = _validate_key(key)
    try:
        # Decode base64
        data = base64.urlsafe_b64decode(ciphertext.encode("ascii"))

        # Minimum size: salt(16) + iv(16) + mac(32) = 64 bytes
        if len(data) < 64:
            raise EncryptionError("Invalid ciphertext: too short")

        # Extract components
        salt = data[:16]
        iv = data[16:32]
        mac = data[-32:]
        encrypted = data[32:-32]

        # Derive encryption key
        derived_key = _derive_key(key, salt)

        # Verify HMAC
        expected_mac = hmac.new(
            derived_key, salt + iv + encrypted, hashlib.sha256
        ).digest()
        if not hmac.compare_digest(mac, expected_mac):
            raise EncryptionError("HMAC verification failed: data may be tampered")

        # Create cipher stream
        cipher_key = hashlib.sha256(iv + derived_key).digest()

        # Decrypt using XOR
        decrypted = _xor_bytes(encrypted, cipher_key)

        return decrypted.decode(encoding)

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
        self.key = _validate_key(key) if key is not None else generate_key()

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
