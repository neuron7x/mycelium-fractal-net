"""
Data encryption utilities for MyceliumFractalNet.

**DEPRECATED**: This module uses a custom XOR-based encryption scheme and is
NOT suitable for production use. It is provided for development/testing only.

For production use, use the AES-256-GCM implementation in:
    mycelium_fractal_net.crypto.symmetric

This module will raise an error if used when MFN_ENV=prod.

Security Properties (DEV/TEST ONLY):
    - PBKDF2 key derivation with 100,000 iterations
    - XOR-based cipher (NOT cryptographically secure for production)
    - HMAC-SHA256 for authentication and tamper detection
    - URL-safe base64 encoding

Production Alternative:
    >>> from mycelium_fractal_net.crypto.symmetric import AESGCMCipher
    >>> cipher = AESGCMCipher()
    >>> ciphertext = cipher.encrypt(b"sensitive data")
    >>> plaintext = cipher.decrypt(ciphertext)

Reference: docs/MFN_SECURITY.md, docs/MFN_CRYPTOGRAPHY.md
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import warnings
from dataclasses import dataclass
from typing import Optional, Union


class EncryptionError(Exception):
    """Raised when encryption or decryption fails."""

    pass


def _check_production_guard() -> None:
    """
    Guard against using this deprecated encryption in production.
    
    Raises:
        RuntimeError: If MFN_ENV is set to 'prod' or 'production'.
    """
    env = os.getenv("MFN_ENV", "dev").lower()
    if env in ("prod", "production"):
        raise RuntimeError(
            "security.encryption module is deprecated and disabled in production. "
            "Use mycelium_fractal_net.crypto.symmetric.AESGCMCipher instead."
        )
    
    # Warn in all cases
    warnings.warn(
        "security.encryption is deprecated. Use crypto.symmetric.AESGCMCipher for production.",
        DeprecationWarning,
        stacklevel=3,
    )


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
    
    **DEPRECATED**: Use crypto.symmetric.AESGCMCipher for production.

    Uses a simplified encryption scheme based on:
    - Random 16-byte salt for key derivation
    - Random 16-byte IV
    - XOR encryption with derived key (NOT secure for production)
    - HMAC-SHA256 for authentication

    Args:
        data: Data to encrypt (string or bytes).
        key: 32-byte encryption key.
        encoding: String encoding (default: utf-8).

    Returns:
        str: Base64-encoded ciphertext with salt, IV, and HMAC.

    Raises:
        EncryptionError: If encryption fails.
        RuntimeError: If called in production environment.

    Example:
        >>> key = generate_key()
        >>> ciphertext = encrypt_data("secret", key)
        >>> isinstance(ciphertext, str)
        True
    """
    _check_production_guard()
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
    
    **DEPRECATED**: Use crypto.symmetric.AESGCMCipher for production.

    Verifies HMAC before decryption to prevent tampering.

    Args:
        ciphertext: Base64-encoded ciphertext.
        key: 32-byte encryption key (same as used for encryption).
        encoding: String encoding for result (default: utf-8).

    Returns:
        str: Decrypted plaintext.

    Raises:
        EncryptionError: If decryption fails or HMAC verification fails.
        RuntimeError: If called in production environment.

    Example:
        >>> key = generate_key()
        >>> ciphertext = encrypt_data("secret", key)
        >>> decrypt_data(ciphertext, key)
        'secret'
    """
    _check_production_guard()
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
