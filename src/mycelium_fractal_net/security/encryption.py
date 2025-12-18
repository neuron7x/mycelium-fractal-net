"""
Data encryption utilities for MyceliumFractalNet.

Implements authenticated encryption using AES-256-GCM from the
``cryptography`` library. This replaces the previous XOR-based
scheme, eliminating replay/tampering risks and providing modern
confidentiality + integrity guarantees.

Security Properties:
    - AES-256-GCM (AEAD) with 96-bit nonces and 128-bit tags
    - URL-safe base64 encoding for transport
    - Strict 32-byte key requirement to enforce AES-256

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
import binascii
from dataclasses import dataclass
from typing import Optional, Union

from mycelium_fractal_net.crypto import (
    AESGCMCipher,
    AES_KEY_SIZE,
    GCM_NONCE_SIZE,
    GCM_TAG_SIZE,
    SymmetricEncryptionError,
    generate_aes_key,
)


class EncryptionError(Exception):
    """Raised when encryption or decryption fails."""

    pass

def generate_key(length: int = AES_KEY_SIZE) -> bytes:
    """
    Generate a cryptographically secure AES key.

    Args:
        length: Key length in bytes (default: 32 for AES-256).

    Returns:
        bytes: Encryption key of the requested length.
    """

    try:
        return generate_aes_key(length)
    except SymmetricEncryptionError as exc:  # pragma: no cover - validated upstream
        raise EncryptionError(str(exc)) from exc


def _validate_key(key: bytes | bytearray) -> bytes:
    """Validate that the encryption key is bytes and AES-256 sized."""

    if not isinstance(key, (bytes, bytearray)):
        raise EncryptionError("Encryption key must be bytes")

    normalized = bytes(key)
    if len(normalized) != AES_KEY_SIZE:
        raise EncryptionError(f"Encryption key must be exactly {AES_KEY_SIZE} bytes")

    return normalized


def encrypt_data(
    data: Union[str, bytes],
    key: bytes,
    encoding: str = "utf-8",
) -> str:
    """
    Encrypt data using symmetric encryption.

    Uses AES-256-GCM authenticated encryption.

    Args:
        data: Data to encrypt (string or bytes).
        key: 32-byte AES-256 encryption key.
        encoding: String encoding (default: utf-8).

    Returns:
        str: URL-safe base64-encoded ciphertext including nonce and tag.

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
        elif not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes or string")

        cipher = AESGCMCipher(key=key)
        ciphertext = cipher.encrypt(bytes(data))

        # Return URL-safe base64 encoding
        return base64.urlsafe_b64encode(ciphertext).decode("ascii")

    except (TypeError, ValueError) as exc:
        raise EncryptionError(f"Encryption failed: {exc}") from exc
    except SymmetricEncryptionError as exc:
        raise EncryptionError(f"Encryption failed: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise EncryptionError(f"Encryption failed: {exc}") from exc


def decrypt_data(
    ciphertext: str,
    key: bytes,
    encoding: str = "utf-8",
) -> str:
    """
    Decrypt data that was encrypted with encrypt_data.

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
        raw_ciphertext = base64.urlsafe_b64decode(ciphertext.encode("ascii"))

        # Validate minimum length for AES-GCM (nonce + tag)
        min_length = GCM_NONCE_SIZE + GCM_TAG_SIZE
        if len(raw_ciphertext) < min_length:
            raise EncryptionError("Invalid ciphertext: too short for AES-GCM")

        cipher = AESGCMCipher(key=key)
        plaintext_bytes = cipher.decrypt(raw_ciphertext, return_bytes=True)

        return plaintext_bytes.decode(encoding)

    except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
        raise EncryptionError("Invalid ciphertext encoding") from exc
    except SymmetricEncryptionError as exc:
        raise EncryptionError("Decryption failed: authentication error") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise EncryptionError(f"Decryption failed: {exc}") from exc


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
