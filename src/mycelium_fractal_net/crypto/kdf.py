"""
Key Derivation Functions for MyceliumFractalNet.

Implements HKDF (HMAC-based Key Derivation Function) following RFC 5869
for deriving cryptographic keys from input key material.

Security Properties:
    - HKDF-SHA256 provides secure key derivation
    - Salt-based derivation prevents rainbow table attacks
    - Info parameter allows domain separation
    - Follows NIST SP 800-56C guidelines

Threat Model:
    - Assumes input key material has sufficient entropy
    - Assumes salt is random and unique per derivation context
    - Provides cryptographic independence between derived keys

Usage:
    >>> from mycelium_fractal_net.crypto.kdf import (
    ...     derive_key,
    ...     generate_salt,
    ... )
    >>> salt = generate_salt()
    >>> key = derive_key(b"password", salt, length=32)

Reference:
    - RFC 5869 (HKDF)
    - NIST SP 800-56C
    - docs/crypto_security.md
"""

from __future__ import annotations

import os
from typing import Optional, Union

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


class KeyDerivationError(Exception):
    """Raised when key derivation fails."""

    pass


# Default salt length (16 bytes = 128 bits)
DEFAULT_SALT_LENGTH = 16

# Default derived key length (32 bytes = 256 bits)
DEFAULT_KEY_LENGTH = 32

# Default HKDF algorithm
DEFAULT_ALGORITHM = hashes.SHA256()


def generate_salt(length: int = DEFAULT_SALT_LENGTH) -> bytes:
    """
    Generate a cryptographically secure random salt.

    Uses os.urandom for secure random generation.

    Args:
        length: Salt length in bytes. Minimum 16 bytes recommended.

    Returns:
        bytes: Random salt.

    Raises:
        ValueError: If length < 8.

    Example:
        >>> salt = generate_salt()
        >>> len(salt) == 16
        True
    """
    if length < 8:
        raise ValueError("Salt length must be at least 8 bytes for security")

    return os.urandom(length)


def derive_key(
    input_key_material: Union[bytes, str],
    salt: Optional[bytes] = None,
    info: Optional[bytes] = None,
    length: int = DEFAULT_KEY_LENGTH,
    encoding: str = "utf-8",
) -> bytes:
    """
    Derive a cryptographic key using HKDF-SHA256.

    HKDF provides a standardized way to derive cryptographic keys
    from input key material (IKM) such as passwords or shared secrets.

    Args:
        input_key_material: Source material for key derivation.
                           Can be a password, shared secret, or other key material.
        salt: Random salt value. If None, uses a zero-filled salt.
              For best security, use generate_salt() and store with derived key.
        info: Optional context/application-specific info for domain separation.
              Example: b"encryption-key" vs b"signing-key"
        length: Desired output key length in bytes. Default: 32 (256 bits).
        encoding: String encoding if input_key_material is str.

    Returns:
        bytes: Derived cryptographic key.

    Raises:
        KeyDerivationError: If key derivation fails.
        ValueError: If length is invalid.

    Example:
        >>> salt = generate_salt()
        >>> key = derive_key(b"shared_secret", salt, info=b"encryption-key", length=32)
        >>> len(key) == 32
        True

    Note:
        - Same input_key_material, salt, and info will always produce the same key
        - Different salt values will produce different keys
        - Use info parameter to derive multiple independent keys from same material
    """
    if length < 16:
        raise ValueError("Derived key length must be at least 16 bytes for security")

    if length > 255 * 32:  # HKDF-SHA256 output limit
        raise ValueError(f"Derived key length cannot exceed {255 * 32} bytes")

    try:
        # Convert string to bytes if needed
        if isinstance(input_key_material, str):
            input_key_material = input_key_material.encode(encoding)

        # Use zero salt if not provided (HKDF allows this but not recommended)
        if salt is None:
            salt = bytes(DEFAULT_SALT_LENGTH)

        # Create HKDF instance
        hkdf = HKDF(
            algorithm=DEFAULT_ALGORITHM,
            length=length,
            salt=salt,
            info=info or b"",
        )

        # Derive key
        derived_key = hkdf.derive(input_key_material)

        return derived_key

    except ValueError:
        raise
    except Exception as e:
        raise KeyDerivationError(f"Key derivation failed: {e}") from e


def derive_multiple_keys(
    input_key_material: Union[bytes, str],
    salt: bytes,
    key_specs: list[tuple[bytes, int]],
    encoding: str = "utf-8",
) -> list[bytes]:
    """
    Derive multiple independent keys from the same input material.

    Uses different info parameters to ensure cryptographic independence
    between derived keys.

    Args:
        input_key_material: Source material for key derivation.
        salt: Random salt value (should be stored with derived keys).
        key_specs: List of (info, length) tuples specifying each key.
        encoding: String encoding if input_key_material is str.

    Returns:
        list[bytes]: List of derived keys in order matching key_specs.

    Raises:
        KeyDerivationError: If any key derivation fails.

    Example:
        >>> salt = generate_salt()
        >>> keys = derive_multiple_keys(
        ...     b"master_secret",
        ...     salt,
        ...     [(b"encryption", 32), (b"signing", 64)],
        ... )
        >>> len(keys) == 2
        True
        >>> len(keys[0]) == 32
        True
        >>> len(keys[1]) == 64
        True
    """
    derived_keys = []

    for info, length in key_specs:
        key = derive_key(
            input_key_material,
            salt=salt,
            info=info,
            length=length,
            encoding=encoding,
        )
        derived_keys.append(key)

    return derived_keys


__all__ = [
    "KeyDerivationError",
    "generate_salt",
    "derive_key",
    "derive_multiple_keys",
    "DEFAULT_SALT_LENGTH",
    "DEFAULT_KEY_LENGTH",
]
