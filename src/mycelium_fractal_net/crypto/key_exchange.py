"""
Elliptic Curve Diffie-Hellman key exchange for MyceliumFractalNet.

Implements X25519 ECDH for secure key exchange between parties.
The derived shared secret can be used with a KDF to produce symmetric keys.

Security Properties:
    - X25519 provides ~128-bit security level
    - Constant-time implementation prevents timing attacks
    - Forward secrecy when ephemeral keys are used
    - No small subgroup attacks (curve is twist-safe)

Threat Model:
    - Assumes secure key generation (os.urandom)
    - Assumes secure channel for public key exchange (or PKI)
    - Provides computational Diffie-Hellman (CDH) security

Usage:
    >>> from mycelium_fractal_net.crypto.key_exchange import (
    ...     generate_key_exchange_keypair,
    ...     perform_key_exchange,
    ... )
    >>> alice = generate_key_exchange_keypair()
    >>> bob = generate_key_exchange_keypair()
    >>> shared_alice = perform_key_exchange(alice.private_key, bob.public_key)
    >>> shared_bob = perform_key_exchange(bob.private_key, alice.public_key)
    >>> shared_alice == shared_bob
    True

Reference:
    - RFC 7748 (Elliptic Curves for Security)
    - RFC 8422 (Elliptic Curve Cryptography for TLS)
    - docs/crypto_security.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    BestAvailableEncryption,
    NoEncryption,
)

# Type alias for encryption algorithm
EncryptionAlgorithm = BestAvailableEncryption | NoEncryption


class KeyExchangeError(Exception):
    """Base exception for key exchange operations."""

    pass


class KeyGenerationError(KeyExchangeError):
    """Raised when key generation fails."""

    pass


class KeyExchangeOperationError(KeyExchangeError):
    """Raised when key exchange operation fails."""

    pass


@dataclass
class KeyExchangeKeyPair:
    """
    X25519 key pair container for ECDH key exchange.

    Attributes:
        private_key: X25519 private key
        public_key: X25519 public key
    """

    private_key: X25519PrivateKey
    public_key: X25519PublicKey


def generate_key_exchange_keypair() -> KeyExchangeKeyPair:
    """
    Generate an X25519 key pair for key exchange.

    Uses secure random number generation for key material.

    Returns:
        KeyExchangeKeyPair containing private and public keys.

    Raises:
        KeyGenerationError: If key generation fails.

    Example:
        >>> keypair = generate_key_exchange_keypair()
        >>> keypair.private_key is not None
        True
    """
    try:
        private_key = X25519PrivateKey.generate()
        public_key = private_key.public_key()

        return KeyExchangeKeyPair(
            private_key=private_key,
            public_key=public_key,
        )

    except Exception as e:
        raise KeyGenerationError(f"X25519 key generation failed: {e}") from e


def perform_key_exchange(
    private_key: X25519PrivateKey,
    peer_public_key: X25519PublicKey,
) -> bytes:
    """
    Perform X25519 ECDH key exchange.

    Derives a shared secret from your private key and peer's public key.
    The shared secret should be passed through a KDF before use as a
    symmetric key.

    Note on forward secrecy:
        Use ephemeral (newly generated) keys for each session to achieve
        forward secrecy. If long-term keys are compromised, previous
        sessions remain secure.

    Args:
        private_key: Your X25519 private key.
        peer_public_key: Peer's X25519 public key.

    Returns:
        bytes: 32-byte shared secret. Use with derive_key() for symmetric keys.

    Raises:
        KeyExchangeOperationError: If key exchange fails.

    Example:
        >>> alice = generate_key_exchange_keypair()
        >>> bob = generate_key_exchange_keypair()
        >>> shared_alice = perform_key_exchange(alice.private_key, bob.public_key)
        >>> shared_bob = perform_key_exchange(bob.private_key, alice.public_key)
        >>> shared_alice == shared_bob
        True
        >>> len(shared_alice) == 32
        True
    """
    try:
        shared_secret = private_key.exchange(peer_public_key)
        return shared_secret

    except Exception as e:
        raise KeyExchangeOperationError(f"Key exchange failed: {e}") from e


def serialize_key_exchange_public_key(
    public_key: X25519PublicKey,
    format: str = "RAW",
) -> bytes:
    """
    Serialize X25519 public key to bytes.

    Args:
        public_key: X25519 public key to serialize.
        format: Output format ("RAW" for 32-byte key, "PEM", or "DER").

    Returns:
        bytes: Serialized public key.

    Raises:
        ValueError: If format is not supported.

    Example:
        >>> keypair = generate_key_exchange_keypair()
        >>> raw = serialize_key_exchange_public_key(keypair.public_key, format="RAW")
        >>> len(raw) == 32
        True
    """
    if format.upper() == "RAW":
        return public_key.public_bytes_raw()
    elif format.upper() == "PEM":
        return public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    elif format.upper() == "DER":
        return public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'RAW', 'PEM', or 'DER'.")


def serialize_key_exchange_private_key(
    private_key: X25519PrivateKey,
    password: Optional[bytes] = None,
    format: str = "RAW",
) -> bytes:
    """
    Serialize X25519 private key to bytes.

    WARNING: Private keys should be stored securely (secrets manager, HSM).

    Args:
        private_key: X25519 private key to serialize.
        password: Optional password for encryption. If None, key is unencrypted.
                  Only used for PEM/DER formats.
        format: Output format ("RAW" for 32-byte key, "PEM", or "DER").

    Returns:
        bytes: Serialized private key.

    Raises:
        ValueError: If format is not supported.

    Example:
        >>> keypair = generate_key_exchange_keypair()
        >>> raw = serialize_key_exchange_private_key(keypair.private_key, format="RAW")
        >>> len(raw) == 32
        True
    """
    if format.upper() == "RAW":
        # Note: password not used for RAW format
        return private_key.private_bytes_raw()
    else:
        # Encryption algorithm for private key
        encryption: EncryptionAlgorithm
        if password is not None:
            encryption = serialization.BestAvailableEncryption(password)
        else:
            encryption = serialization.NoEncryption()

        if format.upper() == "PEM":
            return private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=encryption,
            )
        elif format.upper() == "DER":
            return private_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=encryption,
            )
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'RAW', 'PEM', or 'DER'.")


__all__ = [
    "KeyExchangeError",
    "KeyGenerationError",
    "KeyExchangeOperationError",
    "KeyExchangeKeyPair",
    "generate_key_exchange_keypair",
    "perform_key_exchange",
    "serialize_key_exchange_public_key",
    "serialize_key_exchange_private_key",
]
