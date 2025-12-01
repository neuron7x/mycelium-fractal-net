"""
RSA asymmetric encryption for MyceliumFractalNet.

Implements RSA-4096 encryption with OAEP padding for secure data encryption.
Provides IND-CCA2 (Indistinguishability under Adaptive Chosen Ciphertext Attack)
security guarantees.

Security Properties:
    - RSA-4096 key size provides ~128-bit security level
    - OAEP padding with SHA-256 prevents common padding oracle attacks
    - MGF1 mask generation function with SHA-256

Threat Model:
    - Assumes secure key generation (os.urandom)
    - Assumes secure key storage (not handled by this module)
    - Protects against chosen-ciphertext attacks

Usage:
    >>> from mycelium_fractal_net.crypto.asymmetric import (
    ...     generate_rsa_keypair,
    ...     rsa_encrypt,
    ...     rsa_decrypt,
    ... )
    >>> keypair = generate_rsa_keypair()
    >>> ciphertext = rsa_encrypt(b"secret data", keypair.public_key)
    >>> plaintext = rsa_decrypt(ciphertext, keypair.private_key)

Reference:
    - PKCS#1 v2.1 (RFC 8017)
    - NIST SP 800-56B Rev. 2
    - docs/crypto_security.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.rsa import (
    RSAPrivateKey,
    RSAPublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    BestAvailableEncryption,
    NoEncryption,
)

# Type alias for encryption algorithm
EncryptionAlgorithm = BestAvailableEncryption | NoEncryption


class CryptoError(Exception):
    """Base exception for cryptographic operations."""

    pass


class KeyGenerationError(CryptoError):
    """Raised when key generation fails."""

    pass


class EncryptionError(CryptoError):
    """Raised when encryption fails."""

    pass


class DecryptionError(CryptoError):
    """Raised when decryption fails."""

    pass


# Default RSA key size for production use (128-bit security level)
DEFAULT_RSA_KEY_SIZE = 4096

# Public exponent (F4 = 65537, widely used and secure)
DEFAULT_PUBLIC_EXPONENT = 65537


@dataclass
class RSAKeyPair:
    """
    RSA key pair container.

    Attributes:
        private_key: RSA private key (for decryption and signing)
        public_key: RSA public key (for encryption and verification)
        key_size: Key size in bits (default: 4096)
    """

    private_key: RSAPrivateKey
    public_key: RSAPublicKey
    key_size: int = DEFAULT_RSA_KEY_SIZE


def generate_rsa_keypair(
    key_size: int = DEFAULT_RSA_KEY_SIZE,
    public_exponent: int = DEFAULT_PUBLIC_EXPONENT,
) -> RSAKeyPair:
    """
    Generate an RSA key pair.

    Uses secure random number generation for key material.

    Args:
        key_size: Key size in bits. Must be at least 2048.
                  Recommended: 4096 for production use.
        public_exponent: Public exponent. Recommended: 65537 (0x10001).

    Returns:
        RSAKeyPair containing private and public keys.

    Raises:
        KeyGenerationError: If key generation fails.
        ValueError: If key_size < 2048.

    Example:
        >>> keypair = generate_rsa_keypair()
        >>> keypair.key_size
        4096
    """
    if key_size < 2048:
        raise ValueError("RSA key size must be at least 2048 bits for security")

    try:
        private_key = rsa.generate_private_key(
            public_exponent=public_exponent,
            key_size=key_size,
        )
        public_key = private_key.public_key()

        return RSAKeyPair(
            private_key=private_key,
            public_key=public_key,
            key_size=key_size,
        )

    except Exception as e:
        raise KeyGenerationError(f"RSA key generation failed: {e}") from e


def _get_oaep_padding() -> padding.OAEP:
    """
    Get OAEP padding configuration.

    Uses SHA-256 for both hash and MGF1 for maximum security.
    """
    return padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None,
    )


def rsa_encrypt(
    plaintext: Union[bytes, str],
    public_key: RSAPublicKey,
    encoding: str = "utf-8",
) -> bytes:
    """
    Encrypt data using RSA-OAEP.

    Uses OAEP padding with SHA-256 for IND-CCA2 security.

    Note on message size:
        RSA-OAEP with SHA-256 can encrypt up to (key_size/8 - 66) bytes.
        For RSA-4096: max 446 bytes.
        For larger data, use hybrid encryption (encrypt symmetric key with RSA).

    Args:
        plaintext: Data to encrypt (bytes or string).
        public_key: RSA public key for encryption.
        encoding: String encoding if plaintext is str.

    Returns:
        bytes: Encrypted ciphertext.

    Raises:
        EncryptionError: If encryption fails.
        ValueError: If plaintext is too large for the key size.

    Example:
        >>> keypair = generate_rsa_keypair()
        >>> ciphertext = rsa_encrypt(b"secret", keypair.public_key)
        >>> len(ciphertext) == 512  # 4096 bits = 512 bytes
        True
    """
    try:
        # Convert string to bytes if needed
        if isinstance(plaintext, str):
            plaintext = plaintext.encode(encoding)

        # Check message size
        key_size_bytes = public_key.key_size // 8
        max_message_size = key_size_bytes - 2 * 32 - 2  # OAEP overhead with SHA-256
        if len(plaintext) > max_message_size:
            raise ValueError(
                f"Plaintext too large ({len(plaintext)} bytes). "
                f"Maximum for RSA-{public_key.key_size} with OAEP-SHA256: "
                f"{max_message_size} bytes. Use hybrid encryption for larger data."
            )

        ciphertext = public_key.encrypt(
            plaintext,
            _get_oaep_padding(),
        )

        return ciphertext

    except ValueError:
        raise
    except Exception as e:
        raise EncryptionError(f"RSA encryption failed: {e}") from e


def rsa_decrypt(
    ciphertext: bytes,
    private_key: RSAPrivateKey,
    encoding: Optional[str] = None,
) -> Union[bytes, str]:
    """
    Decrypt data using RSA-OAEP.

    Verifies OAEP padding and decrypts the ciphertext.

    Args:
        ciphertext: Encrypted data to decrypt.
        private_key: RSA private key for decryption.
        encoding: If provided, decode result to string with this encoding.

    Returns:
        Decrypted plaintext (bytes or str if encoding provided).

    Raises:
        DecryptionError: If decryption fails (invalid ciphertext or key).

    Example:
        >>> keypair = generate_rsa_keypair()
        >>> ciphertext = rsa_encrypt(b"secret", keypair.public_key)
        >>> plaintext = rsa_decrypt(ciphertext, keypair.private_key)
        >>> plaintext
        b'secret'
    """
    try:
        plaintext = private_key.decrypt(
            ciphertext,
            _get_oaep_padding(),
        )

        if encoding is not None:
            return plaintext.decode(encoding)

        return plaintext

    except Exception as e:
        raise DecryptionError(f"RSA decryption failed: {e}") from e


def serialize_public_key(
    public_key: RSAPublicKey,
    format: str = "PEM",
) -> bytes:
    """
    Serialize RSA public key to bytes.

    Args:
        public_key: RSA public key to serialize.
        format: Output format ("PEM" or "DER").

    Returns:
        bytes: Serialized public key.

    Raises:
        ValueError: If format is not supported.

    Example:
        >>> keypair = generate_rsa_keypair()
        >>> pem = serialize_public_key(keypair.public_key)
        >>> pem.startswith(b'-----BEGIN PUBLIC KEY-----')
        True
    """
    if format.upper() == "PEM":
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
        raise ValueError(f"Unsupported format: {format}. Use 'PEM' or 'DER'.")


def serialize_private_key(
    private_key: RSAPrivateKey,
    password: Optional[bytes] = None,
    format: str = "PEM",
) -> bytes:
    """
    Serialize RSA private key to bytes.

    WARNING: Private keys should be stored securely (secrets manager, HSM).

    Args:
        private_key: RSA private key to serialize.
        password: Optional password for encryption. If None, key is unencrypted.
        format: Output format ("PEM" or "DER").

    Returns:
        bytes: Serialized private key.

    Raises:
        ValueError: If format is not supported.

    Example:
        >>> keypair = generate_rsa_keypair()
        >>> pem = serialize_private_key(keypair.private_key)
        >>> pem.startswith(b'-----BEGIN PRIVATE KEY-----')
        True
    """
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
        raise ValueError(f"Unsupported format: {format}. Use 'PEM' or 'DER'.")


__all__ = [
    "CryptoError",
    "KeyGenerationError",
    "EncryptionError",
    "DecryptionError",
    "RSAKeyPair",
    "generate_rsa_keypair",
    "rsa_encrypt",
    "rsa_decrypt",
    "serialize_public_key",
    "serialize_private_key",
    "DEFAULT_RSA_KEY_SIZE",
]
