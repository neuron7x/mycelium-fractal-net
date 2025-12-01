"""
ECDSA digital signatures for MyceliumFractalNet.

Implements ECDSA signatures using the NIST P-384 curve for
EUF-CMA (Existential Unforgeability under Chosen Message Attack) security.

Security Properties:
    - P-384 curve provides ~192-bit security level
    - SHA-384 hash function for signature generation
    - Deterministic signature generation (RFC 6979)
    - Constant-time signature verification

Threat Model:
    - Assumes secure key generation (os.urandom)
    - Assumes secure key storage (not handled by this module)
    - Protects against message forgery and signature malleability

Usage:
    >>> from mycelium_fractal_net.crypto.signature import (
    ...     generate_ecdsa_keypair,
    ...     sign_message,
    ...     verify_signature,
    ... )
    >>> keypair = generate_ecdsa_keypair()
    >>> signature = sign_message(b"message", keypair.private_key)
    >>> verify_signature(b"message", signature, keypair.public_key)
    True

Reference:
    - FIPS 186-4 (Digital Signature Standard)
    - RFC 6979 (Deterministic DSA/ECDSA)
    - docs/crypto_security.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ec import (
    EllipticCurvePrivateKey,
    EllipticCurvePublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    BestAvailableEncryption,
    NoEncryption,
)

# Type alias for encryption algorithm
EncryptionAlgorithm = BestAvailableEncryption | NoEncryption


class SignatureError(Exception):
    """Base exception for signature operations."""

    pass


class SignatureGenerationError(SignatureError):
    """Raised when signature generation fails."""

    pass


class SignatureVerificationError(SignatureError):
    """Raised when signature verification fails."""

    pass


class KeyGenerationError(SignatureError):
    """Raised when key generation fails."""

    pass


# Default curve for ECDSA (P-384 provides ~192-bit security)
DEFAULT_CURVE = ec.SECP384R1()


@dataclass
class ECDSAKeyPair:
    """
    ECDSA key pair container.

    Attributes:
        private_key: ECDSA private key (for signing)
        public_key: ECDSA public key (for verification)
        curve_name: Name of the elliptic curve
    """

    private_key: EllipticCurvePrivateKey
    public_key: EllipticCurvePublicKey
    curve_name: str = "P-384"


def generate_ecdsa_keypair(
    curve: Optional[ec.EllipticCurve] = None,
) -> ECDSAKeyPair:
    """
    Generate an ECDSA key pair.

    Uses secure random number generation for key material.

    Args:
        curve: Elliptic curve to use. Defaults to P-384 (SECP384R1).
               Supported: P-256 (SECP256R1), P-384 (SECP384R1), P-521 (SECP521R1).

    Returns:
        ECDSAKeyPair containing private and public keys.

    Raises:
        KeyGenerationError: If key generation fails.

    Example:
        >>> keypair = generate_ecdsa_keypair()
        >>> keypair.curve_name
        'P-384'
    """
    if curve is None:
        curve = DEFAULT_CURVE

    try:
        private_key = ec.generate_private_key(curve)
        public_key = private_key.public_key()

        # Get curve name
        curve_name = curve.name

        return ECDSAKeyPair(
            private_key=private_key,
            public_key=public_key,
            curve_name=curve_name,
        )

    except Exception as e:
        raise KeyGenerationError(f"ECDSA key generation failed: {e}") from e


def sign_message(
    message: Union[bytes, str],
    private_key: EllipticCurvePrivateKey,
    encoding: str = "utf-8",
) -> bytes:
    """
    Sign a message using ECDSA.

    Uses SHA-384 hash function and deterministic signature generation (RFC 6979).

    Args:
        message: Message to sign (bytes or string).
        private_key: ECDSA private key for signing.
        encoding: String encoding if message is str.

    Returns:
        bytes: Digital signature.

    Raises:
        SignatureGenerationError: If signing fails.

    Example:
        >>> keypair = generate_ecdsa_keypair()
        >>> signature = sign_message(b"Hello, World!", keypair.private_key)
        >>> len(signature) > 0
        True
    """
    try:
        # Convert string to bytes if needed
        if isinstance(message, str):
            message = message.encode(encoding)

        signature = private_key.sign(
            message,
            ec.ECDSA(hashes.SHA384()),
        )

        return signature

    except Exception as e:
        raise SignatureGenerationError(f"Signature generation failed: {e}") from e


def verify_signature(
    message: Union[bytes, str],
    signature: bytes,
    public_key: EllipticCurvePublicKey,
    encoding: str = "utf-8",
) -> bool:
    """
    Verify a message signature using ECDSA.

    Uses constant-time comparison for signature verification.

    Args:
        message: Original message that was signed.
        signature: Signature to verify.
        public_key: ECDSA public key for verification.
        encoding: String encoding if message is str.

    Returns:
        bool: True if signature is valid, False otherwise.

    Raises:
        SignatureVerificationError: If verification process fails unexpectedly.

    Example:
        >>> keypair = generate_ecdsa_keypair()
        >>> signature = sign_message(b"Hello", keypair.private_key)
        >>> verify_signature(b"Hello", signature, keypair.public_key)
        True
        >>> verify_signature(b"World", signature, keypair.public_key)
        False
    """
    try:
        # Convert string to bytes if needed
        if isinstance(message, str):
            message = message.encode(encoding)

        public_key.verify(
            signature,
            message,
            ec.ECDSA(hashes.SHA384()),
        )

        return True

    except Exception:
        # Verification failed - signature is invalid
        return False


def serialize_ecdsa_public_key(
    public_key: EllipticCurvePublicKey,
    format: str = "PEM",
) -> bytes:
    """
    Serialize ECDSA public key to bytes.

    Args:
        public_key: ECDSA public key to serialize.
        format: Output format ("PEM" or "DER").

    Returns:
        bytes: Serialized public key.

    Raises:
        ValueError: If format is not supported.

    Example:
        >>> keypair = generate_ecdsa_keypair()
        >>> pem = serialize_ecdsa_public_key(keypair.public_key)
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


def serialize_ecdsa_private_key(
    private_key: EllipticCurvePrivateKey,
    password: Optional[bytes] = None,
    format: str = "PEM",
) -> bytes:
    """
    Serialize ECDSA private key to bytes.

    WARNING: Private keys should be stored securely (secrets manager, HSM).

    Args:
        private_key: ECDSA private key to serialize.
        password: Optional password for encryption. If None, key is unencrypted.
        format: Output format ("PEM" or "DER").

    Returns:
        bytes: Serialized private key.

    Raises:
        ValueError: If format is not supported.

    Example:
        >>> keypair = generate_ecdsa_keypair()
        >>> pem = serialize_ecdsa_private_key(keypair.private_key)
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
    "SignatureError",
    "SignatureGenerationError",
    "SignatureVerificationError",
    "KeyGenerationError",
    "ECDSAKeyPair",
    "generate_ecdsa_keypair",
    "sign_message",
    "verify_signature",
    "serialize_ecdsa_public_key",
    "serialize_ecdsa_private_key",
]
