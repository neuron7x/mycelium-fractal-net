"""
Pydantic schemas for Cryptographic API endpoints.

Provides request/response models for the crypto API including encryption,
decryption, signing, verification, and key pair generation.

Reference: docs/MFN_CRYPTOGRAPHY.md
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class KeyType(str, Enum):
    """Type of cryptographic key pair to generate."""

    ECDH = "ecdh"
    ED25519 = "ed25519"


# =============================================================================
# Encryption / Decryption
# =============================================================================


class EncryptRequest(BaseModel):
    """
    Request for data encryption.

    Attributes:
        data: Base64-encoded plaintext data to encrypt.
        key_id: Optional key identifier for server-side key.
               If not provided, uses the default server key.
        public_key: Optional base64-encoded recipient's public key (for ECDH).
               If provided, derives shared secret for encryption.
        associated_data: Optional base64-encoded AAD (Additional Authenticated Data).
    """

    data: str = Field(
        ...,
        min_length=1,
        max_length=14_000_000,  # ~10MB base64 encoded
        description="Base64-encoded plaintext data to encrypt",
    )
    key_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional key identifier for server-side key",
    )
    public_key: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional base64-encoded recipient's public key",
    )
    associated_data: Optional[str] = Field(
        default=None,
        max_length=1024,
        description="Optional base64-encoded additional authenticated data",
    )

    @field_validator("data")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate that data is valid base64."""
        import base64

        try:
            base64.b64decode(v)
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {e}") from e
        return v


class EncryptResponse(BaseModel):
    """
    Response from encryption operation.

    Attributes:
        ciphertext: Base64-encoded encrypted data (includes nonce and tag).
        key_id: Key identifier used for encryption.
        algorithm: Algorithm used for encryption.
    """

    ciphertext: str = Field(..., description="Base64-encoded encrypted data")
    key_id: str = Field(..., description="Key identifier used for encryption")
    algorithm: str = Field(..., description="Encryption algorithm used")


class DecryptRequest(BaseModel):
    """
    Request for data decryption.

    Attributes:
        ciphertext: Base64-encoded ciphertext to decrypt.
        key_id: Key identifier for the decryption key.
        associated_data: Optional base64-encoded AAD (must match encryption).
    """

    ciphertext: str = Field(
        ...,
        min_length=1,
        max_length=14_000_000,  # ~10MB base64 encoded
        description="Base64-encoded ciphertext to decrypt",
    )
    key_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Key identifier for decryption",
    )
    associated_data: Optional[str] = Field(
        default=None,
        max_length=1024,
        description="Optional base64-encoded additional authenticated data",
    )

    @field_validator("ciphertext")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate that ciphertext is valid base64."""
        import base64

        try:
            base64.b64decode(v)
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {e}") from e
        return v


class DecryptResponse(BaseModel):
    """
    Response from decryption operation.

    Attributes:
        data: Base64-encoded decrypted plaintext.
        key_id: Key identifier used for decryption.
        algorithm: Algorithm used for decryption.
    """

    data: str = Field(..., description="Base64-encoded decrypted plaintext")
    key_id: str = Field(..., description="Key identifier used for decryption")
    algorithm: str = Field(..., description="Decryption algorithm used")


# =============================================================================
# Digital Signatures
# =============================================================================


class SignRequest(BaseModel):
    """
    Request for message signing.

    Attributes:
        message: Base64-encoded message or hash to sign.
        key_id: Optional key identifier for the signing key.
               If not provided, uses the default server signing key.
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=14_000_000,  # ~10MB base64 encoded
        description="Base64-encoded message to sign",
    )
    key_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional key identifier for signing key",
    )

    @field_validator("message")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate that message is valid base64."""
        import base64

        try:
            base64.b64decode(v)
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {e}") from e
        return v


class SignResponse(BaseModel):
    """
    Response from signing operation.

    Attributes:
        signature: Base64-encoded digital signature.
        key_id: Key identifier of the signing key.
        algorithm: Signature algorithm used.
        public_key: Base64-encoded public key for verification.
    """

    signature: str = Field(..., description="Base64-encoded digital signature")
    key_id: str = Field(..., description="Key identifier of the signing key")
    algorithm: str = Field(..., description="Signature algorithm used")
    public_key: str = Field(
        ..., description="Base64-encoded public key for verification"
    )


class VerifyRequest(BaseModel):
    """
    Request for signature verification.

    Attributes:
        message: Base64-encoded original message.
        signature: Base64-encoded signature to verify.
        public_key: Optional base64-encoded public key for verification.
        key_id: Optional key identifier (if public_key not provided).
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=14_000_000,  # ~10MB base64 encoded
        description="Base64-encoded original message",
    )
    signature: str = Field(
        ...,
        min_length=1,
        max_length=256,  # Ed25519 signature is 64 bytes -> ~88 chars base64
        description="Base64-encoded signature to verify",
    )
    public_key: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Base64-encoded public key for verification",
    )
    key_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Key identifier (if public_key not provided)",
    )

    @field_validator("message", "signature")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate that input is valid base64."""
        import base64

        try:
            base64.b64decode(v)
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {e}") from e
        return v


class VerifyResponse(BaseModel):
    """
    Response from verification operation.

    Attributes:
        valid: True if signature is valid, False otherwise.
        algorithm: Signature algorithm used.
        key_id: Key identifier used for verification (if applicable).
    """

    valid: bool = Field(..., description="True if signature is valid")
    algorithm: str = Field(..., description="Signature algorithm used")
    key_id: Optional[str] = Field(
        default=None, description="Key identifier used for verification"
    )


# =============================================================================
# Key Pair Generation
# =============================================================================


class KeyPairRequest(BaseModel):
    """
    Request for key pair generation.

    Attributes:
        key_type: Type of key pair to generate (ecdh or ed25519).
        key_id: Optional custom key identifier.
               If not provided, a random ID is generated.
    """

    key_type: KeyType = Field(
        ...,
        description="Type of key pair to generate (ecdh or ed25519)",
    )
    key_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Optional custom key identifier",
    )


class KeyPairResponse(BaseModel):
    """
    Response from key pair generation.

    The private key is stored securely on the server and is NOT returned.

    Attributes:
        key_id: Unique identifier for the generated key pair.
        public_key: Base64-encoded public key.
        key_type: Type of key pair generated.
        algorithm: Algorithm used for key generation.
    """

    key_id: str = Field(..., description="Unique identifier for the key pair")
    public_key: str = Field(..., description="Base64-encoded public key")
    key_type: KeyType = Field(..., description="Type of key pair generated")
    algorithm: str = Field(..., description="Algorithm used for key generation")


# =============================================================================
# Crypto Status
# =============================================================================


class CryptoStatusResponse(BaseModel):
    """
    Response for crypto module status check.

    Attributes:
        enabled: Whether crypto operations are enabled.
        algorithms: Dictionary of supported algorithms.
        key_count: Number of keys currently stored.
    """

    enabled: bool = Field(..., description="Whether crypto module is enabled")
    algorithms: dict = Field(..., description="Supported algorithms")
    key_count: dict = Field(..., description="Number of keys stored by type")


__all__ = [
    "KeyType",
    # Encryption/Decryption
    "EncryptRequest",
    "EncryptResponse",
    "DecryptRequest",
    "DecryptResponse",
    # Signing/Verification
    "SignRequest",
    "SignResponse",
    "VerifyRequest",
    "VerifyResponse",
    # Key Generation
    "KeyPairRequest",
    "KeyPairResponse",
    # Status
    "CryptoStatusResponse",
]
