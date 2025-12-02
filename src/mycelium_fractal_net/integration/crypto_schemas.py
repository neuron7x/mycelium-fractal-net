"""
Pydantic schemas for Cryptographic API endpoints.

Provides request/response models for cryptographic operations including:
- Encryption/Decryption (AES-256-GCM)
- Digital signatures (Ed25519)
- Key pair generation (ECDH, Ed25519)

Reference: docs/MFN_CRYPTOGRAPHY.md
"""

from __future__ import annotations

import base64
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class KeyPairType(str, Enum):
    """Key pair type for generation."""

    ECDH = "ecdh"
    ED25519 = "ed25519"


# =============================================================================
# Encryption/Decryption
# =============================================================================


class EncryptRequest(BaseModel):
    """
    Request for encrypting data.

    Attributes:
        data: Base64-encoded plaintext data to encrypt.
        key_id: Optional key identifier. If not provided, uses server-managed key.
    """

    data: str = Field(
        ...,
        description="Base64-encoded plaintext data to encrypt",
        min_length=1,
        max_length=1_500_000,  # ~1MB base64 encoded
    )
    key_id: Optional[str] = Field(
        default=None,
        description="Key identifier. If not provided, uses server-managed key.",
    )

    @field_validator("data")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate that data is valid base64."""
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError("Invalid base64 encoding")
        return v


class EncryptResponse(BaseModel):
    """
    Response from encryption operation.

    Attributes:
        ciphertext: Base64-encoded encrypted data.
        key_id: Identifier of the key used for encryption.
        algorithm: Encryption algorithm used.
    """

    ciphertext: str = Field(description="Base64-encoded encrypted data")
    key_id: str = Field(description="Identifier of the key used")
    algorithm: str = Field(default="AES-256-GCM", description="Encryption algorithm")


class DecryptRequest(BaseModel):
    """
    Request for decrypting data.

    Attributes:
        ciphertext: Base64-encoded encrypted data to decrypt.
        key_id: Optional key identifier. If not provided, uses server-managed key.
    """

    ciphertext: str = Field(
        ...,
        description="Base64-encoded ciphertext to decrypt",
        min_length=1,
        max_length=1_500_000,
    )
    key_id: Optional[str] = Field(
        default=None,
        description="Key identifier. If not provided, uses server-managed key.",
    )

    @field_validator("ciphertext")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate that ciphertext is valid base64."""
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError("Invalid base64 encoding")
        return v


class DecryptResponse(BaseModel):
    """
    Response from decryption operation.

    Attributes:
        data: Base64-encoded decrypted plaintext.
        key_id: Identifier of the key used for decryption.
    """

    data: str = Field(description="Base64-encoded decrypted data")
    key_id: str = Field(description="Identifier of the key used")


# =============================================================================
# Digital Signatures
# =============================================================================


class SignRequest(BaseModel):
    """
    Request for signing a message.

    Attributes:
        message: Base64-encoded message to sign.
        key_id: Optional key identifier. If not provided, uses server-managed key.
    """

    message: str = Field(
        ...,
        description="Base64-encoded message to sign",
        min_length=1,
        max_length=1_500_000,
    )
    key_id: Optional[str] = Field(
        default=None,
        description="Key identifier for signing. If not provided, uses server key.",
    )

    @field_validator("message")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate that message is valid base64."""
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError("Invalid base64 encoding")
        return v


class SignResponse(BaseModel):
    """
    Response from signing operation.

    Attributes:
        signature: Base64-encoded digital signature.
        key_id: Identifier of the key used for signing.
        algorithm: Signature algorithm used.
    """

    signature: str = Field(description="Base64-encoded signature")
    key_id: str = Field(description="Identifier of the signing key")
    algorithm: str = Field(default="Ed25519", description="Signature algorithm")


class VerifyRequest(BaseModel):
    """
    Request for verifying a signature.

    Attributes:
        message: Base64-encoded original message.
        signature: Base64-encoded signature to verify.
        public_key: Optional Base64-encoded public key. If not provided, uses server key.
        key_id: Optional key identifier to use for verification.
    """

    message: str = Field(
        ...,
        description="Base64-encoded original message",
        min_length=1,
        max_length=1_500_000,
    )
    signature: str = Field(
        ...,
        description="Base64-encoded signature to verify",
        min_length=1,
    )
    public_key: Optional[str] = Field(
        default=None,
        description="Base64-encoded public key for verification",
    )
    key_id: Optional[str] = Field(
        default=None,
        description="Key identifier for verification. Used if public_key not provided.",
    )

    @field_validator("message", "signature")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate that value is valid base64."""
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError("Invalid base64 encoding")
        return v

    @field_validator("public_key")
    @classmethod
    def validate_public_key(cls, v: Optional[str]) -> Optional[str]:
        """Validate public key is valid base64 if provided."""
        if v is not None:
            try:
                decoded = base64.b64decode(v)
                if len(decoded) != 32:
                    raise ValueError("Ed25519 public key must be 32 bytes")
            except Exception as e:
                raise ValueError(f"Invalid public key: {e}")
        return v


class VerifyResponse(BaseModel):
    """
    Response from signature verification.

    Attributes:
        valid: True if signature is valid, False otherwise.
        key_id: Identifier of the key used for verification (if server key).
    """

    valid: bool = Field(description="True if signature is valid")
    key_id: Optional[str] = Field(
        default=None,
        description="Identifier of the key used for verification",
    )


# =============================================================================
# Key Pair Generation
# =============================================================================


class KeyPairRequest(BaseModel):
    """
    Request for generating a new key pair.

    Attributes:
        key_type: Type of key pair to generate (ecdh or ed25519).
    """

    key_type: KeyPairType = Field(
        default=KeyPairType.ECDH,
        description="Type of key pair: 'ecdh' for key exchange, 'ed25519' for signatures",
    )


class KeyPairResponse(BaseModel):
    """
    Response from key pair generation.

    Attributes:
        public_key: Base64-encoded public key.
        key_id: Unique identifier for the generated key pair.
        key_type: Type of key pair generated.
        algorithm: Algorithm used (X25519 for ECDH, Ed25519 for signatures).
    """

    public_key: str = Field(description="Base64-encoded public key")
    key_id: str = Field(description="Unique identifier for the key pair")
    key_type: KeyPairType = Field(description="Type of key pair")
    algorithm: str = Field(description="Algorithm used")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "KeyPairType",
    # Encryption
    "EncryptRequest",
    "EncryptResponse",
    "DecryptRequest",
    "DecryptResponse",
    # Signatures
    "SignRequest",
    "SignResponse",
    "VerifyRequest",
    "VerifyResponse",
    # Key Pairs
    "KeyPairRequest",
    "KeyPairResponse",
]
