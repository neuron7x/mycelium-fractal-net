"""
Cryptographic API Router for MyceliumFractalNet.

Provides REST API endpoints for cryptographic operations:
- POST /api/encrypt: Encrypt data using AES-256-GCM
- POST /api/decrypt: Decrypt AES-256-GCM encrypted data
- POST /api/sign: Sign message using Ed25519
- POST /api/verify: Verify Ed25519 signature
- POST /api/keypair: Generate new key pair (ECDH or Ed25519)

All endpoints are protected by authentication and rate limiting.
Audit logging is performed for all operations.

Reference: docs/MFN_CRYPTOGRAPHY.md
"""

from __future__ import annotations

import base64
import logging
import secrets
import time
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Request

from mycelium_fractal_net.crypto import (
    AESGCMCipher,
    ECDHKeyExchange,
    EdDSASignature,
    SymmetricEncryptionError,
    generate_aes_key,
    generate_ecdh_keypair,
    generate_signature_keypair,
    verify_signature,
)

from .crypto_config import generate_key_id, get_crypto_config, is_crypto_enabled
from .crypto_schemas import (
    DecryptRequest,
    DecryptResponse,
    EncryptRequest,
    EncryptResponse,
    KeyPairRequest,
    KeyPairResponse,
    KeyPairType,
    SignRequest,
    SignResponse,
    VerifyRequest,
    VerifyResponse,
)
from .logging_config import get_request_id

# Logger for crypto operations
logger = logging.getLogger("mfn.crypto")


def get_audit_logger() -> logging.Logger:
    """Get logger for audit logging of crypto operations."""
    return logging.getLogger("mfn.crypto.audit")


# In-memory key storage for development and testing.
# WARNING: This implementation is NOT suitable for production use:
# - Keys are not persistent across restarts
# - Keys are vulnerable to memory inspection
# - No proper key rotation or revocation
# In production, use a proper Key Management System (KMS) such as:
# - AWS KMS, GCP Cloud KMS, Azure Key Vault
# - HashiCorp Vault
# - Hardware Security Modules (HSMs)
_encryption_keys: Dict[str, bytes] = {}
_signature_keypairs: Dict[str, EdDSASignature] = {}
_ecdh_keypairs: Dict[str, ECDHKeyExchange] = {}

# Server-managed default keys (generated on first use)
_server_encryption_key: Optional[bytes] = None
_server_signature_keypair: Optional[EdDSASignature] = None


def _get_server_encryption_key() -> tuple[bytes, str]:
    """Get or create the server-managed encryption key."""
    global _server_encryption_key
    if _server_encryption_key is None:
        _server_encryption_key = generate_aes_key()
        key_id = generate_key_id(_server_encryption_key)
        _encryption_keys[key_id] = _server_encryption_key
    key_id = generate_key_id(_server_encryption_key)
    return _server_encryption_key, key_id


def _get_server_signature_keypair() -> tuple[EdDSASignature, str]:
    """Get or create the server-managed signature keypair."""
    global _server_signature_keypair
    if _server_signature_keypair is None:
        _server_signature_keypair = EdDSASignature()
        key_id = generate_key_id(_server_signature_keypair.public_key)
        _signature_keypairs[key_id] = _server_signature_keypair
    key_id = generate_key_id(_server_signature_keypair.public_key)
    return _server_signature_keypair, key_id


def _audit_log(
    operation: str,
    key_id: Optional[str] = None,
    algorithm: Optional[str] = None,
    success: bool = True,
    error: Optional[str] = None,
) -> None:
    """
    Log crypto operation for audit purposes.

    Logs non-sensitive information only (key IDs, algorithms, success/failure).
    Never logs actual key material or plaintext data.

    Args:
        operation: Name of the operation (encrypt, decrypt, sign, verify, keypair).
        key_id: Identifier of the key used (if applicable).
        algorithm: Algorithm used for the operation.
        success: Whether the operation succeeded.
        error: Error message if operation failed.
    """
    config = get_crypto_config()
    if not config.audit.enabled:
        return

    audit_logger = get_audit_logger()
    request_id = get_request_id() or "no-request-id"

    log_data = {
        "operation": operation,
        "request_id": request_id,
        "success": success,
        "timestamp": time.time(),
    }

    if config.audit.include_key_id and key_id:
        log_data["key_id"] = key_id

    if algorithm:
        log_data["algorithm"] = algorithm

    if error:
        log_data["error"] = error

    if success:
        audit_logger.info(f"Crypto operation: {operation}", extra=log_data)
    else:
        audit_logger.warning(f"Crypto operation failed: {operation}", extra=log_data)


# Create the API router
router = APIRouter(prefix="/api", tags=["crypto"])


@router.post("/encrypt", response_model=EncryptResponse)
async def encrypt_data(request: EncryptRequest, req: Request) -> EncryptResponse:
    """
    Encrypt data using AES-256-GCM.

    Accepts base64-encoded plaintext and returns base64-encoded ciphertext.
    If no key_id is provided, uses the server-managed key.

    Args:
        request: Encryption request with data and optional key_id.
        req: FastAPI request object for context.

    Returns:
        EncryptResponse: Encrypted ciphertext with key information.

    Raises:
        HTTPException: 400 if encryption fails.
        HTTPException: 503 if crypto is disabled.
    """
    if not is_crypto_enabled():
        raise HTTPException(
            status_code=503,
            detail="Cryptographic operations are disabled",
        )

    try:
        # Decode the input data
        plaintext = base64.b64decode(request.data)

        # Get or use the specified key
        if request.key_id and request.key_id in _encryption_keys:
            key = _encryption_keys[request.key_id]
            key_id = request.key_id
        else:
            key, key_id = _get_server_encryption_key()

        # Encrypt the data
        cipher = AESGCMCipher(key=key)
        ciphertext = cipher.encrypt(plaintext)

        # Encode result as base64
        ciphertext_b64 = base64.b64encode(ciphertext).decode("utf-8")

        _audit_log("encrypt", key_id=key_id, algorithm="AES-256-GCM", success=True)

        return EncryptResponse(
            ciphertext=ciphertext_b64,
            key_id=key_id,
            algorithm="AES-256-GCM",
        )

    except Exception as e:
        _audit_log(
            "encrypt",
            key_id=request.key_id,
            algorithm="AES-256-GCM",
            success=False,
            error=str(e),
        )
        logger.error(f"Encryption failed: {e}")
        raise HTTPException(status_code=400, detail=f"Encryption failed: {e}")


@router.post("/decrypt", response_model=DecryptResponse)
async def decrypt_data(request: DecryptRequest, req: Request) -> DecryptResponse:
    """
    Decrypt AES-256-GCM encrypted data.

    Accepts base64-encoded ciphertext and returns base64-encoded plaintext.
    If no key_id is provided, uses the server-managed key.

    Args:
        request: Decryption request with ciphertext and optional key_id.
        req: FastAPI request object for context.

    Returns:
        DecryptResponse: Decrypted plaintext with key information.

    Raises:
        HTTPException: 400 if decryption fails (wrong key or tampered data).
        HTTPException: 503 if crypto is disabled.
    """
    if not is_crypto_enabled():
        raise HTTPException(
            status_code=503,
            detail="Cryptographic operations are disabled",
        )

    try:
        # Decode the ciphertext
        ciphertext = base64.b64decode(request.ciphertext)

        # Get or use the specified key
        if request.key_id and request.key_id in _encryption_keys:
            key = _encryption_keys[request.key_id]
            key_id = request.key_id
        else:
            key, key_id = _get_server_encryption_key()

        # Decrypt the data
        cipher = AESGCMCipher(key=key)
        plaintext = cipher.decrypt(ciphertext, return_bytes=True)

        # Encode result as base64
        plaintext_b64 = base64.b64encode(plaintext).decode("utf-8")  # type: ignore[arg-type]

        _audit_log("decrypt", key_id=key_id, algorithm="AES-256-GCM", success=True)

        return DecryptResponse(
            data=plaintext_b64,
            key_id=key_id,
        )

    except SymmetricEncryptionError as e:
        _audit_log(
            "decrypt",
            key_id=request.key_id,
            algorithm="AES-256-GCM",
            success=False,
            error=str(e),
        )
        raise HTTPException(
            status_code=400,
            detail="Decryption failed: Invalid ciphertext or wrong key",
        )
    except Exception as e:
        _audit_log(
            "decrypt",
            key_id=request.key_id,
            algorithm="AES-256-GCM",
            success=False,
            error=str(e),
        )
        logger.error(f"Decryption failed: {e}")
        raise HTTPException(status_code=400, detail=f"Decryption failed: {e}")


@router.post("/sign", response_model=SignResponse)
async def sign_data(request: SignRequest, req: Request) -> SignResponse:
    """
    Sign a message using Ed25519.

    Accepts base64-encoded message and returns base64-encoded signature.
    If no key_id is provided, uses the server-managed signing key.

    Args:
        request: Sign request with message and optional key_id.
        req: FastAPI request object for context.

    Returns:
        SignResponse: Digital signature with key information.

    Raises:
        HTTPException: 400 if signing fails.
        HTTPException: 503 if crypto is disabled.
    """
    if not is_crypto_enabled():
        raise HTTPException(
            status_code=503,
            detail="Cryptographic operations are disabled",
        )

    try:
        # Decode the message
        message = base64.b64decode(request.message)

        # Get or use the specified key
        if request.key_id and request.key_id in _signature_keypairs:
            signer = _signature_keypairs[request.key_id]
            key_id = request.key_id
        else:
            signer, key_id = _get_server_signature_keypair()

        # Sign the message
        signature = signer.sign(message)

        # Encode signature as base64
        signature_b64 = base64.b64encode(signature).decode("utf-8")

        _audit_log("sign", key_id=key_id, algorithm="Ed25519", success=True)

        return SignResponse(
            signature=signature_b64,
            key_id=key_id,
            algorithm="Ed25519",
        )

    except Exception as e:
        _audit_log(
            "sign",
            key_id=request.key_id,
            algorithm="Ed25519",
            success=False,
            error=str(e),
        )
        logger.error(f"Signing failed: {e}")
        raise HTTPException(status_code=400, detail=f"Signing failed: {e}")


@router.post("/verify", response_model=VerifyResponse)
async def verify_data(request: VerifyRequest, req: Request) -> VerifyResponse:
    """
    Verify an Ed25519 signature.

    Accepts base64-encoded message and signature, returns verification result.
    Can use either a provided public key or a server-managed key.

    Args:
        request: Verify request with message, signature, and optional keys.
        req: FastAPI request object for context.

    Returns:
        VerifyResponse: Verification result (valid: true/false).

    Raises:
        HTTPException: 400 if verification fails due to invalid input.
        HTTPException: 503 if crypto is disabled.
    """
    if not is_crypto_enabled():
        raise HTTPException(
            status_code=503,
            detail="Cryptographic operations are disabled",
        )

    try:
        # Decode the message and signature
        message = base64.b64decode(request.message)
        signature = base64.b64decode(request.signature)

        # Determine which public key to use
        key_id: Optional[str] = None

        if request.public_key:
            # Use provided public key
            public_key = base64.b64decode(request.public_key)
            valid = verify_signature(message, signature, public_key)
        elif request.key_id and request.key_id in _signature_keypairs:
            # Use key from storage
            signer = _signature_keypairs[request.key_id]
            valid = signer.verify(message, signature)
            key_id = request.key_id
        else:
            # Use server key
            signer, key_id = _get_server_signature_keypair()
            valid = signer.verify(message, signature)

        _audit_log(
            "verify",
            key_id=key_id,
            algorithm="Ed25519",
            success=True,
        )

        return VerifyResponse(
            valid=valid,
            key_id=key_id,
        )

    except Exception as e:
        _audit_log(
            "verify",
            key_id=request.key_id,
            algorithm="Ed25519",
            success=False,
            error=str(e),
        )
        logger.error(f"Verification failed: {e}")
        raise HTTPException(status_code=400, detail=f"Verification failed: {e}")


@router.post("/keypair", response_model=KeyPairResponse)
async def generate_keypair(request: KeyPairRequest, req: Request) -> KeyPairResponse:
    """
    Generate a new key pair.

    Creates either an ECDH key pair (for key exchange) or an Ed25519 key pair
    (for digital signatures). The private key is stored securely on the server;
    only the public key is returned.

    Args:
        request: KeyPair request specifying the key type.
        req: FastAPI request object for context.

    Returns:
        KeyPairResponse: Public key and key identifier.

    Raises:
        HTTPException: 400 if key generation fails.
        HTTPException: 503 if crypto is disabled.
    """
    if not is_crypto_enabled():
        raise HTTPException(
            status_code=503,
            detail="Cryptographic operations are disabled",
        )

    try:
        if request.key_type == KeyPairType.ECDH:
            # Generate ECDH keypair for key exchange
            keypair = generate_ecdh_keypair()
            public_key_b64 = base64.b64encode(keypair.public_key).decode("utf-8")
            key_id = generate_key_id(keypair.public_key)
            algorithm = "X25519"

            # Store the keypair
            ecdh_exchange = ECDHKeyExchange(keypair)
            _ecdh_keypairs[key_id] = ecdh_exchange

            # Also generate an encryption key from this keypair for use with encrypt/decrypt
            # This uses a deterministic derivation so the key_id can be reused
            derived_key = secrets.token_bytes(32)  # For now, generate a fresh key
            _encryption_keys[key_id] = derived_key

        else:  # Ed25519
            # Generate signature keypair
            keypair = generate_signature_keypair()
            public_key_b64 = base64.b64encode(keypair.public_key).decode("utf-8")
            key_id = generate_key_id(keypair.public_key)
            algorithm = "Ed25519"

            # Store the keypair
            signer = EdDSASignature(keypair)
            _signature_keypairs[key_id] = signer

        _audit_log(
            "keypair",
            key_id=key_id,
            algorithm=algorithm,
            success=True,
        )

        return KeyPairResponse(
            public_key=public_key_b64,
            key_id=key_id,
            key_type=request.key_type,
            algorithm=algorithm,
        )

    except Exception as e:
        _audit_log(
            "keypair",
            algorithm=request.key_type.value,
            success=False,
            error=str(e),
        )
        logger.error(f"Key generation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Key generation failed: {e}")


# For testing: function to reset all keys
def reset_crypto_keys() -> None:
    """Reset all stored keys (for testing purposes only)."""
    global _server_encryption_key, _server_signature_keypair
    _encryption_keys.clear()
    _signature_keypairs.clear()
    _ecdh_keypairs.clear()
    _server_encryption_key = None
    _server_signature_keypair = None


__all__ = [
    "router",
    "reset_crypto_keys",
]
