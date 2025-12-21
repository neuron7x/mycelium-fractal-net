"""
Tests for cryptography API adapters.

Validates encryption, decryption, signing, verification, and keypair generation
through the integration adapter layer.
"""

from __future__ import annotations

import base64

import pytest

from mycelium_fractal_net.integration.crypto_adapters import (
    CryptoAPIError,
    decrypt_data_adapter,
    encrypt_data_adapter,
    generate_keypair_adapter,
    sign_message_adapter,
    verify_signature_adapter,
)
from mycelium_fractal_net.integration.crypto_config import (
    get_key_store,
    reset_crypto_config,
    reset_key_store,
)
from mycelium_fractal_net.integration.schemas import (
    DecryptRequest,
    EncryptRequest,
    KeypairRequest,
    SignRequest,
    VerifyRequest,
)


@pytest.fixture(autouse=True)
def reset_crypto_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure crypto config and key store are reset for each test."""
    monkeypatch.setenv("MFN_CRYPTO_ENABLED", "true")
    monkeypatch.setenv("MFN_CRYPTO_AUDIT_LOG", "false")
    reset_crypto_config()
    reset_key_store()
    yield
    reset_crypto_config()
    reset_key_store()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def test_encrypt_decrypt_roundtrip_with_default_key() -> None:
    plaintext = b"secure payload"
    encrypt_request = EncryptRequest(plaintext=_b64(plaintext))

    encrypt_response = encrypt_data_adapter(encrypt_request)

    assert encrypt_response.key_id
    key_store = get_key_store()
    assert encrypt_response.key_id in key_store.encryption_keys

    decrypt_request = DecryptRequest(
        ciphertext=encrypt_response.ciphertext,
        key_id=encrypt_response.key_id,
    )
    decrypt_response = decrypt_data_adapter(decrypt_request)

    assert base64.b64decode(decrypt_response.plaintext) == plaintext


def test_encrypt_decrypt_roundtrip_with_aad() -> None:
    plaintext = b"context-bound"
    aad = b"session-123"
    encrypt_request = EncryptRequest(
        plaintext=_b64(plaintext),
        associated_data=_b64(aad),
    )

    encrypt_response = encrypt_data_adapter(encrypt_request)
    decrypt_request = DecryptRequest(
        ciphertext=encrypt_response.ciphertext,
        key_id=encrypt_response.key_id,
        associated_data=_b64(aad),
    )
    decrypt_response = decrypt_data_adapter(decrypt_request)

    assert base64.b64decode(decrypt_response.plaintext) == plaintext


def test_encrypt_rejects_invalid_plaintext_base64() -> None:
    request = EncryptRequest(plaintext="not-base64$$$")

    with pytest.raises(CryptoAPIError, match="Invalid base64-encoded plaintext"):
        encrypt_data_adapter(request)


def test_encrypt_rejects_invalid_aad_base64() -> None:
    request = EncryptRequest(
        plaintext=_b64(b"payload"),
        associated_data="invalid-aad!!",
    )

    with pytest.raises(CryptoAPIError, match="Invalid base64-encoded associated data"):
        encrypt_data_adapter(request)


def test_decrypt_rejects_invalid_ciphertext_base64() -> None:
    request = DecryptRequest(ciphertext="invalid-ciphertext!!")

    with pytest.raises(CryptoAPIError, match="Invalid base64-encoded ciphertext"):
        decrypt_data_adapter(request)


def test_decrypt_rejects_missing_key() -> None:
    plaintext = b"orphaned"
    encrypt_response = encrypt_data_adapter(EncryptRequest(plaintext=_b64(plaintext)))

    reset_key_store()
    request = DecryptRequest(ciphertext=encrypt_response.ciphertext)

    with pytest.raises(CryptoAPIError, match="No encryption key available"):
        decrypt_data_adapter(request)


def test_sign_and_verify_with_default_key() -> None:
    message = b"sign me"
    sign_response = sign_message_adapter(SignRequest(message=_b64(message)))

    verify_request = VerifyRequest(
        message=_b64(message),
        signature=sign_response.signature,
        key_id=sign_response.key_id,
    )
    verify_response = verify_signature_adapter(verify_request)

    assert verify_response.valid is True


def test_verify_rejects_missing_key_and_public_key() -> None:
    message = b"no key"
    signature = _b64(b"sig")

    request = VerifyRequest(message=_b64(message), signature=signature)

    with pytest.raises(CryptoAPIError, match="No public key provided"):
        verify_signature_adapter(request)


def test_verify_returns_false_for_tampered_signature() -> None:
    message = b"original"
    sign_response = sign_message_adapter(SignRequest(message=_b64(message)))

    tampered_signature = _b64(b"tampered")
    verify_request = VerifyRequest(
        message=_b64(message),
        signature=tampered_signature,
        key_id=sign_response.key_id,
    )
    verify_response = verify_signature_adapter(verify_request)

    assert verify_response.valid is False


def test_generate_keypair_for_signature_and_ecdh() -> None:
    signature_response = generate_keypair_adapter(KeypairRequest(algorithm="Ed25519"))
    ecdh_response = generate_keypair_adapter(KeypairRequest(algorithm="ECDH"))

    key_store = get_key_store()
    assert signature_response.key_id in key_store.signature_keys
    assert ecdh_response.key_id in key_store.ecdh_keys
    assert key_store.default_signature_key_id == signature_response.key_id


def test_crypto_disabled_blocks_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MFN_CRYPTO_ENABLED", "false")
    reset_crypto_config()

    with pytest.raises(CryptoAPIError, match="disabled"):
        encrypt_data_adapter(EncryptRequest(plaintext=_b64(b"blocked")))
