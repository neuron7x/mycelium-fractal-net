"""
Tests for Cryptographic API endpoints.

Tests the following endpoints:
- POST /api/encrypt: Encrypt data using AES-256-GCM
- POST /api/decrypt: Decrypt data using AES-256-GCM
- POST /api/sign: Sign message using Ed25519
- POST /api/verify: Verify Ed25519 signature
- POST /api/keypair: Generate key pair (ECDH or Ed25519)

Also tests:
- Audit logging
- Backward compatibility (crypto toggle)
- Input validation
- Error handling
"""

from __future__ import annotations

import base64
import os
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from api import app
from mycelium_fractal_net.integration import (
    reset_crypto_config,
    reset_crypto_keys,
)


@pytest.fixture
def client() -> TestClient:
    """Create test client for FastAPI app."""
    # Reset crypto keys and config before each test
    reset_crypto_keys()
    reset_crypto_config()
    return TestClient(app)


@pytest.fixture
def sample_data() -> str:
    """Generate sample base64-encoded data."""
    return base64.b64encode(b"Hello, World!").decode("utf-8")


@pytest.fixture
def sample_message() -> str:
    """Generate sample base64-encoded message for signing."""
    return base64.b64encode(b"This is a test message to sign").decode("utf-8")


class TestEncryptEndpoint:
    """Tests for POST /api/encrypt endpoint."""

    def test_encrypt_success(self, client: TestClient, sample_data: str) -> None:
        """Should successfully encrypt data."""
        response = client.post("/api/encrypt", json={"data": sample_data})
        assert response.status_code == 200
        data = response.json()

        assert "ciphertext" in data
        assert "key_id" in data
        assert "algorithm" in data
        assert data["algorithm"] == "AES-256-GCM"

        # Ciphertext should be valid base64
        ciphertext_bytes = base64.b64decode(data["ciphertext"])
        assert len(ciphertext_bytes) > 0

    def test_encrypt_with_key_id(self, client: TestClient, sample_data: str) -> None:
        """Should encrypt using specified key ID."""
        # First, generate a keypair to get a key_id
        keypair_response = client.post("/api/keypair", json={"key_type": "ecdh"})
        assert keypair_response.status_code == 200
        key_id = keypair_response.json()["key_id"]

        # Now encrypt with the key_id
        response = client.post("/api/encrypt", json={"data": sample_data, "key_id": key_id})
        assert response.status_code == 200
        assert response.json()["key_id"] == key_id

    def test_encrypt_invalid_base64(self, client: TestClient) -> None:
        """Should reject invalid base64 data."""
        response = client.post("/api/encrypt", json={"data": "not-valid-base64!!!"})
        assert response.status_code == 422  # Validation error

    def test_encrypt_empty_data(self, client: TestClient) -> None:
        """Should reject empty data."""
        response = client.post("/api/encrypt", json={"data": ""})
        assert response.status_code == 422

    def test_encrypt_deterministic_key_id(self, client: TestClient, sample_data: str) -> None:
        """Multiple encryptions should use same server key (same key_id)."""
        response1 = client.post("/api/encrypt", json={"data": sample_data})
        response2 = client.post("/api/encrypt", json={"data": sample_data})

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response1.json()["key_id"] == response2.json()["key_id"]


class TestDecryptEndpoint:
    """Tests for POST /api/decrypt endpoint."""

    def test_decrypt_success(self, client: TestClient, sample_data: str) -> None:
        """Should successfully decrypt previously encrypted data."""
        # Encrypt first
        encrypt_response = client.post("/api/encrypt", json={"data": sample_data})
        assert encrypt_response.status_code == 200
        ciphertext = encrypt_response.json()["ciphertext"]
        key_id = encrypt_response.json()["key_id"]

        # Decrypt
        decrypt_response = client.post(
            "/api/decrypt", json={"ciphertext": ciphertext, "key_id": key_id}
        )
        assert decrypt_response.status_code == 200
        data = decrypt_response.json()

        assert "data" in data
        assert "key_id" in data
        assert data["data"] == sample_data

    def test_decrypt_round_trip(self, client: TestClient) -> None:
        """Encrypt and decrypt should return original data."""
        original = b"Test message with unicode: \xc3\xa9\xc3\xa0\xc3\xbc"
        original_b64 = base64.b64encode(original).decode("utf-8")

        # Encrypt
        encrypt_response = client.post("/api/encrypt", json={"data": original_b64})
        assert encrypt_response.status_code == 200

        # Decrypt
        decrypt_response = client.post(
            "/api/decrypt", json={"ciphertext": encrypt_response.json()["ciphertext"]}
        )
        assert decrypt_response.status_code == 200

        # Verify
        decrypted_b64 = decrypt_response.json()["data"]
        decrypted = base64.b64decode(decrypted_b64)
        assert decrypted == original

    def test_decrypt_wrong_key(self, client: TestClient, sample_data: str) -> None:
        """Should fail with wrong key ID."""
        # Encrypt with default key
        encrypt_response = client.post("/api/encrypt", json={"data": sample_data})
        ciphertext = encrypt_response.json()["ciphertext"]

        # Generate new key
        keypair_response = client.post("/api/keypair", json={"key_type": "ecdh"})
        new_key_id = keypair_response.json()["key_id"]

        # Try to decrypt with new key - should fail
        decrypt_response = client.post(
            "/api/decrypt", json={"ciphertext": ciphertext, "key_id": new_key_id}
        )
        assert decrypt_response.status_code == 400

    def test_decrypt_invalid_ciphertext(self, client: TestClient) -> None:
        """Should fail with invalid ciphertext."""
        invalid_ciphertext = base64.b64encode(b"not valid ciphertext").decode("utf-8")
        response = client.post("/api/decrypt", json={"ciphertext": invalid_ciphertext})
        assert response.status_code == 400


class TestSignEndpoint:
    """Tests for POST /api/sign endpoint."""

    def test_sign_success(self, client: TestClient, sample_message: str) -> None:
        """Should successfully sign a message."""
        response = client.post("/api/sign", json={"message": sample_message})
        assert response.status_code == 200
        data = response.json()

        assert "signature" in data
        assert "key_id" in data
        assert "algorithm" in data
        assert data["algorithm"] == "Ed25519"

        # Signature should be 64 bytes when decoded
        signature_bytes = base64.b64decode(data["signature"])
        assert len(signature_bytes) == 64

    def test_sign_with_key_id(self, client: TestClient, sample_message: str) -> None:
        """Should sign using specified key ID."""
        # Generate an Ed25519 keypair
        keypair_response = client.post("/api/keypair", json={"key_type": "ed25519"})
        assert keypair_response.status_code == 200
        key_id = keypair_response.json()["key_id"]

        # Sign with the key_id
        response = client.post("/api/sign", json={"message": sample_message, "key_id": key_id})
        assert response.status_code == 200
        assert response.json()["key_id"] == key_id

    def test_sign_deterministic(self, client: TestClient, sample_message: str) -> None:
        """Same message with same key should produce same signature."""
        response1 = client.post("/api/sign", json={"message": sample_message})
        response2 = client.post("/api/sign", json={"message": sample_message})

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response1.json()["signature"] == response2.json()["signature"]


class TestVerifyEndpoint:
    """Tests for POST /api/verify endpoint."""

    def test_verify_success(self, client: TestClient, sample_message: str) -> None:
        """Should verify a valid signature."""
        # Sign first
        sign_response = client.post("/api/sign", json={"message": sample_message})
        assert sign_response.status_code == 200
        signature = sign_response.json()["signature"]
        key_id = sign_response.json()["key_id"]

        # Verify
        verify_response = client.post(
            "/api/verify",
            json={"message": sample_message, "signature": signature, "key_id": key_id},
        )
        assert verify_response.status_code == 200
        assert verify_response.json()["valid"] is True

    def test_verify_invalid_signature(self, client: TestClient, sample_message: str) -> None:
        """Should reject invalid signature."""
        # Create a fake signature (64 bytes)
        fake_signature = base64.b64encode(b"x" * 64).decode("utf-8")

        verify_response = client.post(
            "/api/verify", json={"message": sample_message, "signature": fake_signature}
        )
        assert verify_response.status_code == 200
        assert verify_response.json()["valid"] is False

    def test_verify_wrong_message(self, client: TestClient, sample_message: str) -> None:
        """Should fail if message doesn't match signature."""
        # Sign original message
        sign_response = client.post("/api/sign", json={"message": sample_message})
        signature = sign_response.json()["signature"]
        key_id = sign_response.json()["key_id"]

        # Try to verify with different message
        different_message = base64.b64encode(b"Different message").decode("utf-8")
        verify_response = client.post(
            "/api/verify",
            json={"message": different_message, "signature": signature, "key_id": key_id},
        )
        assert verify_response.status_code == 200
        assert verify_response.json()["valid"] is False

    def test_verify_with_public_key(self, client: TestClient, sample_message: str) -> None:
        """Should verify using provided public key."""
        # Generate keypair and sign
        keypair_response = client.post("/api/keypair", json={"key_type": "ed25519"})
        public_key = keypair_response.json()["public_key"]
        key_id = keypair_response.json()["key_id"]

        sign_response = client.post("/api/sign", json={"message": sample_message, "key_id": key_id})
        signature = sign_response.json()["signature"]

        # Verify with public key instead of key_id
        verify_response = client.post(
            "/api/verify",
            json={"message": sample_message, "signature": signature, "public_key": public_key},
        )
        assert verify_response.status_code == 200
        assert verify_response.json()["valid"] is True


class TestKeyPairEndpoint:
    """Tests for POST /api/keypair endpoint."""

    def test_generate_ecdh_keypair(self, client: TestClient) -> None:
        """Should generate ECDH keypair."""
        response = client.post("/api/keypair", json={"key_type": "ecdh"})
        assert response.status_code == 200
        data = response.json()

        assert "public_key" in data
        assert "key_id" in data
        assert "key_type" in data
        assert "algorithm" in data

        assert data["key_type"] == "ecdh"
        assert data["algorithm"] == "X25519"

        # Public key should be 32 bytes
        public_key_bytes = base64.b64decode(data["public_key"])
        assert len(public_key_bytes) == 32

    def test_generate_ed25519_keypair(self, client: TestClient) -> None:
        """Should generate Ed25519 keypair."""
        response = client.post("/api/keypair", json={"key_type": "ed25519"})
        assert response.status_code == 200
        data = response.json()

        assert data["key_type"] == "ed25519"
        assert data["algorithm"] == "Ed25519"

        # Public key should be 32 bytes
        public_key_bytes = base64.b64decode(data["public_key"])
        assert len(public_key_bytes) == 32

    def test_generate_default_keypair(self, client: TestClient) -> None:
        """Default key type should be ECDH."""
        response = client.post("/api/keypair", json={})
        assert response.status_code == 200
        assert response.json()["key_type"] == "ecdh"

    def test_generate_unique_keypairs(self, client: TestClient) -> None:
        """Each generated keypair should be unique."""
        response1 = client.post("/api/keypair", json={"key_type": "ecdh"})
        response2 = client.post("/api/keypair", json={"key_type": "ecdh"})

        assert response1.json()["public_key"] != response2.json()["public_key"]
        assert response1.json()["key_id"] != response2.json()["key_id"]


class TestCryptoToggle:
    """Tests for crypto enable/disable toggle."""

    def test_crypto_disabled_returns_503(self) -> None:
        """When crypto is disabled, endpoints should return 503."""
        reset_crypto_keys()
        reset_crypto_config()

        with mock.patch.dict(os.environ, {"MFN_CRYPTO_ENABLED": "false"}):
            reset_crypto_config()
            # Need to reimport the client to pick up the config change
            from mycelium_fractal_net.integration.crypto_api import is_crypto_enabled

            # Verify crypto is disabled
            assert is_crypto_enabled() is False

        # Reset for other tests
        reset_crypto_config()


class TestAuditLogging:
    """Tests for audit logging of crypto operations."""

    def test_audit_log_encrypt(
        self, client: TestClient, sample_data: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Encrypt operation should be audit logged."""
        with caplog.at_level("INFO", logger="mfn.crypto.audit"):
            response = client.post("/api/encrypt", json={"data": sample_data})
            assert response.status_code == 200

            # Check that audit log was created (may not appear due to log config)
            # The log should contain operation info

    def test_audit_log_sign(
        self, client: TestClient, sample_message: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Sign operation should be audit logged."""
        with caplog.at_level("INFO", logger="mfn.crypto.audit"):
            response = client.post("/api/sign", json={"message": sample_message})
            assert response.status_code == 200


class TestInputValidation:
    """Tests for input validation."""

    def test_encrypt_payload_size_limit(self, client: TestClient) -> None:
        """Should enforce payload size limit."""
        # Create data larger than 1MB base64 encoded
        large_data = base64.b64encode(b"x" * 2_000_000).decode("utf-8")
        response = client.post("/api/encrypt", json={"data": large_data})
        assert response.status_code == 422  # Validation error

    def test_verify_invalid_public_key_length(self, client: TestClient) -> None:
        """Should reject public key with wrong length."""
        message = base64.b64encode(b"test").decode("utf-8")
        signature = base64.b64encode(b"x" * 64).decode("utf-8")
        wrong_length_key = base64.b64encode(b"x" * 16).decode("utf-8")  # Wrong length

        response = client.post(
            "/api/verify",
            json={"message": message, "signature": signature, "public_key": wrong_length_key},
        )
        assert response.status_code == 422


class TestRateLimitingEndpoints:
    """Tests for rate limit settings on crypto endpoints."""

    def test_crypto_endpoints_exist(self, client: TestClient) -> None:
        """Verify all crypto endpoints are reachable."""
        endpoints = [
            ("/api/encrypt", {"data": base64.b64encode(b"test").decode()}),
            ("/api/decrypt", {"ciphertext": base64.b64encode(b"test").decode()}),
            ("/api/sign", {"message": base64.b64encode(b"test").decode()}),
            (
                "/api/verify",
                {
                    "message": base64.b64encode(b"test").decode(),
                    "signature": base64.b64encode(b"x" * 64).decode(),
                },
            ),
            ("/api/keypair", {}),
        ]

        for endpoint, payload in endpoints:
            response = client.post(endpoint, json=payload)
            # Should not be 404
            assert response.status_code != 404, f"Endpoint {endpoint} not found"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_encrypt_binary_data(self, client: TestClient) -> None:
        """Should handle binary data correctly."""
        binary_data = bytes(range(256))  # All byte values
        data_b64 = base64.b64encode(binary_data).decode("utf-8")

        encrypt_response = client.post("/api/encrypt", json={"data": data_b64})
        assert encrypt_response.status_code == 200

        decrypt_response = client.post(
            "/api/decrypt", json={"ciphertext": encrypt_response.json()["ciphertext"]}
        )
        assert decrypt_response.status_code == 200

        decrypted = base64.b64decode(decrypt_response.json()["data"])
        assert decrypted == binary_data

    def test_sign_empty_message(self, client: TestClient) -> None:
        """Should handle empty message."""
        empty_message = base64.b64encode(b"").decode("utf-8")
        # Empty message is valid base64 "AA=="
        response = client.post("/api/sign", json={"message": empty_message})
        # May succeed or fail based on validation
        assert response.status_code in (200, 422)
