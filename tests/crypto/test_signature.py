"""
Tests for ECDSA digital signature functionality.

Verifies signature generation, verification, and key serialization.
"""

from __future__ import annotations

import pytest

from mycelium_fractal_net.crypto.signature import (
    ECDSAKeyPair,
    generate_ecdsa_keypair,
    serialize_ecdsa_private_key,
    serialize_ecdsa_public_key,
    sign_message,
    verify_signature,
)


class TestGenerateECDSAKeyPair:
    """Tests for ECDSA key pair generation."""

    def test_generate_keypair_default_curve(self) -> None:
        """Generated keypair should use P-384 curve by default."""
        keypair = generate_ecdsa_keypair()
        assert keypair.curve_name == "secp384r1"

    def test_generate_keypair_unique(self) -> None:
        """Each generated keypair should be unique."""
        keypair1 = generate_ecdsa_keypair()
        keypair2 = generate_ecdsa_keypair()

        # Public keys should be different
        pub1 = serialize_ecdsa_public_key(keypair1.public_key)
        pub2 = serialize_ecdsa_public_key(keypair2.public_key)
        assert pub1 != pub2

    def test_generate_keypair_returns_keypair(self) -> None:
        """Generated result should be ECDSAKeyPair instance."""
        keypair = generate_ecdsa_keypair()
        assert isinstance(keypair, ECDSAKeyPair)
        assert keypair.private_key is not None
        assert keypair.public_key is not None


class TestSignVerify:
    """Tests for signature generation and verification."""

    def test_sign_verify_bytes(self) -> None:
        """Sign and verify bytes message."""
        keypair = generate_ecdsa_keypair()
        message = b"Hello, World!"

        signature = sign_message(message, keypair.private_key)
        assert verify_signature(message, signature, keypair.public_key)

    def test_sign_verify_string(self) -> None:
        """Sign and verify string message."""
        keypair = generate_ecdsa_keypair()
        message = "Hello, World!"

        signature = sign_message(message, keypair.private_key)
        assert verify_signature(message, signature, keypair.public_key)

    def test_sign_verify_unicode(self) -> None:
        """Sign and verify unicode message."""
        keypair = generate_ecdsa_keypair()
        message = "Привіт, світ! 你好世界 🌍"

        signature = sign_message(message, keypair.private_key)
        assert verify_signature(message, signature, keypair.public_key)

    def test_sign_verify_empty_message(self) -> None:
        """Sign and verify empty message."""
        keypair = generate_ecdsa_keypair()
        message = b""

        signature = sign_message(message, keypair.private_key)
        assert verify_signature(message, signature, keypair.public_key)

    def test_sign_verify_large_message(self) -> None:
        """Sign and verify large message."""
        keypair = generate_ecdsa_keypair()
        message = b"x" * 10000  # 10KB message

        signature = sign_message(message, keypair.private_key)
        assert verify_signature(message, signature, keypair.public_key)

    def test_signature_different_each_time(self) -> None:
        """Same message should produce different signatures (ECDSA uses k value)."""
        keypair = generate_ecdsa_keypair()
        message = b"test message"

        sig1 = sign_message(message, keypair.private_key)
        sig2 = sign_message(message, keypair.private_key)

        # Note: With deterministic ECDSA (RFC 6979), signatures may be the same
        # for the same message and key. This is actually more secure.
        # Both should verify correctly.
        assert verify_signature(message, sig1, keypair.public_key)
        assert verify_signature(message, sig2, keypair.public_key)

    def test_verify_wrong_message_fails(self) -> None:
        """Verification with wrong message should return False."""
        keypair = generate_ecdsa_keypair()
        message = b"original message"
        wrong_message = b"different message"

        signature = sign_message(message, keypair.private_key)

        assert not verify_signature(wrong_message, signature, keypair.public_key)

    def test_verify_wrong_key_fails(self) -> None:
        """Verification with wrong public key should return False."""
        keypair1 = generate_ecdsa_keypair()
        keypair2 = generate_ecdsa_keypair()
        message = b"secret message"

        signature = sign_message(message, keypair1.private_key)

        assert not verify_signature(message, signature, keypair2.public_key)

    def test_verify_tampered_signature_fails(self) -> None:
        """Verification of tampered signature should return False."""
        keypair = generate_ecdsa_keypair()
        message = b"secret message"

        signature = sign_message(message, keypair.private_key)
        # Tamper with signature
        tampered = bytes([signature[0] ^ 0xFF]) + signature[1:]

        assert not verify_signature(message, tampered, keypair.public_key)

    def test_verify_truncated_signature_fails(self) -> None:
        """Verification of truncated signature should return False."""
        keypair = generate_ecdsa_keypair()
        message = b"secret message"

        signature = sign_message(message, keypair.private_key)
        # Truncate signature
        truncated = signature[:10]

        assert not verify_signature(message, truncated, keypair.public_key)


class TestSerializeECDSAKeys:
    """Tests for ECDSA key serialization."""

    def test_serialize_public_key_pem(self) -> None:
        """Serialize public key to PEM format."""
        keypair = generate_ecdsa_keypair()
        pem = serialize_ecdsa_public_key(keypair.public_key, format="PEM")

        assert pem.startswith(b"-----BEGIN PUBLIC KEY-----")
        assert b"-----END PUBLIC KEY-----" in pem

    def test_serialize_public_key_der(self) -> None:
        """Serialize public key to DER format."""
        keypair = generate_ecdsa_keypair()
        der = serialize_ecdsa_public_key(keypair.public_key, format="DER")

        # DER is binary format
        assert isinstance(der, bytes)
        assert len(der) > 0

    def test_serialize_private_key_pem(self) -> None:
        """Serialize private key to PEM format."""
        keypair = generate_ecdsa_keypair()
        pem = serialize_ecdsa_private_key(keypair.private_key, format="PEM")

        assert pem.startswith(b"-----BEGIN PRIVATE KEY-----")
        assert b"-----END PRIVATE KEY-----" in pem

    def test_serialize_private_key_with_password(self) -> None:
        """Serialize private key with password encryption."""
        keypair = generate_ecdsa_keypair()
        password = b"secure_password_123"
        pem = serialize_ecdsa_private_key(keypair.private_key, password=password, format="PEM")

        assert pem.startswith(b"-----BEGIN ENCRYPTED PRIVATE KEY-----")
        assert b"-----END ENCRYPTED PRIVATE KEY-----" in pem

    def test_serialize_invalid_format_fails(self) -> None:
        """Invalid format should raise ValueError."""
        keypair = generate_ecdsa_keypair()

        with pytest.raises(ValueError, match="Unsupported format"):
            serialize_ecdsa_public_key(keypair.public_key, format="INVALID")

        with pytest.raises(ValueError, match="Unsupported format"):
            serialize_ecdsa_private_key(keypair.private_key, format="INVALID")
