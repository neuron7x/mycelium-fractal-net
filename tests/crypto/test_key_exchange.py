"""
Tests for X25519 key exchange functionality.

Verifies ECDH key exchange and key serialization.
"""

from __future__ import annotations

import pytest

from mycelium_fractal_net.crypto.key_exchange import (
    KeyExchangeKeyPair,
    generate_key_exchange_keypair,
    perform_key_exchange,
    serialize_key_exchange_private_key,
    serialize_key_exchange_public_key,
)


class TestGenerateKeyExchangeKeyPair:
    """Tests for key exchange key pair generation."""

    def test_generate_keypair_returns_keypair(self) -> None:
        """Generated result should be KeyExchangeKeyPair instance."""
        keypair = generate_key_exchange_keypair()
        assert isinstance(keypair, KeyExchangeKeyPair)
        assert keypair.private_key is not None
        assert keypair.public_key is not None

    def test_generate_keypair_unique(self) -> None:
        """Each generated keypair should be unique."""
        keypair1 = generate_key_exchange_keypair()
        keypair2 = generate_key_exchange_keypair()

        # Public keys should be different
        pub1 = serialize_key_exchange_public_key(keypair1.public_key)
        pub2 = serialize_key_exchange_public_key(keypair2.public_key)
        assert pub1 != pub2


class TestPerformKeyExchange:
    """Tests for X25519 ECDH key exchange."""

    def test_key_exchange_produces_shared_secret(self) -> None:
        """Key exchange should produce the same shared secret for both parties."""
        alice = generate_key_exchange_keypair()
        bob = generate_key_exchange_keypair()

        # Each party performs exchange with other's public key
        shared_alice = perform_key_exchange(alice.private_key, bob.public_key)
        shared_bob = perform_key_exchange(bob.private_key, alice.public_key)

        # Both should derive the same shared secret
        assert shared_alice == shared_bob

    def test_shared_secret_length(self) -> None:
        """Shared secret should be 32 bytes."""
        alice = generate_key_exchange_keypair()
        bob = generate_key_exchange_keypair()

        shared = perform_key_exchange(alice.private_key, bob.public_key)

        assert len(shared) == 32

    def test_different_keypairs_different_secrets(self) -> None:
        """Different key pairs should produce different shared secrets."""
        alice = generate_key_exchange_keypair()
        bob1 = generate_key_exchange_keypair()
        bob2 = generate_key_exchange_keypair()

        shared1 = perform_key_exchange(alice.private_key, bob1.public_key)
        shared2 = perform_key_exchange(alice.private_key, bob2.public_key)

        assert shared1 != shared2

    def test_key_exchange_deterministic(self) -> None:
        """Same key pairs should produce same shared secret."""
        alice = generate_key_exchange_keypair()
        bob = generate_key_exchange_keypair()

        shared1 = perform_key_exchange(alice.private_key, bob.public_key)
        shared2 = perform_key_exchange(alice.private_key, bob.public_key)

        assert shared1 == shared2


class TestSerializeKeyExchangeKeys:
    """Tests for key exchange key serialization."""

    def test_serialize_public_key_raw(self) -> None:
        """Serialize public key to RAW format (32 bytes)."""
        keypair = generate_key_exchange_keypair()
        raw = serialize_key_exchange_public_key(keypair.public_key, format="RAW")

        assert len(raw) == 32

    def test_serialize_public_key_pem(self) -> None:
        """Serialize public key to PEM format."""
        keypair = generate_key_exchange_keypair()
        pem = serialize_key_exchange_public_key(keypair.public_key, format="PEM")

        assert pem.startswith(b"-----BEGIN PUBLIC KEY-----")
        assert b"-----END PUBLIC KEY-----" in pem

    def test_serialize_public_key_der(self) -> None:
        """Serialize public key to DER format."""
        keypair = generate_key_exchange_keypair()
        der = serialize_key_exchange_public_key(keypair.public_key, format="DER")

        # DER is binary format
        assert isinstance(der, bytes)
        assert len(der) > 0

    def test_serialize_private_key_raw(self) -> None:
        """Serialize private key to RAW format (32 bytes)."""
        keypair = generate_key_exchange_keypair()
        raw = serialize_key_exchange_private_key(keypair.private_key, format="RAW")

        assert len(raw) == 32

    def test_serialize_private_key_pem(self) -> None:
        """Serialize private key to PEM format."""
        keypair = generate_key_exchange_keypair()
        pem = serialize_key_exchange_private_key(keypair.private_key, format="PEM")

        assert pem.startswith(b"-----BEGIN PRIVATE KEY-----")
        assert b"-----END PRIVATE KEY-----" in pem

    def test_serialize_private_key_with_password(self) -> None:
        """Serialize private key with password encryption."""
        keypair = generate_key_exchange_keypair()
        password = b"secure_password_123"
        pem = serialize_key_exchange_private_key(
            keypair.private_key, password=password, format="PEM"
        )

        assert pem.startswith(b"-----BEGIN ENCRYPTED PRIVATE KEY-----")
        assert b"-----END ENCRYPTED PRIVATE KEY-----" in pem

    def test_serialize_invalid_format_fails(self) -> None:
        """Invalid format should raise ValueError."""
        keypair = generate_key_exchange_keypair()

        with pytest.raises(ValueError, match="Unsupported format"):
            serialize_key_exchange_public_key(keypair.public_key, format="INVALID")

        with pytest.raises(ValueError, match="Unsupported format"):
            serialize_key_exchange_private_key(keypair.private_key, format="INVALID")


class TestKeyExchangeIntegration:
    """Integration tests for key exchange with KDF."""

    def test_key_exchange_with_kdf(self) -> None:
        """Key exchange shared secret should work with KDF for symmetric key derivation."""
        from mycelium_fractal_net.crypto.kdf import derive_key, generate_salt

        # Perform key exchange
        alice = generate_key_exchange_keypair()
        bob = generate_key_exchange_keypair()
        shared_secret = perform_key_exchange(alice.private_key, bob.public_key)

        # Use shared secret to derive symmetric key
        salt = generate_salt()
        symmetric_key = derive_key(shared_secret, salt, info=b"encryption-key")

        assert len(symmetric_key) == 32

    def test_key_exchange_derive_multiple_keys(self) -> None:
        """Derive multiple independent keys from shared secret."""
        from mycelium_fractal_net.crypto.kdf import derive_multiple_keys, generate_salt

        # Perform key exchange
        alice = generate_key_exchange_keypair()
        bob = generate_key_exchange_keypair()
        shared_secret = perform_key_exchange(alice.private_key, bob.public_key)

        # Derive multiple keys for different purposes
        salt = generate_salt()
        keys = derive_multiple_keys(
            shared_secret,
            salt,
            [
                (b"client-to-server-encryption", 32),
                (b"server-to-client-encryption", 32),
                (b"client-mac", 32),
                (b"server-mac", 32),
            ],
        )

        assert len(keys) == 4
        # All keys should be different
        assert len(set(keys)) == 4
