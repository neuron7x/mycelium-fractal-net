"""
Tests for RSA asymmetric encryption functionality.

Verifies RSA key generation, encryption, decryption, and serialization.
"""

from __future__ import annotations

import pytest

from mycelium_fractal_net.crypto.asymmetric import (
    DecryptionError,
    RSAKeyPair,
    generate_rsa_keypair,
    rsa_decrypt,
    rsa_encrypt,
    serialize_private_key,
    serialize_public_key,
)


class TestGenerateRSAKeyPair:
    """Tests for RSA key pair generation."""

    def test_generate_keypair_default_size(self) -> None:
        """Generated keypair should have default 4096-bit key size."""
        keypair = generate_rsa_keypair()
        assert keypair.key_size == 4096

    def test_generate_keypair_custom_size(self) -> None:
        """Generated keypair should respect custom key size."""
        keypair = generate_rsa_keypair(key_size=2048)
        assert keypair.key_size == 2048

    def test_generate_keypair_minimum_size(self) -> None:
        """Key size below 2048 should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2048 bits"):
            generate_rsa_keypair(key_size=1024)

    def test_generate_keypair_unique(self) -> None:
        """Each generated keypair should be unique."""
        keypair1 = generate_rsa_keypair(key_size=2048)
        keypair2 = generate_rsa_keypair(key_size=2048)

        # Public keys should be different
        pub1 = serialize_public_key(keypair1.public_key)
        pub2 = serialize_public_key(keypair2.public_key)
        assert pub1 != pub2

    def test_generate_keypair_returns_keypair(self) -> None:
        """Generated result should be RSAKeyPair instance."""
        keypair = generate_rsa_keypair(key_size=2048)
        assert isinstance(keypair, RSAKeyPair)
        assert keypair.private_key is not None
        assert keypair.public_key is not None


class TestRSAEncryptDecrypt:
    """Tests for RSA encryption and decryption."""

    def test_encrypt_decrypt_bytes(self) -> None:
        """Encrypt and decrypt bytes successfully."""
        keypair = generate_rsa_keypair(key_size=2048)
        plaintext = b"secret data"

        ciphertext = rsa_encrypt(plaintext, keypair.public_key)
        decrypted = rsa_decrypt(ciphertext, keypair.private_key)

        assert decrypted == plaintext

    def test_encrypt_decrypt_string(self) -> None:
        """Encrypt and decrypt string successfully."""
        keypair = generate_rsa_keypair(key_size=2048)
        plaintext = "secret message"

        ciphertext = rsa_encrypt(plaintext, keypair.public_key)
        decrypted = rsa_decrypt(ciphertext, keypair.private_key, encoding="utf-8")

        assert decrypted == plaintext

    def test_encrypt_decrypt_unicode(self) -> None:
        """Encrypt and decrypt unicode string successfully."""
        keypair = generate_rsa_keypair(key_size=2048)
        plaintext = "Привіт, світ! 你好世界 🌍"

        ciphertext = rsa_encrypt(plaintext, keypair.public_key)
        decrypted = rsa_decrypt(ciphertext, keypair.private_key, encoding="utf-8")

        assert decrypted == plaintext

    def test_encrypt_decrypt_empty_string(self) -> None:
        """Encrypt and decrypt empty bytes."""
        keypair = generate_rsa_keypair(key_size=2048)
        plaintext = b""

        ciphertext = rsa_encrypt(plaintext, keypair.public_key)
        decrypted = rsa_decrypt(ciphertext, keypair.private_key)

        assert decrypted == plaintext

    def test_ciphertext_different_each_time(self) -> None:
        """Same plaintext should produce different ciphertext (OAEP uses random padding)."""
        keypair = generate_rsa_keypair(key_size=2048)
        plaintext = b"test data"

        ciphertext1 = rsa_encrypt(plaintext, keypair.public_key)
        ciphertext2 = rsa_encrypt(plaintext, keypair.public_key)

        # Ciphertexts should be different due to random padding
        assert ciphertext1 != ciphertext2

        # But both should decrypt to same plaintext
        assert rsa_decrypt(ciphertext1, keypair.private_key) == plaintext
        assert rsa_decrypt(ciphertext2, keypair.private_key) == plaintext

    def test_ciphertext_length(self) -> None:
        """Ciphertext length should match key size in bytes."""
        keypair = generate_rsa_keypair(key_size=2048)
        plaintext = b"test"

        ciphertext = rsa_encrypt(plaintext, keypair.public_key)

        # RSA-2048 produces 256-byte ciphertext
        assert len(ciphertext) == 256

    def test_decrypt_wrong_key_fails(self) -> None:
        """Decryption with wrong key should fail."""
        keypair1 = generate_rsa_keypair(key_size=2048)
        keypair2 = generate_rsa_keypair(key_size=2048)
        plaintext = b"secret"

        ciphertext = rsa_encrypt(plaintext, keypair1.public_key)

        with pytest.raises(DecryptionError):
            rsa_decrypt(ciphertext, keypair2.private_key)

    def test_decrypt_tampered_data_fails(self) -> None:
        """Decryption of tampered data should fail."""
        keypair = generate_rsa_keypair(key_size=2048)
        plaintext = b"secret"

        ciphertext = rsa_encrypt(plaintext, keypair.public_key)
        # Tamper with ciphertext
        tampered = bytes([ciphertext[0] ^ 0xFF]) + ciphertext[1:]

        with pytest.raises(DecryptionError):
            rsa_decrypt(tampered, keypair.private_key)

    def test_encrypt_too_large_message_fails(self) -> None:
        """Encrypting message larger than max size should fail."""
        keypair = generate_rsa_keypair(key_size=2048)
        # RSA-2048 with OAEP-SHA256 can encrypt max 190 bytes
        plaintext = b"x" * 200

        with pytest.raises(ValueError, match="Plaintext too large"):
            rsa_encrypt(plaintext, keypair.public_key)


class TestSerializeKeys:
    """Tests for key serialization."""

    def test_serialize_public_key_pem(self) -> None:
        """Serialize public key to PEM format."""
        keypair = generate_rsa_keypair(key_size=2048)
        pem = serialize_public_key(keypair.public_key, format="PEM")

        assert pem.startswith(b"-----BEGIN PUBLIC KEY-----")
        assert b"-----END PUBLIC KEY-----" in pem

    def test_serialize_public_key_der(self) -> None:
        """Serialize public key to DER format."""
        keypair = generate_rsa_keypair(key_size=2048)
        der = serialize_public_key(keypair.public_key, format="DER")

        # DER is binary format
        assert isinstance(der, bytes)
        assert len(der) > 0

    def test_serialize_private_key_pem(self) -> None:
        """Serialize private key to PEM format."""
        keypair = generate_rsa_keypair(key_size=2048)
        pem = serialize_private_key(keypair.private_key, format="PEM")

        assert pem.startswith(b"-----BEGIN PRIVATE KEY-----")
        assert b"-----END PRIVATE KEY-----" in pem

    def test_serialize_private_key_with_password(self) -> None:
        """Serialize private key with password encryption."""
        keypair = generate_rsa_keypair(key_size=2048)
        password = b"secure_password_123"
        pem = serialize_private_key(keypair.private_key, password=password, format="PEM")

        assert pem.startswith(b"-----BEGIN ENCRYPTED PRIVATE KEY-----")
        assert b"-----END ENCRYPTED PRIVATE KEY-----" in pem

    def test_serialize_invalid_format_fails(self) -> None:
        """Invalid format should raise ValueError."""
        keypair = generate_rsa_keypair(key_size=2048)

        with pytest.raises(ValueError, match="Unsupported format"):
            serialize_public_key(keypair.public_key, format="INVALID")

        with pytest.raises(ValueError, match="Unsupported format"):
            serialize_private_key(keypair.private_key, format="INVALID")
