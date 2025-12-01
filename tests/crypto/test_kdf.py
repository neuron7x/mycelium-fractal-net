"""
Tests for key derivation functionality.

Verifies HKDF key derivation function.
"""

from __future__ import annotations

import pytest

from mycelium_fractal_net.crypto.kdf import (
    derive_key,
    derive_multiple_keys,
    generate_salt,
)


class TestGenerateSalt:
    """Tests for salt generation."""

    def test_generate_salt_default_length(self) -> None:
        """Generated salt should have default 16-byte length."""
        salt = generate_salt()
        assert len(salt) == 16

    def test_generate_salt_custom_length(self) -> None:
        """Generated salt should respect custom length."""
        salt = generate_salt(length=32)
        assert len(salt) == 32

    def test_generate_salt_minimum_length(self) -> None:
        """Salt length below 8 should raise ValueError."""
        with pytest.raises(ValueError, match="at least 8 bytes"):
            generate_salt(length=4)

    def test_generate_salt_unique(self) -> None:
        """Each generated salt should be unique."""
        salts = [generate_salt() for _ in range(10)]
        unique_salts = set(salts)
        assert len(unique_salts) == 10

    def test_generate_salt_bytes(self) -> None:
        """Generated salt should be bytes."""
        salt = generate_salt()
        assert isinstance(salt, bytes)


class TestDeriveKey:
    """Tests for key derivation."""

    def test_derive_key_default_length(self) -> None:
        """Derived key should have default 32-byte length."""
        salt = generate_salt()
        key = derive_key(b"password", salt)
        assert len(key) == 32

    def test_derive_key_custom_length(self) -> None:
        """Derived key should respect custom length."""
        salt = generate_salt()
        key = derive_key(b"password", salt, length=64)
        assert len(key) == 64

    def test_derive_key_from_string(self) -> None:
        """Derive key from string input."""
        salt = generate_salt()
        key = derive_key("password_string", salt)
        assert len(key) == 32

    def test_derive_key_deterministic(self) -> None:
        """Same inputs should produce same key."""
        salt = generate_salt()
        key1 = derive_key(b"password", salt)
        key2 = derive_key(b"password", salt)
        assert key1 == key2

    def test_derive_key_different_salt(self) -> None:
        """Different salts should produce different keys."""
        salt1 = generate_salt()
        salt2 = generate_salt()

        key1 = derive_key(b"password", salt1)
        key2 = derive_key(b"password", salt2)

        assert key1 != key2

    def test_derive_key_different_password(self) -> None:
        """Different passwords should produce different keys."""
        salt = generate_salt()

        key1 = derive_key(b"password1", salt)
        key2 = derive_key(b"password2", salt)

        assert key1 != key2

    def test_derive_key_with_info(self) -> None:
        """Derive key with info parameter for domain separation."""
        salt = generate_salt()

        key1 = derive_key(b"password", salt, info=b"encryption")
        key2 = derive_key(b"password", salt, info=b"signing")

        # Same password and salt, different info = different keys
        assert key1 != key2

    def test_derive_key_minimum_length(self) -> None:
        """Key length below 16 should raise ValueError."""
        salt = generate_salt()

        with pytest.raises(ValueError, match="at least 16 bytes"):
            derive_key(b"password", salt, length=8)

    def test_derive_key_maximum_length(self) -> None:
        """Key length above maximum should raise ValueError."""
        salt = generate_salt()

        with pytest.raises(ValueError, match="cannot exceed"):
            derive_key(b"password", salt, length=255 * 32 + 1)

    def test_derive_key_no_salt(self) -> None:
        """Derive key without salt (uses zero salt)."""
        key = derive_key(b"password", salt=None)
        assert len(key) == 32

    def test_derive_key_empty_input(self) -> None:
        """Derive key from empty input."""
        salt = generate_salt()
        key = derive_key(b"", salt)
        assert len(key) == 32


class TestDeriveMultipleKeys:
    """Tests for deriving multiple keys."""

    def test_derive_multiple_keys_basic(self) -> None:
        """Derive multiple keys from same material."""
        salt = generate_salt()
        keys = derive_multiple_keys(
            b"master_secret",
            salt,
            [(b"encryption", 32), (b"signing", 64), (b"mac", 32)],
        )

        assert len(keys) == 3
        assert len(keys[0]) == 32
        assert len(keys[1]) == 64
        assert len(keys[2]) == 32

    def test_derive_multiple_keys_independent(self) -> None:
        """Multiple derived keys should be cryptographically independent."""
        salt = generate_salt()
        keys = derive_multiple_keys(
            b"master_secret",
            salt,
            [(b"key1", 32), (b"key2", 32), (b"key3", 32)],
        )

        # All keys should be different
        assert len(set(keys)) == 3

    def test_derive_multiple_keys_deterministic(self) -> None:
        """Same inputs should produce same keys."""
        salt = generate_salt()
        specs = [(b"enc", 32), (b"sign", 32)]

        keys1 = derive_multiple_keys(b"secret", salt, specs)
        keys2 = derive_multiple_keys(b"secret", salt, specs)

        assert keys1 == keys2

    def test_derive_multiple_keys_from_string(self) -> None:
        """Derive multiple keys from string input."""
        salt = generate_salt()
        keys = derive_multiple_keys(
            "password_string",
            salt,
            [(b"key1", 32)],
        )

        assert len(keys) == 1
        assert len(keys[0]) == 32
