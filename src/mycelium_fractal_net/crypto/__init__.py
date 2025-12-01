"""
Cryptography module for MyceliumFractalNet.

Provides production-quality cryptographic primitives with formal security guarantees:
    - RSA-4096 asymmetric encryption (IND-CCA2 secure)
    - ECDSA digital signatures (EUF-CMA secure)
    - HKDF-SHA256 key derivation
    - X25519 Elliptic Curve Diffie-Hellman key exchange

Security Standards:
    - Follows NIST SP 800-56A/B key derivation guidelines
    - PKCS#1 v2.1 (OAEP) padding for RSA encryption
    - Constant-time comparison for signature verification
    - Secure random number generation via os.urandom()

Usage:
    >>> from mycelium_fractal_net.crypto import (
    ...     RSAKeyPair,
    ...     generate_rsa_keypair,
    ...     rsa_encrypt,
    ...     rsa_decrypt,
    ...     sign_message,
    ...     verify_signature,
    ...     derive_key,
    ...     perform_key_exchange,
    ... )

Reference: docs/crypto_security.md
"""

from .asymmetric import (
    RSAKeyPair,
    generate_rsa_keypair,
    rsa_decrypt,
    rsa_encrypt,
    serialize_private_key,
    serialize_public_key,
)
from .kdf import (
    derive_key,
    generate_salt,
)
from .key_exchange import (
    KeyExchangeKeyPair,
    generate_key_exchange_keypair,
    perform_key_exchange,
    serialize_key_exchange_private_key,
    serialize_key_exchange_public_key,
)
from .signature import (
    ECDSAKeyPair,
    generate_ecdsa_keypair,
    serialize_ecdsa_private_key,
    serialize_ecdsa_public_key,
    sign_message,
    verify_signature,
)

__all__ = [
    # RSA Asymmetric Encryption
    "RSAKeyPair",
    "generate_rsa_keypair",
    "rsa_encrypt",
    "rsa_decrypt",
    "serialize_public_key",
    "serialize_private_key",
    # Digital Signatures
    "ECDSAKeyPair",
    "generate_ecdsa_keypair",
    "sign_message",
    "verify_signature",
    "serialize_ecdsa_public_key",
    "serialize_ecdsa_private_key",
    # Key Derivation
    "derive_key",
    "generate_salt",
    # Key Exchange
    "KeyExchangeKeyPair",
    "generate_key_exchange_keypair",
    "perform_key_exchange",
    "serialize_key_exchange_public_key",
    "serialize_key_exchange_private_key",
]
