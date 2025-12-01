"""
Cryptographic Module for MyceliumFractalNet.

Provides cryptographic primitives with formal mathematical proofs of security:
    - ECDH (Elliptic Curve Diffie-Hellman) key exchange
    - EdDSA (Ed25519) digital signatures
    - Secure key derivation functions

Mathematical Security Foundations:
    - Discrete Logarithm Problem (DLP) hardness on elliptic curves
    - Collision resistance of SHA-512
    - Computational indistinguishability under CDH assumption

Security Standards:
    - RFC 7748 (X25519 key exchange)
    - RFC 8032 (Ed25519 signatures)
    - NIST SP 800-56A (key derivation)

Usage:
    >>> from mycelium_fractal_net.crypto import (
    ...     ECDHKeyExchange,
    ...     EdDSASignature,
    ...     derive_symmetric_key,
    ... )

Reference: docs/MFN_CRYPTOGRAPHY.md
"""

from .key_exchange import (
    ECDHKeyExchange,
    ECDHKeyPair,
    derive_symmetric_key,
    generate_ecdh_keypair,
)
from .signatures import (
    EdDSASignature,
    SignatureKeyPair,
    generate_signature_keypair,
    sign_message,
    verify_signature,
)

__all__ = [
    # Key Exchange
    "ECDHKeyExchange",
    "ECDHKeyPair",
    "generate_ecdh_keypair",
    "derive_symmetric_key",
    # Digital Signatures
    "EdDSASignature",
    "SignatureKeyPair",
    "generate_signature_keypair",
    "sign_message",
    "verify_signature",
]
