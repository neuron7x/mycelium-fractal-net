# Cryptography Security Documentation

**Version:** v4.1-crypto  
**Last Updated:** 2025-12-01  
**Status:** Production Ready

---

## Executive Summary

MyceliumFractalNet provides a comprehensive cryptography module implementing industry-standard algorithms with formal security guarantees. This document describes the cryptographic protocols, security models, threat analysis, and usage guidelines.

---

## 1. Cryptographic Primitives

### 1.1 Asymmetric Encryption (RSA-4096)

| Property | Value |
|----------|-------|
| **Algorithm** | RSA with OAEP Padding |
| **Key Size** | 4096 bits (default), minimum 2048 bits |
| **Padding** | PKCS#1 v2.1 OAEP |
| **Hash Function** | SHA-256 |
| **MGF** | MGF1-SHA256 |
| **Security Level** | ~128-bit equivalent |

**Security Definition: IND-CCA2**

The RSA-OAEP encryption scheme achieves IND-CCA2 (Indistinguishability under Adaptive Chosen Ciphertext Attack) security in the Random Oracle Model.

**Formal Definition:**  
For any probabilistic polynomial-time adversary A with access to decryption oracle, the advantage in distinguishing encryptions of two chosen plaintexts is negligible:

```
Adv^{IND-CCA2}_{RSA-OAEP}(A) = |Pr[Exp^{IND-CCA2-1} = 1] - Pr[Exp^{IND-CCA2-0} = 1]| ≤ negl(λ)
```

**Theorem (Bellare-Rogaway, 1994):**  
RSA-OAEP is IND-CCA2 secure in the Random Oracle Model assuming the RSA problem is hard.

**Reference:** [1] M. Bellare and P. Rogaway, "Optimal Asymmetric Encryption," EUROCRYPT 1994.

---

### 1.2 Digital Signatures (ECDSA)

| Property | Value |
|----------|-------|
| **Algorithm** | ECDSA |
| **Curve** | NIST P-384 (secp384r1) |
| **Hash Function** | SHA-384 |
| **Security Level** | ~192-bit equivalent |

**Security Definition: EUF-CMA**

The ECDSA signature scheme achieves EUF-CMA (Existential Unforgeability under Chosen Message Attack) security.

**Formal Definition:**  
For any probabilistic polynomial-time adversary A with access to a signing oracle, the probability of producing a valid signature on a message not queried to the oracle is negligible:

```
Adv^{EUF-CMA}_{ECDSA}(A) = Pr[Verify(pk, m*, σ*) = 1 ∧ m* ∉ Q] ≤ negl(λ)
```

where Q is the set of messages queried to the signing oracle.

**Theorem (Brown, 2005):**  
ECDSA is EUF-CMA secure in the Generic Group Model assuming the DLP (Discrete Logarithm Problem) is hard on the chosen elliptic curve.

**Reference:** [2] D. Brown, "Generic Groups, Collision Resistance, and ECDSA," Designs, Codes and Cryptography, 2005.

---

### 1.3 Key Derivation (HKDF)

| Property | Value |
|----------|-------|
| **Algorithm** | HKDF (RFC 5869) |
| **Hash Function** | SHA-256 |
| **Output Length** | Configurable (16-8160 bytes) |

**Security Definition: PRF Security**

HKDF provides a pseudorandom function (PRF) that extracts and expands key material.

**Formal Definition:**  
For any probabilistic polynomial-time adversary A, the output of HKDF is computationally indistinguishable from uniform random:

```
|Pr[A^{HKDF(IKM,salt,info)} = 1] - Pr[A^{$} = 1]| ≤ negl(λ)
```

where $ denotes uniform random sampling.

**Theorem (Krawczyk, 2010):**  
HKDF is a secure key derivation function when instantiated with HMAC-SHA256, assuming HMAC is a secure PRF.

**Reference:** [3] H. Krawczyk and P. Eronen, "HMAC-based Extract-and-Expand Key Derivation Function (HKDF)," RFC 5869, 2010.

---

### 1.4 Key Exchange (X25519)

| Property | Value |
|----------|-------|
| **Algorithm** | X25519 (Curve25519) |
| **Key Size** | 256 bits |
| **Security Level** | ~128-bit equivalent |

**Security Definition: CDH Security**

X25519 provides computational Diffie-Hellman (CDH) security on Curve25519.

**Formal Definition:**  
For any probabilistic polynomial-time adversary A, given (G, aG, bG) for random a, b, computing abG is hard:

```
Adv^{CDH}_{X25519}(A) = Pr[A(G, aG, bG) = abG] ≤ negl(λ)
```

**Additional Properties:**
- **Twist-safety:** Curve25519 is twist-secure, preventing invalid curve attacks
- **Constant-time:** Implementation uses constant-time algorithms preventing timing attacks
- **Forward Secrecy:** When used with ephemeral keys, provides forward secrecy

**Reference:** [4] D. J. Bernstein, "Curve25519: New Diffie-Hellman Speed Records," PKC 2006.

---

## 2. Threat Model

### 2.1 Adversary Capabilities

| Capability | Assumed |
|------------|---------|
| Passive eavesdropping | ✅ |
| Active network manipulation | ✅ |
| Chosen ciphertext attacks | ✅ |
| Chosen message attacks | ✅ |
| Timing side-channels | ✅ |
| Quantum computation | ❌ |

### 2.2 Security Assumptions

1. **Key Secrecy:** Private keys are stored securely and never transmitted
2. **Random Number Quality:** os.urandom() provides cryptographically secure random bytes
3. **Algorithm Integrity:** The cryptography library correctly implements algorithms
4. **No Side Channels:** The execution environment doesn't leak secret data through side channels

### 2.3 Out of Scope Threats

- Hardware-based attacks (EM analysis, power analysis)
- Quantum computing attacks
- Malicious execution environment
- Social engineering

---

## 3. Security Properties

### 3.1 Confidentiality (RSA-OAEP)

| Property | Guarantee |
|----------|-----------|
| Semantic security | ✅ IND-CPA |
| Chosen ciphertext security | ✅ IND-CCA2 |
| Non-malleability | ✅ NM-CCA2 |

### 3.2 Authenticity (ECDSA)

| Property | Guarantee |
|----------|-----------|
| Existential unforgeability | ✅ EUF-CMA |
| Strong unforgeability | ✅ SUF-CMA |
| Message authentication | ✅ |

### 3.3 Key Agreement (X25519)

| Property | Guarantee |
|----------|-----------|
| Key indistinguishability | ✅ |
| Forward secrecy | ✅ (with ephemeral keys) |
| Resistance to small subgroup | ✅ |

---

## 4. Implementation Security

### 4.1 Secure Defaults

```python
# RSA: 4096-bit keys by default
keypair = generate_rsa_keypair()  # Uses RSA-4096

# ECDSA: P-384 curve by default  
keypair = generate_ecdsa_keypair()  # Uses secp384r1

# KDF: SHA-256 with 32-byte output
key = derive_key(ikm, salt)  # Uses HKDF-SHA256
```

### 4.2 Constant-Time Operations

All cryptographic comparisons use constant-time algorithms to prevent timing attacks:

- Key comparison
- HMAC verification
- Signature verification

### 4.3 Memory Safety

- Keys are not logged
- Sensitive data is not included in error messages
- Serialized keys use standard formats (PEM, DER)

---

## 5. API Usage Guidelines

### 5.1 RSA Encryption

```python
from mycelium_fractal_net.crypto import (
    generate_rsa_keypair,
    rsa_encrypt,
    rsa_decrypt,
)

# Generate keypair
keypair = generate_rsa_keypair()

# Encrypt (max 446 bytes for RSA-4096)
ciphertext = rsa_encrypt(b"secret data", keypair.public_key)

# Decrypt
plaintext = rsa_decrypt(ciphertext, keypair.private_key)
```

**Note:** For data larger than 446 bytes, use hybrid encryption:
1. Generate random symmetric key
2. Encrypt data with symmetric key (AES-GCM)
3. Encrypt symmetric key with RSA

### 5.2 Digital Signatures

```python
from mycelium_fractal_net.crypto import (
    generate_ecdsa_keypair,
    sign_message,
    verify_signature,
)

# Generate keypair
keypair = generate_ecdsa_keypair()

# Sign message (any size)
signature = sign_message(b"message to sign", keypair.private_key)

# Verify signature
is_valid = verify_signature(b"message to sign", signature, keypair.public_key)
```

### 5.3 Key Exchange with Derived Keys

```python
from mycelium_fractal_net.crypto import (
    generate_key_exchange_keypair,
    perform_key_exchange,
    derive_key,
    generate_salt,
)

# Each party generates keypair
alice = generate_key_exchange_keypair()
bob = generate_key_exchange_keypair()

# Exchange public keys (over authenticated channel)
# ...

# Perform key exchange
shared_secret = perform_key_exchange(alice.private_key, bob.public_key)

# Derive symmetric keys
salt = generate_salt()
encryption_key = derive_key(shared_secret, salt, info=b"encryption")
mac_key = derive_key(shared_secret, salt, info=b"mac")
```

---

## 6. Key Management

### 6.1 Key Generation

- Use `generate_*_keypair()` functions for all key generation
- Keys are generated using cryptographically secure random bytes
- Minimum key sizes are enforced (RSA: 2048 bits)

### 6.2 Key Storage

**DO:**
- Store private keys in secrets managers (HashiCorp Vault, AWS Secrets Manager)
- Encrypt private keys at rest with strong passwords
- Use Hardware Security Modules (HSM) for high-security applications

**DON'T:**
- Store private keys in source control
- Log private keys or key material
- Transmit private keys over network

### 6.3 Key Rotation

Recommended rotation periods:

| Key Type | Rotation Period |
|----------|-----------------|
| RSA encryption keys | 1-2 years |
| ECDSA signing keys | 1-2 years |
| Key exchange keys | Per session (ephemeral) |
| Derived symmetric keys | Per session |

---

## 7. Security Checklist

### 7.1 Development

- [ ] Use default key sizes (don't reduce for "performance")
- [ ] Never log cryptographic keys or secrets
- [ ] Use secure random for all nonces and salts
- [ ] Validate all input before cryptographic operations
- [ ] Use constant-time comparison for secrets

### 7.2 Deployment

- [ ] Store private keys in secrets manager
- [ ] Enable audit logging for cryptographic operations
- [ ] Configure key rotation policies
- [ ] Use TLS 1.2+ for all network communication
- [ ] Review and update dependencies regularly

### 7.3 Monitoring

- [ ] Monitor for unusual cryptographic operation patterns
- [ ] Alert on signature verification failures
- [ ] Log key generation events
- [ ] Track key usage metrics

---

## 8. Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| cryptography | ≥41.0.0 | Core cryptographic primitives |
| Python | ≥3.10 | Runtime |

The `cryptography` library is a well-maintained, audited library that provides:
- OpenSSL bindings for optimized implementations
- Constant-time implementations where needed
- Regular security updates

---

## 9. Compliance

### 9.1 Standards Compliance

| Standard | Status |
|----------|--------|
| NIST SP 800-56A (Key Agreement) | ✅ Compliant |
| NIST SP 800-56B (Key Transport) | ✅ Compliant |
| NIST SP 800-56C (Key Derivation) | ✅ Compliant |
| FIPS 186-4 (Digital Signatures) | ✅ Compliant |
| RFC 5869 (HKDF) | ✅ Compliant |
| RFC 8017 (RSA) | ✅ Compliant |

### 9.2 Algorithm Suite

For applications requiring FIPS 140-2 compliance:

| Function | FIPS-Approved Algorithm |
|----------|------------------------|
| Asymmetric Encryption | RSA-OAEP with SHA-256 |
| Digital Signatures | ECDSA with P-384 |
| Key Derivation | HKDF-SHA256 |
| Hash Function | SHA-256, SHA-384 |

---

## 10. Version History

| Version | Date | Changes |
|---------|------|---------|
| 4.1-crypto | 2025-12-01 | Initial crypto module with RSA, ECDSA, HKDF, X25519 |

---

## References

[1] M. Bellare and P. Rogaway, "Optimal Asymmetric Encryption – How to Encrypt with RSA," EUROCRYPT 1994.

[2] D. Brown, "Generic Groups, Collision Resistance, and ECDSA," Designs, Codes and Cryptography, 2005.

[3] H. Krawczyk and P. Eronen, "HMAC-based Extract-and-Expand Key Derivation Function (HKDF)," RFC 5869, 2010.

[4] D. J. Bernstein, "Curve25519: New Diffie-Hellman Speed Records," PKC 2006.

[5] NIST, "Digital Signature Standard (DSS)," FIPS 186-4, 2013.

[6] NIST, "Recommendation for Key-Derivation Methods in Key-Establishment Schemes," SP 800-56C, 2018.

---

*For security concerns or vulnerabilities, please contact the security team or open a confidential issue.*
