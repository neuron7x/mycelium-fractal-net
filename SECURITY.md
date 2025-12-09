# Security Policy

## Supported Versions

We actively support the latest major and minor versions of MyceliumFractalNet.

| Version | Supported          |
| ------- | ------------------ |
| 4.1.x   | :white_check_mark: |
| < 4.1   | :x:                |

## Reporting a Vulnerability

We take the security of MyceliumFractalNet seriously. If you discover a security vulnerability, please report it responsibly:

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please report security issues via:
- **Email**: example@example.com (replace with actual security contact)
- **GitHub Security Advisories**: Use the "Security" tab in the repository

### What to Include

Please provide:
1. A clear description of the vulnerability
2. Steps to reproduce the issue
3. Potential impact and affected versions
4. Any suggested fixes (optional)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Varies by severity (Critical: <7 days, High: <30 days, Medium: <90 days)

## Security Best Practices

### Production Deployment

1. **Environment Configuration**
   - Always set `MFN_ENV=prod` in production
   - Never use default/placeholder API keys
   - Configure `MFN_CORS_ORIGINS` explicitly (do not use `*`)

2. **Cryptography**
   - Use `mycelium_fractal_net.crypto.symmetric.AESGCMCipher` for encryption
   - DO NOT use `security.encryption` module in production (deprecated XOR-based encryption)

3. **Rate Limiting**
   - For multi-replica deployments, use distributed rate limiting (Redis)
   - Set `MFN_TRUST_PROXY_HEADERS=true` only when behind a trusted reverse proxy
   - Configure appropriate rate limits for your workload

4. **Container Security**
   - Docker images run as non-root user (uid 1000)
   - Enable Kubernetes securityContext in production
   - Use read-only root filesystems where possible

5. **API Key Management**
   - Store API keys in Kubernetes secrets or secure vaults
   - Rotate API keys regularly
   - Use strong, randomly generated keys (min 32 bytes)

### Known Limitations

1. **In-Memory Rate Limiting**: Does not synchronize across multiple replicas. See documentation for distributed alternatives.

2. **X-Forwarded-For Trust**: By default, the application does not trust proxy headers. Set `MFN_TRUST_PROXY_HEADERS=true` only when behind a trusted reverse proxy to prevent IP spoofing.

## Security Features

- **API Key Authentication**: Configurable per-endpoint authentication
- **Rate Limiting**: Token bucket algorithm with per-endpoint limits
- **AES-256-GCM Encryption**: Production-grade symmetric encryption
- **Ed25519 Signatures**: Digital signature support
- **PBKDF2 Key Derivation**: Secure key derivation with 100,000 iterations
- **Audit Logging**: Structured JSON logging with request IDs
- **Prometheus Metrics**: Security-relevant metrics and monitoring

## Security Updates

Security updates will be released as patch versions and announced via:
- GitHub Security Advisories
- Release notes
- CHANGELOG.md

## Compliance

MyceliumFractalNet provides tools for:
- Secure data encryption at rest (AES-256-GCM)
- Audit trail logging
- Access control via API keys
- Rate limiting for DDoS protection

However, compliance with specific regulations (GDPR, HIPAA, PCI-DSS, etc.) depends on your deployment configuration and operational practices.

## Dependencies

We regularly scan dependencies for vulnerabilities using:
- `pip-audit`
- `bandit`
- GitHub Dependabot

Critical dependency vulnerabilities are patched promptly.
