# Changelog

All notable changes to MyceliumFractalNet will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security (P0 - Critical)
- **BREAKING**: Changed default `MFN_ENV` from `dev` to `prod` for production-safe defaults
- **BREAKING**: Deprecated `security.encryption` XOR-based encryption; raises `RuntimeError` in production
  - Use `crypto.symmetric.AESGCMCipher` for production encryption instead
- **BREAKING**: Removed placeholder API key from k8s.yaml Secret manifest
  - Deployment now requires explicit secret creation via `kubectl create secret`
- **BREAKING**: Disabled CORS wildcard (`*`) by default in k8s.yaml Ingress
  - Must explicitly configure `MFN_CORS_ORIGINS` or ingress annotations
- Added X-Forwarded-For spoofing protection
  - By default, only trusts direct client IP
  - Set `MFN_TRUST_PROXY_HEADERS=true` only when behind trusted reverse proxy
- Added production warning for in-memory rate limiting in multi-replica deployments
  - Logs warning unless `MFN_RATE_LIMIT_WARN_MULTI_REPLICA=false`

### Changed
- **BREAKING**: Docker container now runs API server by default (was: validation script)
  - To run validation: `docker run ... python mycelium_fractal_net_v4_1.py --mode validate`
- **BREAKING**: Docker now runs as non-root user (uid 1000 `mfnuser`)
- **BREAKING**: K8s deployment now explicitly runs API via `command` and `args`
- **BREAKING**: Fixed pyproject.toml package discovery to only include `mycelium_fractal_net*` from `src/`
  - Moved top-level `analytics` module into `mycelium_fractal_net.analytics` (eliminates namespace collision)
  - All imports updated: `from analytics import X` → `from mycelium_fractal_net.analytics import X`
  - Top-level `experiments` excluded from distribution (dev-only)
  - Package now correctly auto-discovers all subpackages
- Docker HEALTHCHECK now uses lightweight `/health` endpoint (was: expensive validation)
- K8s deployment now includes securityContext with non-root user and capability drop
- Increased K8s resource limits for expensive endpoints (validate/simulate): 2Gi memory, 2 CPU
- Synchronized requirements.txt with pyproject.toml dependencies

### Fixed
- pytest now correctly finds `src/` packages without sys.path hacks
- Package build now includes all `mycelium_fractal_net` subpackages automatically

### Documentation
- Added LICENSE (MIT)
- Added SECURITY.md with vulnerability reporting and security best practices
- Added CHANGELOG.md (this file)

## [4.1.0] - 2024-XX-XX

### Added
- Neuro-fractal mycelium dynamics engine
- FastAPI REST API with WebSocket streaming
- Cryptographic operations (AES-256-GCM, Ed25519 signatures)
- Federated learning with Byzantine-robust Krum aggregation
- Prometheus metrics and structured logging
- Docker multi-stage build
- Kubernetes manifests with HPA, network policies, and monitoring
- Comprehensive test suite with security tests

### Security
- API key authentication
- Rate limiting with token bucket algorithm
- Production-grade encryption (AES-256-GCM)
- Digital signatures (Ed25519)
- Audit logging with request IDs

[Unreleased]: https://github.com/neuron7x/mycelium-fractal-net/compare/v4.1.0...HEAD
[4.1.0]: https://github.com/neuron7x/mycelium-fractal-net/releases/tag/v4.1.0
