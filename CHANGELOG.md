# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2025-12-19
### Security
- Require explicit opt-in for public API docs outside dev; added env overrides for public endpoints.
- Hardened rate limiting client identification by disabling untrusted proxy headers by default.

### Infrastructure
- Removed empty Kubernetes Secret manifest to prevent accidental deployments with blank API keys.
- Split development dependencies into `requirements-dev.txt` for leaner runtime images.

### Observability
- Added file-based persistence for API request/response summaries (JSONL) for auditability.
