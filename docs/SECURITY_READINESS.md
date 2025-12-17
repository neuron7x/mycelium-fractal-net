# Security & Reliability Readiness (2025)

**Risk posture**: Internet-facing, defaulting to OWASP ASVS **L2** with selective L3 controls for crypto key handling and audit logs. Process aligned to NIST SSDF v1.1 (Prepare/Protect/Produce/Respond) and SLSA build integrity (target L2, roadmap to L3).

## Technical audit (high-level)
- **Top risks**: weak/invented crypto, missing provenance/SBOM, unbounded payloads, insufficient auditability, supply-chain exposure to transitive vulns.
- **Mitigations delivered**:
  - Replaced home-grown XOR+HMAC with **AES-256-GCM** using `cryptography` and strict key length enforcement.
  - Added payload size guardrails and tamper-evident failures.
  - CI hardening with SBOM emission for dependency visibility (CycloneDX 1.7 JSON).
  - Regression tests for crypto edge cases.
- **Definition of Done (increment)**:
  - All security unit tests green; CI SBOM job succeeds and uploads artifact.
  - No high/critical dependency vulns (pip-audit gate already in CI).
  - Crypto interfaces reject misconfiguration (wrong key length, oversize payloads) with auditable errors.

## Threat model (concise)
- **Assets**: encryption keys, plaintext secrets/config, SBOM/provenance artifacts, audit logs.
- **Actors**: external callers (API), insiders with repo access, CI agents.
- **Trust boundaries**: API <-> crypto adapters; storage of ciphertext; CI pipeline artifact storage.
- **Key threats** (mapped to OWASP Top 10/ASVS):
  - A02 Cryptographic Failures: mitigated via AES-GCM, nonce-per-encryption, auth tags.
  - A08 Software Integrity: mitigated via SBOM + dependency scanning, provenance roadmap.
  - A05 SSRF/DoS via large payloads: mitigated via 1 MiB crypto payload guard.
  - A09 Logging/Auditing gaps: crypto API retains audit hooks (configurable) for success/failure metadata.

## Operational runbook (excerpt)
- **Key management**: 32-byte keys only; rotate by provisioning new key IDs via `crypto_adapters` and deprecate old ones. Store keys in KMS/secret manager; never log plaintext.
- **Incident response (Respond)**: On suspected compromise, rotate keys, invalidate ciphertext, regenerate SBOM to assess exposure, and rerun CI security jobs. Review audit logs for failed decrypt/authentication errors.
- **Build reproducibility**: Prefer `pip install -e .[dev]` with lockfile once available; SBOM captured per CI run (artifact retention >= 30 days).
- **PR security checklist**:
  - Threats addressed/updated? Crypto/auth changes require test additions.
  - Tests: unit + security suites must pass; new inputs validated with negative cases.
  - Dependencies: no HIGH/CRITICAL `pip-audit` findings; SBOM job succeeds.
  - Observability: structured logs for failures; no secrets in logs.

## Roadmap (next steps)
- Add dependency update automation (Renovate/Dependabot) with policy to auto-block high CVEs.
- Add SLSA provenance attestation on build artifacts and signed SBOM (in-toto). 
- Extend DAST in staging (basic auth/ZAP) and fuzzing for parsers.
- Integrate rate limiting and circuit breakers in API layer; expand E2E authz tests to hit 95% branch coverage on critical paths.
