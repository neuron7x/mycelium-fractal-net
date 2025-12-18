# Continuous Integration and Security Policy

This repository ships with a hardened CI/CD baseline that must run on every pull request before code is merged to `main` or `develop`.

## Required status checks before merging

Enable branch protection on `main` (and `develop` if applicable) with **Require status checks to pass before merging** and include these checks:

- `dependency-review`
- `codeql`
- `lint`
- `security`
- `secret-detection`
- `config-scan`
- `test`
- `validate`
- `benchmark`
- `scientific-validation`
- `scalability-test`
- `packaging`

Set **Require branches to be up to date** and **Require pull request reviews before merging** to enforce gated integration.

## Security and quality controls

| Layer | Tooling | Notes |
| --- | --- | --- |
| Dependency governance | Dependabot + `pip-audit` | Dependabot PRs are scheduled weekly for GitHub Actions, Python (`pip`), and Docker ecosystems. |
| Static application security testing | CodeQL | Runs on every PR. |
| Secrets detection | Gitleaks | Blocks accidental key or token commits and uploads SARIF results. |
| Configuration/IaC scanning | Trivy config | Fails on HIGH/CRITICAL findings and uploads SARIF results. |
| Application security tests | `pytest tests/security` | Included in the `security` job. |
| Linting | Ruff + Mypy | Enforced in the `lint` job. |
| Unit/integration tests | Pytest matrix | The `test` job exercises the full suite with coverage across Python 3.10–3.12. |
| Validation/benchmarks | Domain validations and benchmarks | See `validate`, `benchmark`, `scientific-validation`, `scalability-test`, and `packaging` jobs. |

Secrets scanning walks the full git history of the pull request branch and uploads SARIF results to GitHub code scanning. Dependency audits check both `requirements.txt` and `pyproject.toml`.
If Gitleaks produces no SARIF (e.g., no findings), an empty SARIF is generated and stored as an artifact to guarantee uploads succeed for compliance visibility. CodeQL actions are pinned to v4 to avoid deprecation.

## Notifications

The `notify` job aggregates all workflow results and can fan out alerts:

- **Slack:** set a `SLACK_WEBHOOK_URL` secret.
- **Microsoft Teams:** set a `TEAMS_WEBHOOK_URL` secret.
- **Email:** set `EMAIL_SMTP_SERVER`, `EMAIL_SMTP_USERNAME`, `EMAIL_SMTP_PASSWORD`, `EMAIL_FROM`, and `EMAIL_TO` secrets.

If no endpoints are configured, the job logs guidance so you can wire an alerting channel before relying on the automation.

## Operational notes

- All jobs run on every pull request; failures block merges when required status checks are configured.
- Concurrency is enabled to cancel superseded runs for the same branch or PR, keeping feedback fast and current.
- Security scan SARIF uploads populate the repository’s Code Scanning alerts for centralized visibility.
