## CI Facts (PR-visible)

- **pr-gate.yml** — runs on `pull_request` (main), `merge_group`, `push` (main), `workflow_dispatch`; jobs: PR Gate / lint, PR Gate / typecheck, PR Gate / tests-fast (pytest + JUnit + coverage artifacts), PR Gate / security-min (bandit + pip-audit).
- **ci.yml → ci-reusable.yml** — runs on `pull_request` (main, develop), `push` (main, develop), `merge_group`, `workflow_dispatch`; includes workflow lint, dependency review, config lint, lint/format, typecheck, matrix tests with coverage artifacts, security scans, secrets scan, IaC security, docs check, validation, benchmarks, scientific validation, scalability, packaging, and CI summary.
- **codeql.yml** — runs CodeQL analysis on `pull_request` (main), `push` (main), and scheduled weekly.
