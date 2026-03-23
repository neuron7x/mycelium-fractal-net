# Architectural Debt Register

## Active Debt

| Module | Current LOC | Cap | Issue | Target |
|--------|-------------|-----|-------|--------|
| `causal_validation.py` | 1021 | 1050 | Living spec — 46 rules + orchestrator | Accept: monolith justified |
| `api.py` | 937 | 950 | WS handlers still inline | v5.0: extract WS to separate service |
| Frozen surface | 3317 | 3500 | crypto/ + signal/ + federated/stdp/turing | v5.0: remove |

## Resolved Debt

| Module | Was | Now | Action |
|--------|-----|-----|--------|
| `model.py` | 1329 LOC | 13 LOC (re-export) | Split into model_pkg/ (4 modules, max 433 LOC) |
| `config.py` | 810 LOC | 318 LOC | Validation extracted to config_validation.py |
| `api.py` | 1062 LOC | 937 LOC | V1 endpoints extracted to api_v1.py |

## Rules

1. No new module may exceed 800 LOC without explicit exemption
2. Exempt modules have per-file caps — growth requires cap raise with justification
3. Frozen surface budget: 3500 LOC — additions require removal elsewhere
4. New subsystems must be in their own package with `__init__.py` boundary
5. Cross-layer imports require whitelist entry in `.importlinter`
