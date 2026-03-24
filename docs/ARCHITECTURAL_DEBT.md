# Architectural Debt Register

## Resolved

| Item | Was | Now | Commit |
|------|-----|-----|--------|
| model.py monolith | 1329 LOC | 13 LOC (4 modules) | `e11d005` |
| config.py monolith | 810 LOC | 318 LOC + validation | `e11d005` |
| api.py monolith | 1062 LOC | 937 + api_v1.py | `e11d005` |
| Flat fitness function | f≈0.045 all configs | Additive discriminating | `41dd323` |
| Memory query O(N) loop | 1.4ms interleaved | 0.07ms pre-allocated | `78b059a` |
| Physarum rebuild matrix | 28.9ms/step | 3.0ms sparse cached | `845cce2` |
| NaN propagation | Silent leak | np.where guard | `58bb8c9` |
| BioMemory dirty flag | Unconditional rebuild | Append fast path | `78b059a` |

## Active

| Item | Current | Cap | Target |
|------|---------|-----|--------|
| causal_validation.py | 1021 LOC | 1050 | Accept: living spec pattern |
| api.py | 937 LOC | 950 | v5.0: extract WS handlers |
| Frozen surface | ~2300 LOC | 3500 | v5.0: remove crypto + frozen modules |
| Benchmark calibration | Local baselines | CI baselines | CI recalibration on merge |

## Rules

1. No module > 800 LOC without exemption + per-file cap
2. Baseline updates require before/after numbers + commit message justification
3. Performance gates use calibrated baselines × 3.0 multiplier
4. All gates must pass on clean CI run to close protocol
5. hypothesis is mandatory in test environment
