# Known Limitations

## Scale Limits

| Grid Size | Status | Notes |
|-----------|--------|-------|
| Up to 512x512 | Recommended | Production-ready, benchmarked |
| 512x512 long-history | Stress contour | Requires memmap history backend |
| 1024x1024 | Experimental | OOM profiling incomplete; explicit opt-in only |

The default release contour is CPU-first and deterministic. ML/GPU surfaces are optional extras.

For temporal history beyond 64 steps on grids larger than 128x128, use disk-backed history:
```python
mfn.simulate(spec, history_backend="memmap")
```

## Frozen Surfaces

The following surfaces are frozen and excluded from the v1 release contract:

| Surface | Location | Status |
|---------|----------|--------|
| Cryptography helpers | `src/mycelium_fractal_net/crypto/` | Deprecated; signing via `artifact_bundle.py`. Removal in v5.0. |
| Federated logic | `src/mycelium_fractal_net/core/federated.py` | Frozen; no active development |
| WebSocket adapters | `src/mycelium_fractal_net/integration/ws_*` | Frozen; not in v1 scope |
| Infrastructure | `infra/` | Deployment material; not library code |
| Historical docs | `docs/project-history/` | Archive; not maintained |

## Type Checking

- `mypy strict` is enforced for `types/` and `security/` modules.
- Core domain modules (`core/`, `analytics/`, `neurochem/`, `pipelines/`) report type errors but do not block CI. Migration to full strict typing is tracked for v5.0.

## Causal Validation

- 41 rules cover the six pipeline stages plus cross-stage and perturbation checks.
- Exhaustive 512x512 / 1024x1024 OOM profiling with published thresholds is not yet closed.
- Formal public config limit reduction / promotion policy is not yet established.

## Neuromodulation

- Occupancy conservation is enforced numerically (error < 1e-6) but not algebraically proven.
- Observation noise model (`observation_noise_bold_like`) applies Gaussian temporal smoothing, not a hemodynamic response function. The name is aspirational — a true BOLD model requires HRF convolution (Buxton et al. 1998). Planned for v5.0.
- Profile parameter ranges are based on published literature but not independently calibrated.

## Dependencies

- `torch` is optional (`[ml]` extra). All core operations work without it.
- `numba` is optional (`[accel]` extra). JIT acceleration for Laplacian computation only.
- Root compatibility imports (e.g., `import analytics`) are deprecated and will be removed in v5.0.
