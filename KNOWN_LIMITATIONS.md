# Known Limitations

## Frozen surfaces
The following surfaces are frozen outside the v1 release contract:
- cryptography helpers under `src/mycelium_fractal_net/crypto/`
- federated logic under `src/mycelium_fractal_net/core/federated.py`
- websocket adapters under `src/mycelium_fractal_net/integration/ws_*`
- infra/deployment material under `infra/`
- historical notebooks and exploratory planning under `docs/project-history/`

## Current release discipline
- `uv` is the canonical local/CI toolchain
- `mfn` is the canonical user surface
- root compatibility imports are deprecated and shipped only through installed shims from `src/`


## Scale limits
- The default release contour is CPU-first and deterministic; ML/GPU surfaces are optional extras.
- Disk-backed history (`simulate_history(..., history_backend="memmap")`) is the preferred path once temporal history becomes the dominant memory consumer.
- 512x512 long-history runs are treated as the current stress contour for benchmark and nightly evidence.
- 1024x1024 remains experimental and should be invoked only through explicit perf/experimental paths until memory profiling evidence is refreshed and published.
- Bundle verification and signature checks are mandatory before treating any generated report/release/showcase artifact as authoritative.
