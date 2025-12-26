## CFDE (Canonical Fractal Denoiser Engine) Model Contract

### A) Purpose
- Location: `src/mycelium_fractal_net/signal/denoise_1d.py::OptimizedFractalDenoise1D`.
- Role: structure-preserving 1D denoising for MFN pipelines (returns, biosignals) using fractal domain matching and overlap-add smoothing.
- Non-goal: must **not** amplify noise or rescale signals when the gate rejects reconstruction (identity fallback remains valid).

### B) Input / Output Contract
- Accepted shapes: `[L]`, `[B, L]`, `[B, C, L]` via `_canonicalize_1d` in `signal/denoise_1d.py`.
- Dtype: inputs preserved; internal ops run in `torch.float64` and outputs are cast back to the original dtype in `OptimizedFractalDenoise1D.forward`.
- Device: CPU/GPU respected; tensors stay on the caller’s device (no implicit host/device moves).

### C) Cognitive Loop Mapping (code-tied)
- **Thalamic Filter** — variance-ranked domain selection: `_denoise_fractal` → `var_pool`/`topk` in `signal/denoise_1d.py`.
- **Basal Ganglia Gate** — acceptance rule `mse_best < baseline_mse * gate_ratio`: `apply_fractal` mask inside `_denoise_fractal` in `signal/denoise_1d.py`.
- **Dopamine Gating** — ridge + clamp of scale `s`: `ridge_lambda`, `s_max`, `s_threshold` handling in `_denoise_fractal` in `signal/denoise_1d.py`.
- **Recursion Loop** — acceptor iterations: enforced minimum via `acceptor_iterations` in `OptimizedFractalDenoise1D.__init__` and applied in `forward` (fractal mode) in `signal/denoise_1d.py`.

### D) Invariants (testable)
- **Do No Harm gate**: when `do_no_harm=True`, regions with `mse_best >= baseline_mse * gate_ratio` bypass reconstruction, preserving baseline (`_denoise_fractal` in `signal/denoise_1d.py`).
- **Stability**: recursive passes should not increase proxy energy; validated by `tests/test_signal_denoise_1d.py::test_recursive_energy_stability` and `tests/test_signal_denoise_1d.py::test_cfde_recursive_monotonic_energy`.

### E) Failure Modes / Limitations
- Gate-off: if `fractal_dim_threshold` inhibits all ranges, `forward` returns the input unchanged (`signal/denoise_1d.py`).
- Edge padding: reflect padding falls back to replicate on very short signals in `_denoise_fractal`, so boundary artifacts may persist.
- Sensitivity: extremely small `population_size`/`range_size` reduce effectiveness; no exception is raised (see `Fractal1DPreprocessor` presets in `signal/preprocessor.py`).

### Canonical Pipeline Hook
- Finance pipeline hook: `examples/finance_regime_detection.py::map_returns_to_field` (parameter `denoise` + `cfde_preset`) applies `Fractal1DPreprocessor` before mapping returns into the MFN field.
