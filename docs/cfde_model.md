## CFDE (Canonical Fractal Denoiser Engine) Model Contract

### Purpose
- Location: `src/mycelium_fractal_net/signal/denoise_1d.py::OptimizedFractalDenoise1D`.
- CFDE performs structure-preserving 1D denoising for MFN pipelines (financial returns, biosignals) using fractal domain matching.
- It must **not** amplify noise or alter signal scale when the gate rejects fractal reconstruction.

### Input / Output Contract
- Accepted shapes: `[L]`, `[B, L]`, `[B, C, L]` (see `_canonicalize_1d` in `denoise_1d.py`).
- Dtype: inputs preserved; internal ops use `torch.float64`, outputs restored to original dtype in `forward`.
- Device: CPU/GPU supported; tensors stay on input device (no host/device moves).

### Cognitive Loop Mapping
- **Thalamic Input Filter**: low-variance domain selection via variance-ranked domains (`_denoise_fractal`, `var_pool` + `topk` in `denoise_1d.py`).
- **Basal Ganglia Gating**: inhibition rule `mse_best < baseline_mse * gate_ratio` (`apply_fractal` in `_denoise_fractal`); gate ratio derives from `harm_ratio` / `inhibition_epsilon`.
- **Dopamine Gating**: ridge and clamping on scale parameters `s` (`ridge_lambda`, `s_max`, `s_threshold` in `_denoise_fractal`).
- **Recursion / Acceptor Loop**: minimum 7 recursive passes enforced by `acceptor_iterations` in `OptimizedFractalDenoise1D.forward`.

### Invariants (testable)
- **Do No Harm**: When `do_no_harm=True`, segments with `mse_best >= baseline_mse * gate_ratio` bypass reconstruction (baseline retained).
- **Stability Proxy**: Recursive passes should not increase proxy energy beyond tolerance (validated in `tests/test_signal_denoise_1d.py::test_recursive_energy_stability`).

### Failure Modes / Limitations
- Gate may bypass all ranges when `fractal_dim_threshold` inhibits (`forward` in `denoise_1d.py`), yielding identity output.
- Reflect padding falls back to replicate for very short signals (`_denoise_fractal`), so edge effects may persist on length-1 inputs.
- Designed for moderate sequence lengths; extremely small `population_size` or `range_size` can reduce effectiveness without raising errors.

### Canonical Pipeline Hook
- Finance pipeline entry: `examples/finance_regime_detection.py::run_finance_demo` uses `apply_cfde_preprocessing` with `cfde_preset` to enable CFDE on returns before field mapping.
