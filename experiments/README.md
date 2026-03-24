# Experiments

Reproducible experiments for MFN. Anyone can run these and verify identical results.

## Canonical Reproduction

```bash
uv run python experiments/reproduce.py
```

This runs:
1. Simulation (N=32, T=60, seed=42) → deterministic field with hash `b407b808c7c8a03f`
2. Core diagnosis → severity, anomaly, EWS, causal gate
3. Unified engine → basin stability, fractal dynamics, Hurst, chi invariant
4. Math frontier → TDA, Wasserstein, Causal Emergence, RMT

**Expected output:** `experiments/expected_output.json` (committed, versioned)

**Actual output:** `experiments/actual_output.json` (generated on run)

If actual != expected → something changed in the pipeline. Investigate before merging.

## Key Numbers (seed=42, N=32, T=60)

| Metric | Value | Meaning |
|--------|-------|---------|
| field_hash | `b407b808c7c8a03f` | Deterministic — identical across runs |
| severity | `info` | System approaching transition |
| anomaly_score | 0.2157 | Nominal |
| ews_score | 0.4623 | Approaching transition |
| causal_decision | `pass` | All 46 rules satisfied |
| basin_stability | 0.9333 | 93% of perturbations return |
| delta_alpha | 3.5932 | Genuine multifractal |
| hurst_exponent | 2.1046 | Super-persistent (critical) |
| chi_invariant | 0.5986 | Transitional regime |
| W2_speed | 1.3306 | Transport distance from initial state |
| RMT r_ratio | 0.0251 | Physarum highly structured |
