# MFN Validation Report — Experimental Validation & Falsification

**Document Version**: 1.0  
**Last Updated**: 2025-11-29  
**Applies to**: MyceliumFractalNet v4.1.0

---

## Executive Summary

This report documents the experimental validation and potential falsification of 
the MyceliumFractalNet mathematical models. All core invariants have been tested 
and verified. No falsification signals were detected.

**Overall Status**: ✅ **VALIDATED**

| Component | Status | Confidence |
|-----------|--------|------------|
| Nernst Equation | ✅ PASS | HIGH |
| Reaction-Diffusion (Turing) | ✅ PASS | HIGH |
| Fractal Growth (IFS) | ✅ PASS | HIGH |
| Numerical Stability | ✅ PASS | HIGH |
| Feature Extraction | ⚠️ PASS* | MEDIUM |

*Note: Fractal dimension extraction requires adaptive thresholding; see Section 4.

---

## 1. Control Scenarios (Ground Truth)

### 1.1 Scenario: Stability Under Pure Diffusion

**Configuration**:
- `spike_probability = 0.0`
- `turing_enabled = False`
- `quantum_jitter = False`
- `alpha = 0.18`

**Expectation**: Field variance should decrease (diffusion homogenizes).

**Result**: ✅ **PASS**

| Metric | Initial | Final | Status |
|--------|---------|-------|--------|
| Variance | 9.9e-5 | 5.3e-6 | ✅ Decreased |
| NaN/Inf | False | False | ✅ None |
| Bounds | [-95, 40] mV | [-95, 40] mV | ✅ Within |

**Conclusion**: Diffusion equation correctly implemented; smoothing behavior verified.

---

### 1.2 Scenario: Growth with Spike Events

**Configuration**:
- `spike_probability = 0.5`
- `turing_enabled = False`
- `steps = 100`

**Expectation**: Growth events should occur (>0), field evolution observed.

**Result**: ✅ **PASS**

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| Growth events | >0 | ~50 | ✅ |
| Field bounded | Yes | Yes | ✅ |

**Conclusion**: Spike event mechanism works as expected.

---

### 1.3 Scenario: Turing Pattern Formation

**Configuration**:
- `turing_enabled = True`
- `steps = 200`
- `grid_size = 64`

**Expectation**: Turing patterns produce measurably different field than baseline.

**Result**: ✅ **PASS**

| Metric | Value | Status |
|--------|-------|--------|
| Max difference from baseline | 2.2 mV | ✅ >0 |
| Field finite | True | ✅ |

**Conclusion**: Turing morphogenesis produces distinct spatial patterns.

---

### 1.4 Scenario: Quantum Jitter Stability

**Configuration**:
- `quantum_jitter = True`
- `jitter_var = 0.0005`
- `steps = 500`

**Expectation**: System remains stable with stochastic noise.

**Result**: ✅ **PASS**

| Metric | Value | Status |
|--------|-------|--------|
| NaN/Inf | None | ✅ |
| Within bounds | Yes | ✅ |

**Conclusion**: Stochastic term properly bounded and integrated.

---

### 1.5 Scenario: Near-CFL Stability

**Configuration**:
- `alpha = 0.24` (CFL limit = 0.25)
- `steps = 200`

**Expectation**: System stable at near-limit diffusion coefficient.

**Result**: ✅ **PASS**

| Metric | Value | Status |
|--------|-------|--------|
| Finite | True | ✅ |
| Bounded | Yes | ✅ |

**Conclusion**: Numerical scheme stable up to CFL boundary.

---

### 1.6 Scenario: Long-Run Stability (1000 steps)

**Configuration**:
- `steps = 1000`
- `turing_enabled = True`
- `quantum_jitter = True`

**Expectation**: No numerical drift or explosion.

**Result**: ✅ **PASS**

| Metric | Value | Status |
|--------|-------|--------|
| Finite after 1000 steps | True | ✅ |
| Growth events | ~247 | ✅ Expected |

**Conclusion**: Extended simulations numerically stable.

---

## 2. Core Invariants Testing

### 2.1 Nernst Equation Physical Bounds

**Test**: Membrane potentials should be in [-150, +150] mV for physiological concentrations.

| Ion | Computed | Expected | Error | Status |
|-----|----------|----------|-------|--------|
| K⁺ | -89.0 mV | -89 mV | 0.0 mV | ✅ |
| Na⁺ | +66.6 mV | +66 mV | 0.6 mV | ✅ |
| Cl⁻ | -90.9 mV | -89 mV | 1.9 mV | ✅ |
| Ca²⁺ | +101.5 mV | +102 mV | 0.5 mV | ✅ |

**Conclusion**: ✅ Nernst equation correctly implemented per MFN_MATH_MODEL.md.

---

### 2.2 Field Clamping Enforcement

**Test**: Field values always clamped to [-95, 40] mV.

| Condition | Min (mV) | Max (mV) | Status |
|-----------|----------|----------|--------|
| Extreme spikes (p=0.9) | -95.0 | 40.0 | ✅ |
| Long run (1000 steps) | -95.0 | 40.0 | ✅ |

**Conclusion**: ✅ Clamping enforced under all tested conditions.

---

### 2.3 IFS Contraction Guarantee

**Test**: Lyapunov exponent should be negative (contractive dynamics).

| Trials | Mean λ | Min λ | Max λ | All Negative | Status |
|--------|--------|-------|-------|--------------|--------|
| 10 | -2.22 | -2.53 | -1.89 | ✅ Yes | ✅ |

**Conclusion**: ✅ IFS always produces stable, contractive fractals.

---

### 2.4 Fractal Dimension Bounds

**Test**: D ∈ [0, 2] for 2D binary fields.

| Threshold Method | Mean D | Std D | In Range | Status |
|------------------|--------|-------|----------|--------|
| Percentile (50%) | 1.766 | 0.007 | ✅ | ✅ |

**Conclusion**: ✅ Fractal dimension within valid mathematical bounds.

---

### 2.5 Reproducibility

**Test**: Same seed produces identical results.

| Metric | Identical | Status |
|--------|-----------|--------|
| Field values | ✅ | ✅ |
| Growth events | ✅ | ✅ |
| Fractal dimension | ✅ | ✅ |

**Conclusion**: ✅ Determinism preserved.

---

## 3. Falsification Tests

### 3.1 Diffusion Smoothing Effect

**Hypothesis**: Diffusion should reduce spatial variance.

**Test**: Compare initial vs final variance under pure diffusion.

**Result**: ✅ **NOT FALSIFIED**

Initial variance: 9.9e-5, Final variance: 5.3e-6 (94% reduction).

---

### 3.2 Nernst Sign Consistency

**Hypothesis**: If [X]_out > [X]_in and z > 0, then E > 0.

**Test**: Verify sign for multiple ion configurations.

**Result**: ✅ **NOT FALSIFIED**

All tested configurations show correct sign relationship.

---

### 3.3 IFS Bounded Attractor

**Hypothesis**: Contractive IFS must have bounded attractor.

**Test**: Check max coordinate magnitude after 10,000 iterations.

**Result**: ✅ **NOT FALSIFIED**

Max coordinate: <10 (well bounded).

---

### 3.4 CFL Stability Boundary

**Hypothesis**: System stable below CFL limit (α < 0.25).

**Test**: Run at α = 0.24.

**Result**: ✅ **NOT FALSIFIED**

System remains finite and bounded.

---

## 4. Feature Extraction Findings

### 4.1 Threshold Sensitivity Issue

**Finding**: The default -60 mV threshold for fractal dimension calculation may not capture any active cells when field values concentrate around -70 mV.

**Observation**:
- Field range: typically [-71, -66] mV
- At -60 mV threshold: 0% active cells → D = 0 (invalid)
- At 50th percentile threshold: 50% active cells → D ≈ 1.77 (valid)

**Recommendation**: Use adaptive (percentile-based) thresholding for robust feature extraction.

**Status**: ⚠️ **DOCUMENTED** — Not a model failure, but threshold parameter needs tuning per use case.

---

### 4.2 Regime Discrimination

**Test**: Features should differentiate between simulation regimes.

| Regime | Field Std (mV) | Observation |
|--------|----------------|-------------|
| Stable (no activity) | 0.23 ± 0.01 | Baseline |
| Active (spikes) | 0.33 ± 0.04 | Higher variance |
| Turing | 0.29 ± 0.02 | Intermediate |

**Conclusion**: ✅ Standard deviation discriminates between regimes.

---

## 5. Validation Summary Table

| Scenario | Expectation | Result | Status |
|----------|-------------|--------|--------|
| Stability (diffusion only) | Variance decreases | Variance reduced 94% | ✅ PASS |
| Growth events | Events occur with p>0 | ~50 events at p=0.5 | ✅ PASS |
| Turing patterns | Different from baseline | Max diff = 2.2 mV | ✅ PASS |
| Quantum jitter | System stable | No NaN/Inf | ✅ PASS |
| Near-CFL (α=0.24) | Stable below limit | No instability | ✅ PASS |
| Long-run (1000 steps) | No drift | Bounded and finite | ✅ PASS |
| Nernst accuracy | ±5 mV of literature | All within tolerance | ✅ PASS |
| Field clamping | [-95, 40] mV | Enforced | ✅ PASS |
| IFS contraction | λ < 0 | Mean λ = -2.22 | ✅ PASS |
| Fractal dimension | D ∈ [0, 2] | D = 1.77 | ✅ PASS |
| Reproducibility | Same seed → same result | Verified | ✅ PASS |

---

## 6. Conclusions

### 6.1 Model Validity

All core mathematical models have been experimentally validated:

1. **Nernst Equation**: Correctly computes ion equilibrium potentials within literature tolerance.

2. **Reaction-Diffusion**: 
   - Diffusion smoothing verified
   - Turing morphogenesis produces distinct patterns
   - CFL stability condition respected

3. **Fractal Growth**:
   - IFS consistently contractive (λ < 0)
   - Box-counting dimension in valid range

4. **Numerical Stability**:
   - No NaN/Inf under any tested condition
   - Field clamping properly enforced
   - Long-run stability verified

### 6.2 Items for Future Improvement

1. **Threshold Calibration**: Consider dynamic thresholding for fractal dimension to handle varying field distributions.

2. **Feature Normalization**: Feature values depend on grid size and step count; normalization may improve cross-experiment comparability.

3. **Statistical Power**: Current validation uses 10 seeds; larger sample sizes could increase confidence.

### 6.3 Falsification Status

**No falsification signals detected.** All tested predictions align with model expectations.

---

## 7. Test Coverage

The validation tests are implemented in:
- `tests/validation/test_model_falsification.py` — Control scenarios and falsification tests
- `tests/test_math_model_validation.py` — Mathematical property tests
- `validation/scientific_validation.py` — Literature comparison

Run all validation tests:
```bash
pytest tests/validation/ tests/test_math_model_validation.py -v
python validation/scientific_validation.py
```

---

## 8. References

- `docs/MFN_MATH_MODEL.md` — Mathematical model specification
- `docs/MFN_FEATURE_SCHEMA.md` — Feature extraction specification
- `docs/VALIDATION_NOTES.md` — Expected metric ranges

---

*Document Author: Automated Validation System*  
*Review Status: Pending human review*
