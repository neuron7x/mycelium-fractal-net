# MFN Performance Baselines

Performance baselines for MyceliumFractalNet v4.1 key execution paths.

These baselines are used by `tests/perf/test_mfn_performance.py` to verify
that performance does not regress beyond the allowed threshold (+20%).

---

## Test Environment

- Python: 3.10+
- NumPy: 1.24+
- Hardware: Standard GitHub Actions runner (2 vCPU, 7GB RAM)
- Measurement: Average over multiple runs with `time.perf_counter()`

---

## 1. Simulation Performance

### 1.1 Demo Configuration

Configuration: `make_simulation_config_demo()`
- grid_size: 32
- steps: 32
- seed: 42

| Metric | Baseline | Threshold (+20%) |
|--------|----------|------------------|
| Time (s) | 0.040 | 0.048 |
| Peak Memory (KB) | 1000 | 1200 |

### 1.2 Default Configuration

Configuration: `make_simulation_config_default()`
- grid_size: 64
- steps: 100
- seed: 42

| Metric | Baseline | Threshold (+20%) |
|--------|----------|------------------|
| Time (s) | 0.120 | 0.144 |
| Peak Memory (KB) | 500 | 600 |

---

## 2. Feature Extraction Performance

### 2.1 Demo Configuration

Input: SimulationResult from demo config with history

| Metric | Baseline | Threshold (+20%) |
|--------|----------|------------------|
| Time (s) | 0.010 | 0.012 |
| Peak Memory (KB) | 2000 | 2400 |

---

## 3. Dataset Generation Performance

### 3.1 Demo Configuration (5 samples)

Configuration: `make_dataset_config_demo()` (5 samples)
- grid_sizes: [32]
- steps_range: (30, 50)
- num_samples: 5

| Metric | Baseline | Threshold (+20%) |
|--------|----------|------------------|
| Total Time (s) | 0.300 | 0.360 |
| Per Sample (s) | 0.060 | 0.072 |
| Peak Memory (KB) | 1500 | 1800 |

---

## 4. End-to-End Pipeline Performance

### 4.1 Full Pipeline (Demo)

1. SimulationConfig → run_mycelium_simulation → SimulationResult
2. SimulationResult → compute_fractal_features → FeatureVector
3. DatasetConfig → generate_dataset → dataset file

| Stage | Baseline (s) | Threshold (+20%) |
|-------|--------------|------------------|
| Simulation | 0.040 | 0.048 |
| Features | 0.010 | 0.012 |
| Total Pipeline | 0.050 | 0.060 |

---

## 5. Notes

### Measurement Guidelines

1. Baselines are measured on standard CI hardware
2. Warm-up run is excluded from timing
3. Multiple runs averaged (3-5 typically)
4. Memory measured with `tracemalloc`

### Allowed Variance

- Time: +20% from baseline
- Memory: +20% from baseline

### Update Process

When optimizing code:
1. Run profiling script to get new measurements
2. Update baselines in this document
3. Update test thresholds if baselines improve

---

## 6. Commands

### Run Performance Tests

```bash
pytest tests/perf/test_mfn_performance.py -v
```

### Profile Key Paths

```bash
python -c "
from mycelium_fractal_net.config import make_simulation_config_demo
from mycelium_fractal_net import run_mycelium_simulation
import time

config = make_simulation_config_demo()
start = time.perf_counter()
result = run_mycelium_simulation(config)
print(f'Time: {time.perf_counter() - start:.4f}s')
"
```

---

*Document Version: 1.0*
*Last Updated: 2025-11-29*
*Applies to: MyceliumFractalNet v4.1.0*
