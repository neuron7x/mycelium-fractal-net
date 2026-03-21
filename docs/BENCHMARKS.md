# Benchmarks

## Canonical benchmark commands
```bash
uv run python benchmarks/benchmark_core.py
uv run python benchmarks/benchmark_scalability.py
uv run python benchmarks/benchmark_quality.py
```

## Output roots
- `benchmarks/results/benchmark_core.json`
- `benchmarks/results/benchmark_core.csv`
- `benchmarks/results/benchmark_scalability.json`
- `benchmarks/results/benchmark_scalability.csv`
- `benchmarks/results/benchmark_quality.json`
- `benchmarks/results/benchmark_quality.csv`

## Quality benchmark policy
Quality benchmarks must include explicit baselines and ablation-style comparisons where implemented. Public claims must point to generated files, not prose-only statements.


Benchmark acceptance is consumed by the showcase_generation, baseline parity, and attestation release contour.
