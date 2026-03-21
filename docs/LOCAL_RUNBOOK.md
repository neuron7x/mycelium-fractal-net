# Local Runbook

## Install profiles
```bash
uv sync --locked --group dev                 # core-only deterministic engine
uv sync --locked --group dev --extra ml      # ML / torch surfaces
uv sync --locked --group dev --extra accel   # Numba Laplacian acceleration
```


## Hermetic setup
```bash
make sync
make doctor
```

## Verification
```bash
make verify
make validate
make benchmark
```

## API
```bash
make api
```

## Report
```bash
make report
```

## Demo scenarios
```bash
make demo-scenarios
make showcase
```

## Contract checks
```bash
make contracts
```

## Neuromodulation / criticality sweep
```bash
uv run python validation/neurochem_controls.py
uv run python scripts/criticality_sweep.py
uv run mfn report --grid-size 24 --steps 16 --neuromod-profile gabaa_tonic_muscimol_alpha1beta3 --agonist-concentration-um 0.85 --output-root artifacts/runs
```

## Regression hardening
```bash
uv run python scripts/showcase_run.py    # showcase-generation
uv run python scripts/baseline_parity.py  # baseline-parity
uv run python scripts/docs_drift_check.py # docs-drift
uv run python scripts/attest_artifacts.py # release provenance / attestation
```

## Bundle verification
```bash
uv run mfn verify-bundle artifacts/showcase/showcase_manifest.json
uv run mfn verify-bundle artifacts/release/release_manifest.json
```
