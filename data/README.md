# Data Directory

This directory contains generated datasets from simulations.

## Files

Generated files (not committed to Git):
- `mycelium_dataset.parquet` - Main experimental dataset
- `*.npz` / `*.csv` - Alternative formats

## Generation

To generate a dataset:
```bash
python -m experiments.generate_dataset --output data/mycelium_dataset.parquet
```

See `docs/FEATURE_SCHEMA.md` for dataset schema documentation.
