# Data Model

## Canonical runtime contracts
- `FieldSequence`
- `SimulationSpec`
- `MorphologyDescriptor`
- `AnomalyEvent`
- `RegimeState`
- `ForecastResult`
- `ComparisonResult`
- `AnalysisReport`

## Contract guarantees
- explicit `schema_version`
- deterministic dtype normalization
- round-trip `to_dict()/from_dict()` semantics
- JSON-safe summaries for public surfaces
- binary arrays persisted as `.npy` artifacts
- report manifests include `engine_version`, `schema_version`, `seed`, `config_hash`, `git_sha`, `lock_hash`

See `tests/test_schema_roundtrip_completion.py` and `docs/contracts/openapi.v1.json`.


## NeuromodulationSpec completion
`NeuromodulationSpec` includes `profile_id`, `evidence_version`, `baseline_activation_offset_mv`, `tonic_inhibition_scale`, and `gain_fluidity_coeff`. Nested GABAA payloads carry contract-complete kinetics and resting-offset fields.
