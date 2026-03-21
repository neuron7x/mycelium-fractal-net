# Architecture

## Canonical layers
- **contracts/types**: `src/mycelium_fractal_net/types/`
- **domain/core**: `src/mycelium_fractal_net/core/`
- **application/pipelines**: `src/mycelium_fractal_net/pipelines/`
- **adapters/integration**: `src/mycelium_fractal_net/integration/`
- **interfaces**: `src/mycelium_fractal_net/cli.py`, `src/mycelium_fractal_net/api.py`

## Boundary policy
- core must not import interfaces or transport
- pipelines must not import CLI/API transport directly
- interfaces orchestrate only canonical engine operations
- non-v1 surfaces are frozen and not part of the release contract

## Canonical v1 surface
`simulate`, `extract`, `detect`, `forecast`, `compare`, `report`

## Frozen surfaces
- `src/mycelium_fractal_net/crypto/`
- `src/mycelium_fractal_net/core/federated.py`
- websocket extras under `integration/`
- deployment, infra, historical notebooks, planning material

## Opt-in neurochem contour
- `src/mycelium_fractal_net/neurochem/` owns neuromodulation state, kinetics, MWC-like transforms, profile registry, and calibration tasks
- `core/reaction_diffusion_engine.py` remains the single canonical owner of the simulation step
- neuromodulation step order is fixed: bind -> unbind -> desensitize -> recover -> local excitability offset
- `numerics/update_rules.py` is compatibility-only numerical support and must not become a second equation-of-motion owner
- single canonical owner of the simulation step is enforced by `.importlinter` and CI
- neuromodulation is nested under `SimulationSpec.neuromodulation`; `None` is the hard baseline parity mode
- offset application is local and excitability-driven; no global resting-potential shift is allowed in the canonical kernel
- canonical profile registry: `baseline_nominal`, `gabaa_tonic_muscimol_alpha1beta3`, `gabaa_tonic_extrasynaptic_delta_high_affinity`, `serotonergic_reorganization_candidate`, `balanced_criticality_candidate`, `observation_noise_bold_like`
