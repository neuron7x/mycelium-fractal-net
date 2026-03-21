# API

## Canonical endpoints
- `GET /health`
- `GET /metrics`
- `POST /v1/simulate`
- `POST /v1/extract`
- `POST /v1/detect`
- `POST /v1/forecast`
- `POST /v1/compare`
- `POST /v1/report`

## Contract source of truth
- runtime implementation: `src/mycelium_fractal_net/integration/api_server.py`
- exported contract: `docs/contracts/openapi.v2.json`
- verification: `scripts/export_openapi.py` + `scripts/check_openapi_contract.py`

## Health payload
`engine_version`, `api_version`, `uptime`, `status`

## Metrics payload
request counters for `simulate`, `extract`, `detect`, `forecast`, plus latency snapshot.

## Semantic parity rule
SDK is the semantic source. CLI and API are orchestration layers over the same engine functions.

## Neuromodulation surface
`POST /v1/simulate` and all spec-carrying requests accept an optional nested `neuromodulation` object with:
- `profile`
- `enabled`
- `dt_seconds`
- `intrinsic_field_jitter`
- `intrinsic_field_jitter_var`
- `gabaa_tonic`
- `serotonergic`
- `observation_noise`

Backwards compatibility is preserved: omitting `neuromodulation` is equivalent to the v4.1.0 baseline path.


All `/v1/*` simulation endpoints carry nested `neuromodulation` contract coverage in OpenAPI v2.
