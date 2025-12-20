# Defect Registry

This registry captures defects identified during the current review pass.
Scope of evidence: targeted static review of API logging components and
execution of `python -m pytest tests/integration/test_api_logging.py -q`.

| ID | Location | Description | Root Cause | Impact | Severity | Priority | Repro | Resolution | Closed Criteria | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MFN-LOG-INT-001 | `src/mycelium_fractal_net/integration/logging_config.py` (`set_request_id`) | `set_request_id` accepted only `str` while call sites passed `None` to clear context, requiring `# type: ignore` and risking misuse. | Type signature too narrow for context clearing behavior. | Type-checking noise and potential misuse when callers attempt to clear context without a helper. | Low | Medium | Run logging tests or inspect middleware/tests calling `set_request_id(None)`. | Expand `set_request_id` to accept `Optional[str]` and update call sites to remove type ignores. | `set_request_id(None)` is type-safe and tests pass without ignores. | `python -m pytest tests/integration/test_api_logging.py -q` |
