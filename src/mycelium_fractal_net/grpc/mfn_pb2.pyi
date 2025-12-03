from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar
from typing import Optional as _Optional
from typing import Union as _Union

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

class ResponseMeta(_message.Message):
    __slots__ = ("meta",)
    class MetaEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    META_FIELD_NUMBER: _ClassVar[int]
    meta: _containers.ScalarMap[str, str]
    def __init__(self, meta: _Optional[_Mapping[str, str]] = ...) -> None: ...

class FeatureRequest(_message.Message):
    __slots__ = ("request_id", "seed", "grid_size", "steps", "alpha", "spike_probability", "turing_enabled")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    GRID_SIZE_FIELD_NUMBER: _ClassVar[int]
    STEPS_FIELD_NUMBER: _ClassVar[int]
    ALPHA_FIELD_NUMBER: _ClassVar[int]
    SPIKE_PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    TURING_ENABLED_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    seed: int
    grid_size: int
    steps: int
    alpha: float
    spike_probability: float
    turing_enabled: bool
    def __init__(self, request_id: _Optional[str] = ..., seed: _Optional[int] = ..., grid_size: _Optional[int] = ..., steps: _Optional[int] = ..., alpha: _Optional[float] = ..., spike_probability: _Optional[float] = ..., turing_enabled: bool = ...) -> None: ...

class FeatureResponse(_message.Message):
    __slots__ = ("request_id", "meta", "fractal_dimension", "pot_min_mV", "pot_max_mV", "pot_mean_mV", "pot_std_mV", "growth_events")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    META_FIELD_NUMBER: _ClassVar[int]
    FRACTAL_DIMENSION_FIELD_NUMBER: _ClassVar[int]
    POT_MIN_MV_FIELD_NUMBER: _ClassVar[int]
    POT_MAX_MV_FIELD_NUMBER: _ClassVar[int]
    POT_MEAN_MV_FIELD_NUMBER: _ClassVar[int]
    POT_STD_MV_FIELD_NUMBER: _ClassVar[int]
    GROWTH_EVENTS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    meta: ResponseMeta
    fractal_dimension: float
    pot_min_mV: float
    pot_max_mV: float
    pot_mean_mV: float
    pot_std_mV: float
    growth_events: int
    def __init__(self, request_id: _Optional[str] = ..., meta: _Optional[_Union[ResponseMeta, _Mapping]] = ..., fractal_dimension: _Optional[float] = ..., pot_min_mV: _Optional[float] = ..., pot_max_mV: _Optional[float] = ..., pot_mean_mV: _Optional[float] = ..., pot_std_mV: _Optional[float] = ..., growth_events: _Optional[int] = ...) -> None: ...

class FeatureStreamRequest(_message.Message):
    __slots__ = ("request_id", "seed", "grid_size", "total_steps", "steps_per_frame", "alpha", "spike_probability", "turing_enabled")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    GRID_SIZE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_STEPS_FIELD_NUMBER: _ClassVar[int]
    STEPS_PER_FRAME_FIELD_NUMBER: _ClassVar[int]
    ALPHA_FIELD_NUMBER: _ClassVar[int]
    SPIKE_PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    TURING_ENABLED_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    seed: int
    grid_size: int
    total_steps: int
    steps_per_frame: int
    alpha: float
    spike_probability: float
    turing_enabled: bool
    def __init__(self, request_id: _Optional[str] = ..., seed: _Optional[int] = ..., grid_size: _Optional[int] = ..., total_steps: _Optional[int] = ..., steps_per_frame: _Optional[int] = ..., alpha: _Optional[float] = ..., spike_probability: _Optional[float] = ..., turing_enabled: bool = ...) -> None: ...

class FeatureFrame(_message.Message):
    __slots__ = ("request_id", "step", "fractal_dimension", "pot_min_mV", "pot_max_mV", "pot_mean_mV", "pot_std_mV", "growth_events", "is_final")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    FRACTAL_DIMENSION_FIELD_NUMBER: _ClassVar[int]
    POT_MIN_MV_FIELD_NUMBER: _ClassVar[int]
    POT_MAX_MV_FIELD_NUMBER: _ClassVar[int]
    POT_MEAN_MV_FIELD_NUMBER: _ClassVar[int]
    POT_STD_MV_FIELD_NUMBER: _ClassVar[int]
    GROWTH_EVENTS_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    step: int
    fractal_dimension: float
    pot_min_mV: float
    pot_max_mV: float
    pot_mean_mV: float
    pot_std_mV: float
    growth_events: int
    is_final: bool
    def __init__(self, request_id: _Optional[str] = ..., step: _Optional[int] = ..., fractal_dimension: _Optional[float] = ..., pot_min_mV: _Optional[float] = ..., pot_max_mV: _Optional[float] = ..., pot_mean_mV: _Optional[float] = ..., pot_std_mV: _Optional[float] = ..., growth_events: _Optional[int] = ..., is_final: bool = ...) -> None: ...

class SimulationRequest(_message.Message):
    __slots__ = ("request_id", "seed", "grid_size", "steps", "alpha", "spike_probability", "turing_enabled")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    GRID_SIZE_FIELD_NUMBER: _ClassVar[int]
    STEPS_FIELD_NUMBER: _ClassVar[int]
    ALPHA_FIELD_NUMBER: _ClassVar[int]
    SPIKE_PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    TURING_ENABLED_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    seed: int
    grid_size: int
    steps: int
    alpha: float
    spike_probability: float
    turing_enabled: bool
    def __init__(self, request_id: _Optional[str] = ..., seed: _Optional[int] = ..., grid_size: _Optional[int] = ..., steps: _Optional[int] = ..., alpha: _Optional[float] = ..., spike_probability: _Optional[float] = ..., turing_enabled: bool = ...) -> None: ...

class SimulationResult(_message.Message):
    __slots__ = ("request_id", "meta", "growth_events", "pot_min_mV", "pot_max_mV", "pot_mean_mV", "pot_std_mV", "fractal_dimension")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    META_FIELD_NUMBER: _ClassVar[int]
    GROWTH_EVENTS_FIELD_NUMBER: _ClassVar[int]
    POT_MIN_MV_FIELD_NUMBER: _ClassVar[int]
    POT_MAX_MV_FIELD_NUMBER: _ClassVar[int]
    POT_MEAN_MV_FIELD_NUMBER: _ClassVar[int]
    POT_STD_MV_FIELD_NUMBER: _ClassVar[int]
    FRACTAL_DIMENSION_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    meta: ResponseMeta
    growth_events: int
    pot_min_mV: float
    pot_max_mV: float
    pot_mean_mV: float
    pot_std_mV: float
    fractal_dimension: float
    def __init__(self, request_id: _Optional[str] = ..., meta: _Optional[_Union[ResponseMeta, _Mapping]] = ..., growth_events: _Optional[int] = ..., pot_min_mV: _Optional[float] = ..., pot_max_mV: _Optional[float] = ..., pot_mean_mV: _Optional[float] = ..., pot_std_mV: _Optional[float] = ..., fractal_dimension: _Optional[float] = ...) -> None: ...

class SimulationStreamRequest(_message.Message):
    __slots__ = ("request_id", "seed", "grid_size", "total_steps", "steps_per_frame", "alpha", "spike_probability", "turing_enabled")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    GRID_SIZE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_STEPS_FIELD_NUMBER: _ClassVar[int]
    STEPS_PER_FRAME_FIELD_NUMBER: _ClassVar[int]
    ALPHA_FIELD_NUMBER: _ClassVar[int]
    SPIKE_PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    TURING_ENABLED_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    seed: int
    grid_size: int
    total_steps: int
    steps_per_frame: int
    alpha: float
    spike_probability: float
    turing_enabled: bool
    def __init__(self, request_id: _Optional[str] = ..., seed: _Optional[int] = ..., grid_size: _Optional[int] = ..., total_steps: _Optional[int] = ..., steps_per_frame: _Optional[int] = ..., alpha: _Optional[float] = ..., spike_probability: _Optional[float] = ..., turing_enabled: bool = ...) -> None: ...

class SimulationFrame(_message.Message):
    __slots__ = ("request_id", "step", "growth_events", "pot_min_mV", "pot_max_mV", "pot_mean_mV", "pot_std_mV", "is_final")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    GROWTH_EVENTS_FIELD_NUMBER: _ClassVar[int]
    POT_MIN_MV_FIELD_NUMBER: _ClassVar[int]
    POT_MAX_MV_FIELD_NUMBER: _ClassVar[int]
    POT_MEAN_MV_FIELD_NUMBER: _ClassVar[int]
    POT_STD_MV_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    step: int
    growth_events: int
    pot_min_mV: float
    pot_max_mV: float
    pot_mean_mV: float
    pot_std_mV: float
    is_final: bool
    def __init__(self, request_id: _Optional[str] = ..., step: _Optional[int] = ..., growth_events: _Optional[int] = ..., pot_min_mV: _Optional[float] = ..., pot_max_mV: _Optional[float] = ..., pot_mean_mV: _Optional[float] = ..., pot_std_mV: _Optional[float] = ..., is_final: bool = ...) -> None: ...

class ValidationRequest(_message.Message):
    __slots__ = ("request_id", "seed", "epochs", "batch_size", "grid_size", "steps", "turing_enabled", "quantum_jitter")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    EPOCHS_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    GRID_SIZE_FIELD_NUMBER: _ClassVar[int]
    STEPS_FIELD_NUMBER: _ClassVar[int]
    TURING_ENABLED_FIELD_NUMBER: _ClassVar[int]
    QUANTUM_JITTER_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    seed: int
    epochs: int
    batch_size: int
    grid_size: int
    steps: int
    turing_enabled: bool
    quantum_jitter: bool
    def __init__(self, request_id: _Optional[str] = ..., seed: _Optional[int] = ..., epochs: _Optional[int] = ..., batch_size: _Optional[int] = ..., grid_size: _Optional[int] = ..., steps: _Optional[int] = ..., turing_enabled: bool = ..., quantum_jitter: bool = ...) -> None: ...

class ValidationResult(_message.Message):
    __slots__ = ("request_id", "meta", "loss_start", "loss_final", "loss_drop", "pot_min_mV", "pot_max_mV", "example_fractal_dim", "lyapunov_exponent", "growth_events", "nernst_symbolic_mV", "nernst_numeric_mV")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    META_FIELD_NUMBER: _ClassVar[int]
    LOSS_START_FIELD_NUMBER: _ClassVar[int]
    LOSS_FINAL_FIELD_NUMBER: _ClassVar[int]
    LOSS_DROP_FIELD_NUMBER: _ClassVar[int]
    POT_MIN_MV_FIELD_NUMBER: _ClassVar[int]
    POT_MAX_MV_FIELD_NUMBER: _ClassVar[int]
    EXAMPLE_FRACTAL_DIM_FIELD_NUMBER: _ClassVar[int]
    LYAPUNOV_EXPONENT_FIELD_NUMBER: _ClassVar[int]
    GROWTH_EVENTS_FIELD_NUMBER: _ClassVar[int]
    NERNST_SYMBOLIC_MV_FIELD_NUMBER: _ClassVar[int]
    NERNST_NUMERIC_MV_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    meta: ResponseMeta
    loss_start: float
    loss_final: float
    loss_drop: float
    pot_min_mV: float
    pot_max_mV: float
    example_fractal_dim: float
    lyapunov_exponent: float
    growth_events: float
    nernst_symbolic_mV: float
    nernst_numeric_mV: float
    def __init__(self, request_id: _Optional[str] = ..., meta: _Optional[_Union[ResponseMeta, _Mapping]] = ..., loss_start: _Optional[float] = ..., loss_final: _Optional[float] = ..., loss_drop: _Optional[float] = ..., pot_min_mV: _Optional[float] = ..., pot_max_mV: _Optional[float] = ..., example_fractal_dim: _Optional[float] = ..., lyapunov_exponent: _Optional[float] = ..., growth_events: _Optional[float] = ..., nernst_symbolic_mV: _Optional[float] = ..., nernst_numeric_mV: _Optional[float] = ...) -> None: ...
