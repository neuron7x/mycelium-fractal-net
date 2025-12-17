import pytest

from mycelium_fractal_net.integration.service_context import ExecutionMode, ServiceContext
from mycelium_fractal_net.integration.ws_adapters import stream_simulation_live_adapter
from mycelium_fractal_net.integration.ws_schemas import SimulationLiveParams


class _DummyRng:
    def __init__(self, values: list[float]):
        self._values = values
        self.calls: int = 0

    def random(self) -> float:
        value = self._values[min(self.calls, len(self._values) - 1)]
        self.calls += 1
        return value


class _ContextSpy(ServiceContext):
    def __init__(self, *, rng: _DummyRng, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._rng = rng
        self.with_seed_called = False
        self.with_seed_args: list[int] = []

    def with_seed(self, seed: int) -> "_ContextSpy":
        self.with_seed_called = True
        self.with_seed_args.append(seed)
        return _ContextSpy(
            seed=seed,
            mode=self.mode,
            grid_size=self.grid_size,
            steps=self.steps,
            turing_enabled=self.turing_enabled,
            quantum_jitter=self.quantum_jitter,
            rng=self._rng,
        )


def test_stream_simulation_live_adapter_preserves_context_on_seed_override() -> None:
    async def _run() -> None:
        params = SimulationLiveParams(
            seed=7,
            grid_size=8,
            steps=1,
            alpha=0.05,
            spike_probability=0.5,
            turing_enabled=False,
            update_interval_steps=1,
            include_full_state=False,
        )

        original_rng = _DummyRng([0.0])
        replacement_rng = _DummyRng([0.75])

        ctx = _ContextSpy(
            seed=3,
            mode=ExecutionMode.API,
            grid_size=16,
            steps=4,
            turing_enabled=True,
            rng=original_rng,
        )

        # Replace the RNG in the cloned context with one that would prevent growth events.
        def _with_seed_override(seed: int) -> _ContextSpy:
            ctx.with_seed_called = True
            ctx.with_seed_args.append(seed)
            clone = _ContextSpy(
                seed=seed,
                mode=ctx.mode,
                grid_size=ctx.grid_size,
                steps=ctx.steps,
                turing_enabled=ctx.turing_enabled,
                rng=replacement_rng,
            )
            return clone

        ctx.with_seed = _with_seed_override  # type: ignore[assignment]

        stream = stream_simulation_live_adapter("stream-1", params, ctx)
        first_update = await anext(stream)

        assert ctx.with_seed_called is True
        assert ctx.with_seed_args == [params.seed]
        assert first_update.metrics["growth_events"] == 0

        await stream.aclose()

    import asyncio

    asyncio.run(_run())
