"""
Integration tests for gRPC API.

Tests end-to-end functionality with real server and client.
"""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from mycelium_fractal_net.grpc.client import MFNClient
from mycelium_fractal_net.grpc.config import GRPCConfig
from mycelium_fractal_net.grpc.server import serve


@pytest_asyncio.fixture
async def grpc_server():
    """Start test gRPC server."""
    # Use high port to avoid conflicts
    config = GRPCConfig(
        port=50052,
        max_workers=2,
        max_message_size_mb=10,
        max_concurrent_streams=10,
        keepalive_time_ms=30000,
        keepalive_timeout_ms=10000,
        tls_enabled=False,
        tls_cert_path=None,
        tls_key_path=None,
        auth_enabled=False,  # Disable auth for integration tests
        rate_limit_rps=1000,
        rate_limit_concurrent=50,
    )
    
    # Start server without interceptors for simpler testing
    server = await serve(config, interceptors_enabled=False)
    
    # Give server time to start
    await asyncio.sleep(0.5)
    
    yield server
    
    # Cleanup
    await server.stop(grace=1.0)


@pytest.mark.asyncio
async def test_extract_features_integration(grpc_server):
    """Test feature extraction end-to-end."""
    async with MFNClient("localhost:50052", api_key="test-key") as client:
        response = await client.extract_features(
            seed=42,
            grid_size=32,
            steps=10,
        )
        
        assert response.request_id is not None
        assert response.fractal_dimension >= 0
        assert response.growth_events >= 0
        assert response.pot_min_mV < response.pot_max_mV
        assert response.meta.meta["server"] == "mfn-grpc"


@pytest.mark.asyncio
async def test_stream_features_integration(grpc_server):
    """Test feature streaming end-to-end."""
    async with MFNClient("localhost:50052", api_key="test-key") as client:
        frames = []
        async for frame in client.stream_features(
            seed=42,
            grid_size=32,
            total_steps=30,
            steps_per_frame=10,
        ):
            frames.append(frame)
        
        # Should receive multiple frames
        assert len(frames) >= 2
        
        # Last frame should be marked final
        assert frames[-1].is_final is True
        
        # Frames should have increasing steps
        for i in range(len(frames) - 1):
            assert frames[i].step < frames[i + 1].step


@pytest.mark.asyncio
async def test_run_simulation_integration(grpc_server):
    """Test simulation end-to-end."""
    async with MFNClient("localhost:50052", api_key="test-key") as client:
        response = await client.run_simulation(
            seed=42,
            grid_size=32,
            steps=10,
        )
        
        assert response.request_id is not None
        assert response.fractal_dimension >= 0
        assert response.growth_events >= 0


@pytest.mark.asyncio
async def test_stream_simulation_integration(grpc_server):
    """Test simulation streaming end-to-end."""
    async with MFNClient("localhost:50052", api_key="test-key") as client:
        frames = []
        async for frame in client.stream_simulation(
            seed=42,
            grid_size=32,
            total_steps=30,
            steps_per_frame=10,
        ):
            frames.append(frame)
        
        assert len(frames) >= 2
        assert frames[-1].is_final is True


@pytest.mark.asyncio
async def test_validate_pattern_integration(grpc_server):
    """Test validation end-to-end."""
    async with MFNClient("localhost:50052", api_key="test-key") as client:
        response = await client.validate_pattern(
            seed=42,
            epochs=1,
            batch_size=2,
            grid_size=32,
            steps=10,
        )
        
        assert response.request_id is not None
        assert response.loss_start >= 0
        assert response.loss_final >= 0
        assert response.loss_drop >= 0
        assert response.example_fractal_dim >= 0


@pytest.mark.asyncio
async def test_multiple_concurrent_requests(grpc_server):
    """Test multiple concurrent requests."""
    async with MFNClient("localhost:50052", api_key="test-key") as client:
        # Run multiple requests concurrently
        tasks = [
            client.extract_features(seed=i, grid_size=32, steps=10)
            for i in range(5)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(responses) == 5
        
        # All should have valid data
        for response in responses:
            assert response.fractal_dimension >= 0


@pytest.mark.asyncio
async def test_different_parameters(grpc_server):
    """Test with different parameter combinations."""
    async with MFNClient("localhost:50052", api_key="test-key") as client:
        # Test different grid sizes
        for grid_size in [16, 32, 64]:
            response = await client.extract_features(
                seed=42,
                grid_size=grid_size,
                steps=10,
            )
            assert response.fractal_dimension >= 0
        
        # Test with Turing disabled
        response = await client.extract_features(
            seed=42,
            grid_size=32,
            steps=10,
            turing_enabled=False,
        )
        assert response.fractal_dimension >= 0


@pytest.mark.asyncio
async def test_streaming_cancellation(grpc_server):
    """Test that streaming can be cancelled."""
    async with MFNClient("localhost:50052", api_key="test-key") as client:
        frames = []
        
        async for frame in client.stream_features(
            seed=42,
            grid_size=32,
            total_steps=100,
            steps_per_frame=10,
        ):
            frames.append(frame)
            # Cancel after first frame
            if len(frames) >= 1:
                break
        
        # Should have at least one frame
        assert len(frames) >= 1
        
        # Should not have all frames (was cancelled)
        assert len(frames) < 10
