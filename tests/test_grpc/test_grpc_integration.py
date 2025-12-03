"""
Integration tests for MFN gRPC layer.

Tests the full gRPC stack: server, client, and interceptors working together.
"""

import asyncio
import os
import pytest
import pytest_asyncio

from mycelium_fractal_net.grpc import MFNClient, serve


@pytest_asyncio.fixture
async def grpc_server():
    """Start a test gRPC server."""
    # Start server on a test port
    port = 50052  # Use different port to avoid conflicts
    server = await serve(
        port=port,
        enable_auth=False,  # Disable auth for basic integration tests
        enable_audit=False,
        enable_rate_limit=False,
    )
    
    yield f"localhost:{port}"
    
    # Cleanup
    await server.stop(grace=1.0)


@pytest.mark.asyncio
async def test_integration_simulation(grpc_server):
    """Test full simulation flow through gRPC."""
    async with MFNClient(grpc_server) as client:
        result = await client.run_simulation(
            seed=42,
            grid_size=32,
            steps=16,
            alpha=0.18,
            spike_probability=0.25,
            turing_enabled=True,
        )
        
        assert result.request_id is not None
        assert result.growth_events >= 0
        assert result.fractal_dimension > 0
        assert result.pot_min_mV < result.pot_max_mV
        assert result.meta.meta["status"] == "ok"


@pytest.mark.asyncio
async def test_integration_features(grpc_server):
    """Test feature extraction through gRPC."""
    async with MFNClient(grpc_server) as client:
        result = await client.extract_features(
            seed=42,
            grid_size=32,
            steps=16,
        )
        
        assert result.request_id is not None
        assert result.fractal_dimension >= 0  # Can be 0 for small simulations
        assert result.lacunarity >= 0
        assert result.hurst_exponent >= 0
        assert result.active_nodes >= 0
        assert result.meta.meta["status"] == "ok"


@pytest.mark.asyncio
async def test_integration_validation(grpc_server):
    """Test validation through gRPC."""
    async with MFNClient(grpc_server) as client:
        result = await client.validate_pattern(
            seed=42,
            epochs=1,
            batch_size=2,
            grid_size=32,
            steps=16,
            turing_enabled=True,
            quantum_jitter=False,
        )
        
        assert result.request_id is not None
        assert result.loss_start >= 0
        assert result.loss_final >= 0
        assert result.loss_drop >= 0
        assert result.example_fractal_dim >= 0  # Can be 0 for small simulations
        assert result.meta.meta["status"] == "ok"


@pytest.mark.asyncio
async def test_integration_stream_simulation(grpc_server):
    """Test simulation streaming through gRPC."""
    async with MFNClient(grpc_server) as client:
        frames = []
        
        async for frame in client.stream_simulation(
            seed=42,
            grid_size=32,
            steps=20,
            stream_interval=10,
        ):
            frames.append(frame)
            if len(frames) >= 2:  # Limit frames for test speed
                break
        
        assert len(frames) > 0
        for frame in frames:
            assert frame.request_id is not None
            assert frame.step >= 0
            assert frame.growth_events >= 0


@pytest.mark.asyncio
async def test_integration_stream_features(grpc_server):
    """Test feature streaming through gRPC."""
    async with MFNClient(grpc_server) as client:
        frames = []
        
        async for frame in client.stream_features(
            seed=42,
            grid_size=32,
            steps=20,
            stream_interval=10,
        ):
            frames.append(frame)
            if len(frames) >= 2:  # Limit frames for test speed
                break
        
        assert len(frames) > 0
        for frame in frames:
            assert frame.request_id is not None
            assert frame.step >= 0
            assert frame.fractal_dimension >= 0  # Can be 0 for small simulations


@pytest.mark.asyncio
async def test_integration_multiple_concurrent_requests(grpc_server):
    """Test handling multiple concurrent requests."""
    async with MFNClient(grpc_server) as client:
        # Send 5 concurrent simulation requests
        tasks = [
            client.run_simulation(seed=i, grid_size=16, steps=8)
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        for result in results:
            assert result.growth_events >= 0
            assert result.fractal_dimension > 0


@pytest.mark.asyncio
async def test_integration_custom_request_id(grpc_server):
    """Test using custom request IDs."""
    async with MFNClient(grpc_server) as client:
        custom_id = "my-custom-request-123"
        
        result = await client.run_simulation(
            seed=42,
            grid_size=16,
            steps=8,
            request_id=custom_id,
        )
        
        assert result.request_id == custom_id


@pytest_asyncio.fixture
async def grpc_server_with_auth():
    """Start a test gRPC server with authentication enabled."""
    # Set API key in environment
    os.environ["MFN_API_KEYS"] = "test-key-123,test-key-456"
    
    port = 50053  # Different port
    server = await serve(
        port=port,
        enable_auth=True,
        enable_audit=True,
        enable_rate_limit=False,
    )
    
    yield f"localhost:{port}"
    
    # Cleanup
    await server.stop(grace=1.0)
    del os.environ["MFN_API_KEYS"]


@pytest.mark.asyncio
async def test_integration_with_authentication(grpc_server_with_auth):
    """Test gRPC with authentication."""
    async with MFNClient(grpc_server_with_auth, api_key="test-key-123") as client:
        result = await client.run_simulation(
            seed=42,
            grid_size=16,
            steps=8,
        )
        
        assert result.growth_events >= 0


@pytest.mark.asyncio
async def test_integration_determinism(grpc_server):
    """Test that same seed produces same results."""
    async with MFNClient(grpc_server) as client:
        result1 = await client.run_simulation(seed=42, grid_size=32, steps=16)
        result2 = await client.run_simulation(seed=42, grid_size=32, steps=16)
        
        # Same seed should produce same results
        assert result1.growth_events == result2.growth_events
        assert abs(result1.fractal_dimension - result2.fractal_dimension) < 0.01


@pytest.mark.asyncio
async def test_integration_different_seeds(grpc_server):
    """Test that different seeds produce different results."""
    async with MFNClient(grpc_server) as client:
        result1 = await client.run_simulation(seed=1, grid_size=32, steps=16)
        result2 = await client.run_simulation(seed=999, grid_size=32, steps=16)
        
        # Different seeds should produce different results
        assert result1.growth_events != result2.growth_events or \
               abs(result1.fractal_dimension - result2.fractal_dimension) > 0.01
