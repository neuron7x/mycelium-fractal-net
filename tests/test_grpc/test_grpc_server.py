"""
Unit tests for MFN gRPC server.

Tests the servicers and server configuration without running a full server.
"""

import pytest
import grpc.aio

from mycelium_fractal_net.grpc import mfn_pb2, mfn_pb2_grpc
from mycelium_fractal_net.grpc.server import (
    MFNFeaturesServiceServicer,
    MFNSimulationServiceServicer,
    MFNValidationServiceServicer,
)


@pytest.fixture
def features_servicer():
    """Create features servicer instance."""
    return MFNFeaturesServiceServicer()


@pytest.fixture
def simulation_servicer():
    """Create simulation servicer instance."""
    return MFNSimulationServiceServicer()


@pytest.fixture
def validation_servicer():
    """Create validation servicer instance."""
    return MFNValidationServiceServicer()


class MockContext:
    """Mock gRPC context for testing."""
    
    def __init__(self):
        self._cancelled = False
    
    def cancelled(self):
        return self._cancelled
    
    async def abort(self, code, details):
        raise Exception(f"{code}: {details}")


@pytest.mark.asyncio
async def test_extract_features(features_servicer):
    """Test ExtractFeatures RPC."""
    request = mfn_pb2.FeatureRequest(
        request_id="test-123",
        seed=42,
        grid_size=32,
        steps=16,
        alpha=0.18,
        spike_probability=0.25,
        turing_enabled=True,
    )
    
    context = MockContext()
    response = await features_servicer.ExtractFeatures(request, context)
    
    assert response.request_id == "test-123"
    assert response.fractal_dimension > 0
    assert response.lacunarity >= 0
    assert response.hurst_exponent >= 0
    assert response.active_nodes >= 0
    assert response.meta.meta["status"] == "ok"


@pytest.mark.asyncio
async def test_stream_features(features_servicer):
    """Test StreamFeatures RPC."""
    request = mfn_pb2.FeatureStreamRequest(
        request_id="test-456",
        seed=42,
        grid_size=32,
        steps=20,
        stream_interval=10,
    )
    
    context = MockContext()
    frames = []
    
    async for frame in features_servicer.StreamFeatures(request, context):
        frames.append(frame)
    
    assert len(frames) > 0
    assert all(frame.request_id == "test-456" for frame in frames)
    assert all(frame.fractal_dimension > 0 for frame in frames)


@pytest.mark.asyncio
async def test_run_simulation(simulation_servicer):
    """Test RunSimulation RPC."""
    request = mfn_pb2.SimulationRequest(
        request_id="test-789",
        seed=42,
        grid_size=32,
        steps=16,
        alpha=0.18,
        spike_probability=0.25,
        turing_enabled=True,
    )
    
    context = MockContext()
    response = await simulation_servicer.RunSimulation(request, context)
    
    assert response.request_id == "test-789"
    assert response.growth_events >= 0
    assert response.fractal_dimension > 0
    assert response.pot_min_mV < response.pot_max_mV
    assert response.meta.meta["status"] == "ok"


@pytest.mark.asyncio
async def test_stream_simulation(simulation_servicer):
    """Test StreamSimulation RPC."""
    request = mfn_pb2.SimulationStreamRequest(
        request_id="test-abc",
        seed=42,
        grid_size=32,
        steps=20,
        stream_interval=10,
    )
    
    context = MockContext()
    frames = []
    
    async for frame in simulation_servicer.StreamSimulation(request, context):
        frames.append(frame)
    
    assert len(frames) > 0
    assert all(frame.request_id == "test-abc" for frame in frames)
    assert all(frame.growth_events >= 0 for frame in frames)


@pytest.mark.asyncio
async def test_validate_pattern(validation_servicer):
    """Test ValidatePattern RPC."""
    request = mfn_pb2.ValidationRequest(
        request_id="test-def",
        seed=42,
        epochs=1,
        batch_size=2,
        grid_size=32,
        steps=16,
        turing_enabled=True,
        quantum_jitter=False,
    )
    
    context = MockContext()
    response = await validation_servicer.ValidatePattern(request, context)
    
    assert response.request_id == "test-def"
    assert response.loss_start >= 0
    assert response.loss_final >= 0
    assert response.loss_drop >= 0
    assert response.example_fractal_dim > 0
    assert response.meta.meta["status"] == "ok"


@pytest.mark.asyncio
async def test_extract_features_invalid_params():
    """Test ExtractFeatures with edge case parameters."""
    servicer = MFNFeaturesServiceServicer()
    
    request = mfn_pb2.FeatureRequest(
        request_id="test-edge",
        seed=1,
        grid_size=16,  # Small grid
        steps=4,  # Few steps
        alpha=0.1,
        spike_probability=0.1,
        turing_enabled=False,
    )
    
    context = MockContext()
    response = await servicer.ExtractFeatures(request, context)
    
    # Should still succeed with edge case params
    assert response.request_id == "test-edge"
    assert response.fractal_dimension >= 0


@pytest.mark.asyncio
async def test_validation_minimal_config():
    """Test validation with minimal configuration."""
    servicer = MFNValidationServiceServicer()
    
    request = mfn_pb2.ValidationRequest(
        request_id="test-minimal",
        seed=1,
        epochs=1,
        batch_size=1,
        grid_size=16,
        steps=8,
        turing_enabled=False,
        quantum_jitter=False,
    )
    
    context = MockContext()
    response = await servicer.ValidatePattern(request, context)
    
    assert response.request_id == "test-minimal"
    assert response.loss_start >= 0
    assert response.loss_final >= 0
