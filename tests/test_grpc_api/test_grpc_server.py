"""
Unit tests for gRPC server servicers.

Tests individual gRPC service methods without full server setup.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import grpc
from mycelium_fractal_net.grpc import mfn_pb2
from mycelium_fractal_net.grpc.server import (
    MFNFeaturesServiceServicer,
    MFNSimulationServiceServicer,
    MFNValidationServiceServicer,
)


@pytest.fixture
def mock_context():
    """Create mock gRPC context."""
    context = MagicMock(spec=grpc.aio.ServicerContext)
    context.cancelled = MagicMock(return_value=False)
    context.abort = AsyncMock()
    return context


class TestMFNFeaturesServiceServicer:
    """Tests for feature extraction servicer."""
    
    @pytest.mark.asyncio
    async def test_extract_features_success(self, mock_context):
        """Test successful feature extraction."""
        servicer = MFNFeaturesServiceServicer()
        
        request = mfn_pb2.FeatureRequest(
            request_id="test-123",
            seed=42,
            grid_size=32,
            steps=10,
            alpha=0.18,
            spike_probability=0.25,
            turing_enabled=True,
        )
        
        response = await servicer.ExtractFeatures(request, mock_context)
        
        assert response.request_id == "test-123"
        assert response.fractal_dimension >= 0  # Can be 0 for small grids
        assert response.growth_events >= 0
        assert response.pot_min_mV < response.pot_max_mV
        assert response.meta.meta["server"] == "mfn-grpc"
    
    @pytest.mark.asyncio
    async def test_extract_features_different_params(self, mock_context):
        """Test feature extraction with different parameters."""
        servicer = MFNFeaturesServiceServicer()
        
        request = mfn_pb2.FeatureRequest(
            request_id="test-456",
            seed=100,
            grid_size=64,
            steps=50,
            alpha=0.2,
            spike_probability=0.3,
            turing_enabled=False,
        )
        
        response = await servicer.ExtractFeatures(request, mock_context)
        
        assert response.request_id == "test-456"
        assert 0 <= response.fractal_dimension < 3
    
    @pytest.mark.asyncio
    async def test_stream_features(self, mock_context):
        """Test feature streaming."""
        servicer = MFNFeaturesServiceServicer()
        
        request = mfn_pb2.FeatureStreamRequest(
            request_id="stream-123",
            seed=42,
            grid_size=32,
            total_steps=30,
            steps_per_frame=10,
            alpha=0.18,
            spike_probability=0.25,
            turing_enabled=True,
        )
        
        frames = []
        async for frame in servicer.StreamFeatures(request, mock_context):
            frames.append(frame)
        
        # Should have 3 frames (30 steps / 10 per frame)
        assert len(frames) == 3
        assert frames[0].step == 10
        assert frames[1].step == 20
        assert frames[2].step == 30
        assert frames[2].is_final is True
        
        # All frames should have valid data
        for frame in frames:
            assert frame.request_id == "stream-123"
            assert frame.fractal_dimension >= 0


class TestMFNSimulationServiceServicer:
    """Tests for simulation servicer."""
    
    @pytest.mark.asyncio
    async def test_run_simulation_success(self, mock_context):
        """Test successful simulation."""
        servicer = MFNSimulationServiceServicer()
        
        request = mfn_pb2.SimulationRequest(
            request_id="sim-123",
            seed=42,
            grid_size=32,
            steps=10,
            alpha=0.18,
            spike_probability=0.25,
            turing_enabled=True,
        )
        
        response = await servicer.RunSimulation(request, mock_context)
        
        assert response.request_id == "sim-123"
        assert response.fractal_dimension >= 0
        assert response.growth_events >= 0
        assert response.pot_min_mV < response.pot_max_mV
    
    @pytest.mark.asyncio
    async def test_stream_simulation(self, mock_context):
        """Test simulation streaming."""
        servicer = MFNSimulationServiceServicer()
        
        request = mfn_pb2.SimulationStreamRequest(
            request_id="stream-sim-123",
            seed=42,
            grid_size=32,
            total_steps=30,
            steps_per_frame=10,
            alpha=0.18,
            spike_probability=0.25,
            turing_enabled=True,
        )
        
        frames = []
        async for frame in servicer.StreamSimulation(request, mock_context):
            frames.append(frame)
        
        assert len(frames) == 3
        assert frames[2].is_final is True
        
        # Growth events should accumulate
        assert frames[0].growth_events <= frames[1].growth_events
        assert frames[1].growth_events <= frames[2].growth_events


class TestMFNValidationServiceServicer:
    """Tests for validation servicer."""
    
    @pytest.mark.asyncio
    async def test_validate_pattern_success(self, mock_context):
        """Test successful pattern validation."""
        servicer = MFNValidationServiceServicer()
        
        request = mfn_pb2.ValidationRequest(
            request_id="val-123",
            seed=42,
            epochs=1,
            batch_size=2,
            grid_size=32,
            steps=10,
            turing_enabled=True,
            quantum_jitter=False,
        )
        
        response = await servicer.ValidatePattern(request, mock_context)
        
        assert response.request_id == "val-123"
        assert response.loss_start >= response.loss_final
        assert response.loss_drop >= 0
        assert response.example_fractal_dim >= 0
    
    @pytest.mark.asyncio
    async def test_validate_pattern_with_quantum_jitter(self, mock_context):
        """Test validation with quantum jitter enabled."""
        servicer = MFNValidationServiceServicer()
        
        request = mfn_pb2.ValidationRequest(
            request_id="val-456",
            seed=42,
            epochs=1,
            batch_size=2,
            grid_size=32,
            steps=10,
            turing_enabled=True,
            quantum_jitter=True,
        )
        
        response = await servicer.ValidatePattern(request, mock_context)
        
        assert response.request_id == "val-456"
        assert response.loss_drop >= 0
