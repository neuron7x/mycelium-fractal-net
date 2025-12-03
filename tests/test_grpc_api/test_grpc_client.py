"""
Unit tests for gRPC client SDK.

Tests client methods and functionality with mock server responses.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest

from mycelium_fractal_net.grpc import mfn_pb2
from mycelium_fractal_net.grpc.client import MFNClient


@pytest.fixture
def mock_channel():
    """Create mock gRPC channel."""
    channel = MagicMock()
    channel.close = AsyncMock()
    return channel


@pytest.fixture
def mock_features_stub():
    """Create mock features stub."""
    stub = MagicMock()
    stub.ExtractFeatures = AsyncMock()
    stub.StreamFeatures = MagicMock()
    return stub


@pytest.fixture
def mock_simulation_stub():
    """Create mock simulation stub."""
    stub = MagicMock()
    stub.RunSimulation = AsyncMock()
    stub.StreamSimulation = MagicMock()
    return stub


@pytest.fixture
def mock_validation_stub():
    """Create mock validation stub."""
    stub = MagicMock()
    stub.ValidatePattern = AsyncMock()
    return stub


class TestMFNClient:
    """Tests for MFN gRPC client."""
    
    @pytest.mark.asyncio
    async def test_client_context_manager(self):
        """Test client as async context manager."""
        with patch("grpc.aio.insecure_channel") as mock_insecure:
            mock_insecure.return_value = MagicMock()
            mock_insecure.return_value.close = AsyncMock()
            
            async with MFNClient("localhost:50051", api_key="test-key") as client:
                assert client._channel is not None
            
            # Should close on exit
            mock_insecure.return_value.close.assert_called_once()
    
    def test_generate_request_id(self):
        """Test request ID generation."""
        client = MFNClient("localhost:50051", api_key="test-key")
        
        req_id1 = client._generate_request_id()
        req_id2 = client._generate_request_id()
        
        assert req_id1 != req_id2
        assert len(req_id1) > 0
    
    def test_generate_signature(self):
        """Test HMAC signature generation."""
        client = MFNClient("localhost:50051", api_key="test-key")
        
        sig1 = client._generate_signature("req-1", "1234567890")
        sig2 = client._generate_signature("req-1", "1234567890")
        sig3 = client._generate_signature("req-2", "1234567890")
        
        # Same input should produce same signature
        assert sig1 == sig2
        
        # Different request ID should produce different signature
        assert sig1 != sig3
        
        # Signature should be hex string
        assert len(sig1) == 64  # SHA256 hex = 64 chars
    
    def test_build_metadata(self):
        """Test metadata building."""
        client = MFNClient("localhost:50051", api_key="test-key")
        
        metadata = client._build_metadata("req-123")
        metadata_dict = dict(metadata)
        
        assert metadata_dict["x-api-key"] == "test-key"
        assert metadata_dict["x-request-id"] == "req-123"
        assert "x-timestamp" in metadata_dict
        assert "x-signature" in metadata_dict
    
    @pytest.mark.asyncio
    async def test_extract_features(self, mock_channel, mock_features_stub):
        """Test extract features method."""
        with patch("grpc.aio.insecure_channel", return_value=mock_channel):
            with patch(
                "mycelium_fractal_net.grpc.mfn_pb2_grpc.MFNFeaturesServiceStub",
                return_value=mock_features_stub,
            ):
                # Mock response
                mock_response = mfn_pb2.FeatureResponse(
                    request_id="test-123",
                    fractal_dimension=1.5,
                    pot_min_mV=-80.0,
                    pot_max_mV=-50.0,
                    pot_mean_mV=-65.0,
                    pot_std_mV=5.0,
                    growth_events=10,
                )
                mock_features_stub.ExtractFeatures.return_value = mock_response
                
                async with MFNClient("localhost:50051", api_key="test-key") as client:
                    response = await client.extract_features(
                        seed=42,
                        grid_size=64,
                        steps=64,
                    )
                
                assert response.fractal_dimension == 1.5
                assert response.growth_events == 10
                mock_features_stub.ExtractFeatures.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_simulation(self, mock_channel, mock_simulation_stub):
        """Test run simulation method."""
        with patch("grpc.aio.insecure_channel", return_value=mock_channel):
            with patch(
                "mycelium_fractal_net.grpc.mfn_pb2_grpc.MFNSimulationServiceStub",
                return_value=mock_simulation_stub,
            ):
                # Mock response
                mock_response = mfn_pb2.SimulationResult(
                    request_id="sim-123",
                    fractal_dimension=1.6,
                    growth_events=15,
                )
                mock_simulation_stub.RunSimulation.return_value = mock_response
                
                async with MFNClient("localhost:50051", api_key="test-key") as client:
                    response = await client.run_simulation(
                        seed=42,
                        grid_size=64,
                        steps=64,
                    )
                
                assert response.fractal_dimension == 1.6
                assert response.growth_events == 15
    
    @pytest.mark.asyncio
    async def test_validate_pattern(self, mock_channel, mock_validation_stub):
        """Test validate pattern method."""
        with patch("grpc.aio.insecure_channel", return_value=mock_channel):
            with patch(
                "mycelium_fractal_net.grpc.mfn_pb2_grpc.MFNValidationServiceStub",
                return_value=mock_validation_stub,
            ):
                # Mock response
                mock_response = mfn_pb2.ValidationResult(
                    request_id="val-123",
                    loss_start=2.0,
                    loss_final=0.5,
                    loss_drop=1.5,
                    example_fractal_dim=1.7,
                )
                mock_validation_stub.ValidatePattern.return_value = mock_response
                
                async with MFNClient("localhost:50051", api_key="test-key") as client:
                    response = await client.validate_pattern(
                        seed=42,
                        epochs=1,
                        batch_size=4,
                    )
                
                assert response.loss_start == 2.0
                assert response.loss_final == 0.5
                assert response.loss_drop == 1.5
    
    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self, mock_channel, mock_features_stub):
        """Test retry logic on transient errors."""
        with patch("grpc.aio.insecure_channel", return_value=mock_channel):
            with patch(
                "mycelium_fractal_net.grpc.mfn_pb2_grpc.MFNFeaturesServiceStub",
                return_value=mock_features_stub,
            ):
                # First call fails, second succeeds
                mock_response = mfn_pb2.FeatureResponse(request_id="test-123")
                
                error = grpc.RpcError()
                error.code = lambda: grpc.StatusCode.UNAVAILABLE
                
                mock_features_stub.ExtractFeatures.side_effect = [
                    error,
                    mock_response,
                ]
                
                async with MFNClient(
                    "localhost:50051",
                    api_key="test-key",
                    retry_max_attempts=2,
                    retry_backoff_ms=10,
                ) as client:
                    response = await client.extract_features()
                
                # Should succeed after retry
                assert response.request_id == "test-123"
                assert mock_features_stub.ExtractFeatures.call_count == 2
    
    @pytest.mark.asyncio
    async def test_no_retry_on_permanent_error(self, mock_channel, mock_features_stub):
        """Test no retry on permanent errors."""
        with patch("grpc.aio.insecure_channel", return_value=mock_channel):
            with patch(
                "mycelium_fractal_net.grpc.mfn_pb2_grpc.MFNFeaturesServiceStub",
                return_value=mock_features_stub,
            ):
                # Permanent error (INVALID_ARGUMENT)
                error = grpc.RpcError()
                error.code = lambda: grpc.StatusCode.INVALID_ARGUMENT
                
                mock_features_stub.ExtractFeatures.side_effect = error
                
                async with MFNClient(
                    "localhost:50051",
                    api_key="test-key",
                    retry_max_attempts=3,
                ) as client:
                    with pytest.raises(grpc.RpcError):
                        await client.extract_features()
                
                # Should only call once (no retry)
                assert mock_features_stub.ExtractFeatures.call_count == 1
