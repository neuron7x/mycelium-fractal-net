"""
Unit tests for MFN gRPC client.

Tests the client SDK functionality with a mock server.
"""

import asyncio
import pytest

from mycelium_fractal_net.grpc import MFNClient, mfn_pb2


def test_client_initialization():
    """Test client initialization."""
    client = MFNClient("localhost:50051", api_key="test-key")
    
    assert client.address == "localhost:50051"
    assert client.api_key == "test-key"
    assert client.max_retries == 3
    assert client.retry_backoff_sec == 1.0


def test_request_id_generation():
    """Test request ID generation."""
    client = MFNClient("localhost:50051")
    
    req_id1 = client._generate_request_id()
    req_id2 = client._generate_request_id()
    
    assert len(req_id1) > 0
    assert len(req_id2) > 0
    assert req_id1 != req_id2  # Should be unique


def test_signature_generation():
    """Test HMAC signature generation."""
    client = MFNClient("localhost:50051", api_key="test-key")
    
    timestamp = "1234567890.123"
    method = "/mfn.MFNSimulationService/RunSimulation"
    
    sig1 = client._sign_request(timestamp, method)
    sig2 = client._sign_request(timestamp, method)
    
    assert len(sig1) == 64  # SHA256 hex = 64 chars
    assert sig1 == sig2  # Same input = same signature


def test_signature_without_api_key():
    """Test signature generation without API key."""
    client = MFNClient("localhost:50051")  # No API key
    
    sig = client._sign_request("1234567890", "/test")
    
    assert sig == ""  # No signature without key


def test_metadata_creation():
    """Test metadata creation with authentication."""
    client = MFNClient("localhost:50051", api_key="test-key")
    
    method = "/mfn.MFNSimulationService/RunSimulation"
    metadata = client._create_metadata(method)
    
    metadata_dict = dict(metadata)
    
    assert "x-timestamp" in metadata_dict
    assert "x-api-key" in metadata_dict
    assert "x-signature" in metadata_dict
    assert metadata_dict["x-api-key"] == "test-key"


def test_metadata_without_api_key():
    """Test metadata creation without API key."""
    client = MFNClient("localhost:50051")  # No API key
    
    method = "/test"
    metadata = client._create_metadata(method)
    
    metadata_dict = dict(metadata)
    
    assert "x-timestamp" in metadata_dict
    assert "x-api-key" not in metadata_dict
    assert "x-signature" not in metadata_dict


@pytest.mark.asyncio
async def test_client_connect_close():
    """Test client connection lifecycle."""
    client = MFNClient("localhost:9999")  # Non-existent server
    
    # Connect should not raise (connection is lazy)
    await client.connect()
    assert client._channel is not None
    
    # Close should work
    await client.close()
    assert client._channel is None


@pytest.mark.asyncio
async def test_client_context_manager():
    """Test client as async context manager."""
    async with MFNClient("localhost:9999") as client:
        assert client._channel is not None
    
    # Should be closed after context
    assert client._channel is None


@pytest.mark.asyncio
async def test_client_not_connected_error():
    """Test error when calling methods before connect."""
    client = MFNClient("localhost:50051")
    
    with pytest.raises(RuntimeError, match="not connected"):
        await client.run_simulation()


def test_client_signature_verification():
    """Test signature can be verified by same logic."""
    import hmac
    import hashlib
    
    api_key = "test-secret-key"
    timestamp = "1234567890.123"
    method = "/mfn.MFNSimulationService/RunSimulation"
    
    # Generate signature
    message = f"{timestamp}{method}".encode("utf-8")
    expected = hmac.new(
        api_key.encode("utf-8"),
        message,
        hashlib.sha256,
    ).hexdigest()
    
    # Client should produce same signature
    client = MFNClient("localhost:50051", api_key=api_key)
    sig = client._sign_request(timestamp, method)
    
    assert sig == expected


def test_client_with_tls():
    """Test client initialization with TLS."""
    client = MFNClient("localhost:50051", use_tls=True)
    
    assert client.use_tls is True


def test_client_custom_timeout():
    """Test client with custom timeout."""
    client = MFNClient("localhost:50051", timeout_sec=60.0)
    
    assert client.timeout_sec == 60.0


def test_client_custom_retries():
    """Test client with custom retry settings."""
    client = MFNClient(
        "localhost:50051",
        max_retries=5,
        retry_backoff_sec=2.0,
    )
    
    assert client.max_retries == 5
    assert client.retry_backoff_sec == 2.0
