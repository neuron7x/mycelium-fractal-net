"""
Tests for gRPC server endpoints.

Verifies gRPC service functionality including:
- Health check
- Validation cycle
- Simulation (unary and streaming)
- Nernst potential calculation
- Federated aggregation

Reference: docs/MFN_GRPC_API.md
"""

import time
from concurrent import futures

import grpc
import pytest

from src.mycelium_fractal_net.grpc.protos import mycelium_pb2, mycelium_pb2_grpc
from src.mycelium_fractal_net.grpc.server import MyceliumServicer


@pytest.fixture(scope="module")
def grpc_server():
    """Start gRPC server for testing."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    servicer = MyceliumServicer(require_auth=False, api_key="")
    mycelium_pb2_grpc.add_MyceliumServiceServicer_to_server(servicer, server)
    
    # Add port and get actual bound port
    actual_port = server.add_insecure_port('[::]:50052')  # Use different port for tests
    server.start()
    
    # Wait for server to be ready
    time.sleep(0.5)
    
    yield actual_port
    
    server.stop(grace=1)


@pytest.fixture
def grpc_stub(grpc_server):
    """Create gRPC stub for testing."""
    # grpc_server is the actual bound port number
    channel = grpc.insecure_channel(f'localhost:{grpc_server}')
    stub = mycelium_pb2_grpc.MyceliumServiceStub(channel)
    yield stub
    channel.close()


class TestGRPCHealthCheck:
    """Tests for HealthCheck RPC."""
    
    def test_health_check(self, grpc_stub):
        """Test health check endpoint."""
        request = mycelium_pb2.HealthCheckRequest()
        response = grpc_stub.HealthCheck(request)
        
        assert response.status == "healthy"
        assert response.version == "4.1.0"
        assert response.uptime_seconds >= 0


class TestGRPCValidate:
    """Tests for Validate RPC."""
    
    def test_validate_basic(self, grpc_stub):
        """Test basic validation cycle."""
        request = mycelium_pb2.ValidateRequest(
            seed=42,
            epochs=1,
            grid_size=32
        )
        response = grpc_stub.Validate(request, timeout=30.0)
        
        assert response.loss_start > 0
        assert response.loss_final >= 0
        assert response.loss_drop > 0
        assert -100 <= response.pot_min_mV <= 100
        assert -100 <= response.pot_max_mV <= 100
    
    def test_validate_with_different_params(self, grpc_stub):
        """Test validation with various parameters."""
        request = mycelium_pb2.ValidateRequest(
            seed=123,
            epochs=2,
            grid_size=64
        )
        response = grpc_stub.Validate(request, timeout=60.0)
        
        assert response.loss_drop > 0


class TestGRPCSimulate:
    """Tests for Simulate RPC."""
    
    def test_simulate_basic(self, grpc_stub):
        """Test basic simulation."""
        request = mycelium_pb2.SimulateRequest(
            seed=42,
            grid_size=32,
            steps=50,
            turing_enabled=True,
            alpha=0.1,
            gamma=0.8
        )
        response = grpc_stub.Simulate(request, timeout=30.0)
        
        assert len(response.field_mean) == 50
        assert len(response.field_std) == 50
        assert response.growth_events >= 0
        assert 0 < response.fractal_dimension < 3.0
        assert response.field_stats.mean != 0
    
    def test_simulate_small_grid(self, grpc_stub):
        """Test simulation with small grid."""
        request = mycelium_pb2.SimulateRequest(
            seed=42,
            grid_size=16,
            steps=20
        )
        response = grpc_stub.Simulate(request, timeout=20.0)
        
        assert len(response.field_mean) == 20
        assert response.fractal_dimension > 0


class TestGRPCSimulateStream:
    """Tests for SimulateStream RPC (streaming)."""
    
    def test_simulate_stream_basic(self, grpc_stub):
        """Test streaming simulation."""
        request = mycelium_pb2.SimulateRequest(
            seed=42,
            grid_size=16,
            steps=10
        )
        
        updates = []
        for update in grpc_stub.SimulateStream(request, timeout=30.0):
            updates.append(update)
            
            assert 0 <= update.step < update.total_steps
            assert update.pot_mean_mV != 0
            assert update.pot_std_mV >= 0
            
            if update.completed:
                assert update.step == update.total_steps - 1
                break
        
        assert len(updates) == 10
        assert updates[-1].completed is True
    
    def test_simulate_stream_cancellation(self, grpc_stub):
        """Test cancelling streaming simulation."""
        request = mycelium_pb2.SimulateRequest(
            seed=42,
            grid_size=32,
            steps=100
        )
        
        # Receive only first 5 updates then cancel
        count = 0
        for update in grpc_stub.SimulateStream(request, timeout=30.0):
            count += 1
            if count >= 5:
                break
        
        assert count == 5


class TestGRPCNernst:
    """Tests for ComputeNernst RPC."""
    
    def test_nernst_potassium(self, grpc_stub):
        """Test Nernst potential for K+ ion."""
        request = mycelium_pb2.NernstRequest(
            z_valence=1,
            concentration_out_molar=0.005,   # 5 mM
            concentration_in_molar=0.140,    # 140 mM
            temperature_k=310.0              # 37°C
        )
        response = grpc_stub.ComputeNernst(request)
        
        # K+ potential should be around -89 mV
        assert -95 <= response.potential_mV <= -85
    
    def test_nernst_sodium(self, grpc_stub):
        """Test Nernst potential for Na+ ion."""
        request = mycelium_pb2.NernstRequest(
            z_valence=1,
            concentration_out_molar=0.145,   # 145 mM
            concentration_in_molar=0.012,    # 12 mM
            temperature_k=310.0
        )
        response = grpc_stub.ComputeNernst(request)
        
        # Na+ potential should be positive
        assert 55 <= response.potential_mV <= 75


class TestGRPCFederated:
    """Tests for AggregateFederated RPC."""
    
    def test_aggregate_basic(self, grpc_stub):
        """Test basic gradient aggregation."""
        gradients = [
            mycelium_pb2.Gradient(values=[1.0, 2.0, 3.0]),
            mycelium_pb2.Gradient(values=[1.1, 2.1, 2.9]),
            mycelium_pb2.Gradient(values=[0.9, 1.9, 3.1]),
        ]
        
        request = mycelium_pb2.FederatedAggregateRequest(
            gradients=gradients,
            num_clusters=5,
            byzantine_fraction=0.2
        )
        response = grpc_stub.AggregateFederated(request, timeout=10.0)
        
        assert len(response.aggregated_gradient) == 3
        assert response.num_accepted > 0
    
    def test_aggregate_large_batch(self, grpc_stub):
        """Test aggregation with larger gradient batch."""
        # Create 20 similar gradients
        gradients = [
            mycelium_pb2.Gradient(values=[float(i + 0.1), float(i + 0.2), float(i + 0.3)])
            for i in range(20)
        ]
        
        request = mycelium_pb2.FederatedAggregateRequest(
            gradients=gradients,
            num_clusters=10,
            byzantine_fraction=0.2
        )
        response = grpc_stub.AggregateFederated(request, timeout=15.0)
        
        assert len(response.aggregated_gradient) > 0
        assert response.num_accepted > 0


class TestGRPCErrorHandling:
    """Tests for error handling in gRPC."""
    
    def test_invalid_nernst_params(self, grpc_stub):
        """Test Nernst with invalid parameters."""
        request = mycelium_pb2.NernstRequest(
            z_valence=0,  # Invalid: zero valence
            concentration_out_molar=0.005,
            concentration_in_molar=0.140,
            temperature_k=310.0
        )
        
        try:
            grpc_stub.ComputeNernst(request, timeout=5.0)
            # Should either return error or handle gracefully
        except grpc.RpcError as e:
            # Expected error
            assert e.code() in [grpc.StatusCode.INVALID_ARGUMENT, grpc.StatusCode.INTERNAL]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
