"""
gRPC client SDK for MyceliumFractalNet.

High-level client for interacting with MFN gRPC services.
Handles authentication, signatures, retries, and streaming.

Usage:
    async with MFNClient("localhost:50051", api_key="your-key") as client:
        response = await client.extract_features(seed=42, grid_size=64, steps=64)
        print(f"Fractal dimension: {response.fractal_dimension}")
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
import uuid
from typing import Any, AsyncIterator, Awaitable, Callable, List, Optional, Tuple, TypeVar

import grpc

from . import mfn_pb2, mfn_pb2_grpc

T = TypeVar("T")


class MFNClient:
    """
    High-level gRPC client for MyceliumFractalNet.
    
    Features:
        - Automatic request ID generation
        - Metadata signing (HMAC-SHA256)
        - Retry on transient errors
        - Streaming support
    """
    
    def __init__(
        self,
        address: str,
        api_key: str,
        tls_enabled: bool = False,
        retry_max_attempts: int = 3,
        retry_backoff_ms: int = 100,
    ) -> None:
        """
        Initialize gRPC client.
        
        Args:
            address: Server address (host:port)
            api_key: API key for authentication
            tls_enabled: Enable TLS
            retry_max_attempts: Max retry attempts
            retry_backoff_ms: Initial backoff delay (ms)
        """
        self.address = address
        self.api_key = api_key
        self.tls_enabled = tls_enabled
        self.retry_max_attempts = retry_max_attempts
        self.retry_backoff_ms = retry_backoff_ms
        
        self._channel: Optional[grpc.aio.Channel] = None
        self._features_stub: Optional[mfn_pb2_grpc.MFNFeaturesServiceStub] = None
        self._simulation_stub: Optional[mfn_pb2_grpc.MFNSimulationServiceStub] = None
        self._validation_stub: Optional[mfn_pb2_grpc.MFNValidationServiceStub] = None
    
    async def __aenter__(self) -> MFNClient:
        """Enter async context manager."""
        await self.connect()
        return self
    
    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Exit async context manager."""
        await self.close()
    
    async def connect(self) -> None:
        """Establish connection to gRPC server."""
        if self.tls_enabled:
            credentials = grpc.ssl_channel_credentials()
            self._channel = grpc.aio.secure_channel(self.address, credentials)
        else:
            self._channel = grpc.aio.insecure_channel(self.address)
        
        # Create stubs
        self._features_stub = mfn_pb2_grpc.MFNFeaturesServiceStub(self._channel)  # type: ignore[no-untyped-call]
        self._simulation_stub = mfn_pb2_grpc.MFNSimulationServiceStub(self._channel)  # type: ignore[no-untyped-call]
        self._validation_stub = mfn_pb2_grpc.MFNValidationServiceStub(self._channel)  # type: ignore[no-untyped-call]
    
    async def close(self) -> None:
        """Close connection to gRPC server."""
        if self._channel:
            await self._channel.close(grace=5.0)
            self._channel = None
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        return str(uuid.uuid4())
    
    def _generate_signature(self, request_id: str, timestamp: str) -> str:
        """
        Generate HMAC-SHA256 signature.
        
        Args:
            request_id: Request ID
            timestamp: Unix timestamp
            
        Returns:
            Hex-encoded signature
        """
        message = f"{request_id}:{timestamp}"
        return hmac.new(
            self.api_key.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
    
    def _build_metadata(self, request_id: str) -> List[Tuple[str, str]]:
        """
        Build gRPC metadata with authentication.
        
        Args:
            request_id: Request ID
            
        Returns:
            List of metadata tuples
        """
        timestamp = str(time.time())
        signature = self._generate_signature(request_id, timestamp)
        
        return [
            ("x-api-key", self.api_key),
            ("x-request-id", request_id),
            ("x-timestamp", timestamp),
            ("x-signature", signature),
        ]
    
    async def _retry_call(
        self, call_func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        """
        Execute gRPC call with retry logic.
        
        Args:
            call_func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Call result
            
        Raises:
            grpc.RpcError: If all retries fail
        """
        backoff = self.retry_backoff_ms / 1000.0
        
        for attempt in range(self.retry_max_attempts):
            try:
                return await call_func(*args, **kwargs)
            except grpc.RpcError as e:
                # Retry on transient errors
                if e.code() in (
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                ):
                    if attempt < self.retry_max_attempts - 1:
                        await asyncio.sleep(backoff)
                        backoff *= 2  # Exponential backoff
                        continue
                
                # Non-transient error or max retries reached
                raise
        
        raise RuntimeError("Max retry attempts reached")
    
    async def extract_features(
        self,
        seed: int = 42,
        grid_size: int = 64,
        steps: int = 64,
        alpha: float = 0.18,
        spike_probability: float = 0.25,
        turing_enabled: bool = True,
        request_id: Optional[str] = None,
    ) -> mfn_pb2.FeatureResponse:
        """
        Extract features from simulation.
        
        Args:
            seed: Random seed
            grid_size: Grid size (NxN)
            steps: Number of steps
            alpha: Diffusion coefficient
            spike_probability: Growth event probability
            turing_enabled: Enable Turing morphogenesis
            request_id: Optional request ID
            
        Returns:
            FeatureResponse with extracted features
        """
        request_id = request_id or self._generate_request_id()
        
        request = mfn_pb2.FeatureRequest(
            request_id=request_id,
            seed=seed,
            grid_size=grid_size,
            steps=steps,
            alpha=alpha,
            spike_probability=spike_probability,
            turing_enabled=turing_enabled,
        )
        
        metadata = self._build_metadata(request_id)
        
        assert self._features_stub is not None, "Client not connected"
        return await self._retry_call(
            self._features_stub.ExtractFeatures,
            request,
            metadata=metadata,
        )
    
    async def stream_features(
        self,
        seed: int = 42,
        grid_size: int = 64,
        total_steps: int = 100,
        steps_per_frame: int = 10,
        alpha: float = 0.18,
        spike_probability: float = 0.25,
        turing_enabled: bool = True,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[mfn_pb2.FeatureFrame]:
        """
        Stream features during simulation.
        
        Args:
            seed: Random seed
            grid_size: Grid size (NxN)
            total_steps: Total simulation steps
            steps_per_frame: Steps per frame
            alpha: Diffusion coefficient
            spike_probability: Growth event probability
            turing_enabled: Enable Turing morphogenesis
            request_id: Optional request ID
            
        Yields:
            FeatureFrame for each step interval
        """
        request_id = request_id or self._generate_request_id()
        
        request = mfn_pb2.FeatureStreamRequest(
            request_id=request_id,
            seed=seed,
            grid_size=grid_size,
            total_steps=total_steps,
            steps_per_frame=steps_per_frame,
            alpha=alpha,
            spike_probability=spike_probability,
            turing_enabled=turing_enabled,
        )
        
        metadata = self._build_metadata(request_id)
        
        assert self._features_stub is not None, "Client not connected"
        stream = self._features_stub.StreamFeatures(request, metadata=metadata)
        
        async for frame in stream:
            yield frame
    
    async def run_simulation(
        self,
        seed: int = 42,
        grid_size: int = 64,
        steps: int = 64,
        alpha: float = 0.18,
        spike_probability: float = 0.25,
        turing_enabled: bool = True,
        request_id: Optional[str] = None,
    ) -> mfn_pb2.SimulationResult:
        """
        Run complete simulation.
        
        Args:
            seed: Random seed
            grid_size: Grid size (NxN)
            steps: Number of steps
            alpha: Diffusion coefficient
            spike_probability: Growth event probability
            turing_enabled: Enable Turing morphogenesis
            request_id: Optional request ID
            
        Returns:
            SimulationResult with final state
        """
        request_id = request_id or self._generate_request_id()
        
        request = mfn_pb2.SimulationRequest(
            request_id=request_id,
            seed=seed,
            grid_size=grid_size,
            steps=steps,
            alpha=alpha,
            spike_probability=spike_probability,
            turing_enabled=turing_enabled,
        )
        
        metadata = self._build_metadata(request_id)
        
        assert self._simulation_stub is not None, "Client not connected"
        return await self._retry_call(
            self._simulation_stub.RunSimulation,
            request,
            metadata=metadata,
        )
    
    async def stream_simulation(
        self,
        seed: int = 42,
        grid_size: int = 64,
        total_steps: int = 100,
        steps_per_frame: int = 10,
        alpha: float = 0.18,
        spike_probability: float = 0.25,
        turing_enabled: bool = True,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[mfn_pb2.SimulationFrame]:
        """
        Stream simulation state updates.
        
        Args:
            seed: Random seed
            grid_size: Grid size (NxN)
            total_steps: Total simulation steps
            steps_per_frame: Steps per frame
            alpha: Diffusion coefficient
            spike_probability: Growth event probability
            turing_enabled: Enable Turing morphogenesis
            request_id: Optional request ID
            
        Yields:
            SimulationFrame for each step interval
        """
        request_id = request_id or self._generate_request_id()
        
        request = mfn_pb2.SimulationStreamRequest(
            request_id=request_id,
            seed=seed,
            grid_size=grid_size,
            total_steps=total_steps,
            steps_per_frame=steps_per_frame,
            alpha=alpha,
            spike_probability=spike_probability,
            turing_enabled=turing_enabled,
        )
        
        metadata = self._build_metadata(request_id)
        
        assert self._simulation_stub is not None, "Client not connected"
        stream = self._simulation_stub.StreamSimulation(request, metadata=metadata)
        
        async for frame in stream:
            yield frame
    
    async def validate_pattern(
        self,
        seed: int = 42,
        epochs: int = 1,
        batch_size: int = 4,
        grid_size: int = 64,
        steps: int = 64,
        turing_enabled: bool = True,
        quantum_jitter: bool = False,
        request_id: Optional[str] = None,
    ) -> mfn_pb2.ValidationResult:
        """
        Validate pattern using training cycle.
        
        Args:
            seed: Random seed
            epochs: Training epochs
            batch_size: Batch size
            grid_size: Grid size (NxN)
            steps: Simulation steps
            turing_enabled: Enable Turing morphogenesis
            quantum_jitter: Enable quantum jitter
            request_id: Optional request ID
            
        Returns:
            ValidationResult with training metrics
        """
        request_id = request_id or self._generate_request_id()
        
        request = mfn_pb2.ValidationRequest(
            request_id=request_id,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            grid_size=grid_size,
            steps=steps,
            turing_enabled=turing_enabled,
            quantum_jitter=quantum_jitter,
        )
        
        metadata = self._build_metadata(request_id)
        
        assert self._validation_stub is not None, "Client not connected"
        return await self._retry_call(
            self._validation_stub.ValidatePattern,
            request,
            metadata=metadata,
        )


__all__ = ["MFNClient"]
