"""
gRPC client SDK for MyceliumFractalNet.

Provides high-level async client for MFN gRPC API with:
- Automatic request ID generation
- Metadata signing (HMAC-SHA256)
- Retry logic with exponential backoff
- Streaming support

Usage:
    async with MFNClient("localhost:50051", api_key="your-key") as client:
        result = await client.run_simulation(seed=42, grid_size=64, steps=64)
        print(f"Fractal dimension: {result.fractal_dimension}")

Reference: docs/MFN_GRPC_SPEC.md
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
import uuid
from typing import AsyncIterator, Optional

import grpc

from . import mfn_pb2, mfn_pb2_grpc

logger = logging.getLogger(__name__)


class MFNClient:
    """
    High-level async client for MFN gRPC API.
    
    Provides convenient methods for all MFN operations with automatic
    authentication, retry, and error handling.
    """

    def __init__(
        self,
        address: str,
        api_key: Optional[str] = None,
        use_tls: bool = False,
        max_retries: int = 3,
        retry_backoff_sec: float = 1.0,
        timeout_sec: float = 30.0,
    ):
        """
        Initialize MFN client.
        
        Args:
            address: Server address (e.g., "localhost:50051")
            api_key: API key for authentication (optional)
            use_tls: Use TLS connection (default: False)
            max_retries: Max retries on transient errors (default: 3)
            retry_backoff_sec: Initial backoff time for retries (default: 1.0)
            timeout_sec: Default RPC timeout (default: 30.0)
        """
        self.address = address
        self.api_key = api_key
        self.use_tls = use_tls
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.timeout_sec = timeout_sec
        
        self._channel: Optional[grpc.aio.Channel] = None
        self._features_stub: Optional[mfn_pb2_grpc.MFNFeaturesServiceStub] = None
        self._simulation_stub: Optional[mfn_pb2_grpc.MFNSimulationServiceStub] = None
        self._validation_stub: Optional[mfn_pb2_grpc.MFNValidationServiceStub] = None

    async def __aenter__(self) -> MFNClient:
        """Context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        return str(uuid.uuid4())

    def _sign_request(self, timestamp: str, method_name: str) -> str:
        """
        Generate HMAC-SHA256 signature for request.
        
        Args:
            timestamp: Unix timestamp as string
            method_name: gRPC method name
            
        Returns:
            Hex-encoded signature
        """
        if not self.api_key:
            return ""
        
        message = f"{timestamp}{method_name}".encode("utf-8")
        signature = hmac.new(
            self.api_key.encode("utf-8"),
            message,
            hashlib.sha256,
        ).hexdigest()
        
        return signature

    def _create_metadata(self, method_name: str) -> list[tuple[str, str]]:
        """
        Create gRPC metadata with authentication.
        
        Args:
            method_name: gRPC method name
            
        Returns:
            List of metadata tuples
        """
        timestamp = str(time.time())
        metadata = [
            ("x-timestamp", timestamp),
        ]
        
        if self.api_key:
            signature = self._sign_request(timestamp, method_name)
            metadata.extend([
                ("x-api-key", self.api_key),
                ("x-signature", signature),
            ])
        
        return metadata

    async def connect(self) -> None:
        """Establish connection to gRPC server."""
        if self._channel is not None:
            logger.warning("Client already connected")
            return
        
        # Create channel
        if self.use_tls:
            credentials = grpc.ssl_channel_credentials()
            self._channel = grpc.aio.secure_channel(self.address, credentials)
            logger.info(f"Connected to {self.address} (TLS)")
        else:
            self._channel = grpc.aio.insecure_channel(self.address)
            logger.info(f"Connected to {self.address} (insecure)")
        
        # Create stubs
        self._features_stub = mfn_pb2_grpc.MFNFeaturesServiceStub(self._channel)
        self._simulation_stub = mfn_pb2_grpc.MFNSimulationServiceStub(self._channel)
        self._validation_stub = mfn_pb2_grpc.MFNValidationServiceStub(self._channel)

    async def close(self) -> None:
        """Close connection to gRPC server."""
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._features_stub = None
            self._simulation_stub = None
            self._validation_stub = None
            logger.info("Connection closed")

    async def _retry_call(self, call_func, *args, **kwargs):
        """
        Execute RPC with retry logic.
        
        Args:
            call_func: Async function to call
            *args, **kwargs: Arguments to pass
            
        Returns:
            Result from call_func
            
        Raises:
            grpc.RpcError: On final failure
        """
        backoff = self.retry_backoff_sec
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await call_func(*args, **kwargs)
            
            except grpc.RpcError as e:
                last_error = e
                
                # Check if error is retryable
                if e.code() in [
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                ]:
                    if attempt < self.max_retries:
                        logger.warning(
                            f"Transient error on attempt {attempt + 1}: {e.code()} "
                            f"- retrying in {backoff:.2f}s"
                        )
                        await asyncio.sleep(backoff)
                        backoff *= 2  # Exponential backoff
                        continue
                
                # Non-retryable error
                raise
        
        # All retries exhausted
        raise last_error

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
            grid_size: Grid size
            steps: Simulation steps
            alpha: Diffusion coefficient
            spike_probability: Growth probability
            turing_enabled: Enable Turing patterns
            request_id: Custom request ID (auto-generated if None)
            
        Returns:
            FeatureResponse with computed features
        """
        if self._features_stub is None:
            raise RuntimeError("Client not connected")
        
        if request_id is None:
            request_id = self._generate_request_id()
        
        request = mfn_pb2.FeatureRequest(
            request_id=request_id,
            seed=seed,
            grid_size=grid_size,
            steps=steps,
            alpha=alpha,
            spike_probability=spike_probability,
            turing_enabled=turing_enabled,
        )
        
        metadata = self._create_metadata("/mfn.MFNFeaturesService/ExtractFeatures")
        
        async def call():
            return await self._features_stub.ExtractFeatures(
                request,
                metadata=metadata,
                timeout=self.timeout_sec,
            )
        
        return await self._retry_call(call)

    async def stream_features(
        self,
        seed: int = 42,
        grid_size: int = 64,
        steps: int = 64,
        stream_interval: int = 10,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[mfn_pb2.FeatureFrame]:
        """
        Stream features during simulation.
        
        Args:
            seed: Random seed
            grid_size: Grid size
            steps: Total simulation steps
            stream_interval: Frames between updates
            request_id: Custom request ID (auto-generated if None)
            
        Yields:
            FeatureFrame for each update
        """
        if self._features_stub is None:
            raise RuntimeError("Client not connected")
        
        if request_id is None:
            request_id = self._generate_request_id()
        
        request = mfn_pb2.FeatureStreamRequest(
            request_id=request_id,
            seed=seed,
            grid_size=grid_size,
            steps=steps,
            stream_interval=stream_interval,
        )
        
        metadata = self._create_metadata("/mfn.MFNFeaturesService/StreamFeatures")
        
        async for frame in self._features_stub.StreamFeatures(
            request,
            metadata=metadata,
            timeout=self.timeout_sec * 10,  # Longer timeout for streaming
        ):
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
            grid_size: Grid size
            steps: Simulation steps
            alpha: Diffusion coefficient
            spike_probability: Growth probability
            turing_enabled: Enable Turing patterns
            request_id: Custom request ID (auto-generated if None)
            
        Returns:
            SimulationResult with statistics
        """
        if self._simulation_stub is None:
            raise RuntimeError("Client not connected")
        
        if request_id is None:
            request_id = self._generate_request_id()
        
        request = mfn_pb2.SimulationRequest(
            request_id=request_id,
            seed=seed,
            grid_size=grid_size,
            steps=steps,
            alpha=alpha,
            spike_probability=spike_probability,
            turing_enabled=turing_enabled,
        )
        
        metadata = self._create_metadata("/mfn.MFNSimulationService/RunSimulation")
        
        async def call():
            return await self._simulation_stub.RunSimulation(
                request,
                metadata=metadata,
                timeout=self.timeout_sec,
            )
        
        return await self._retry_call(call)

    async def stream_simulation(
        self,
        seed: int = 42,
        grid_size: int = 64,
        steps: int = 64,
        stream_interval: int = 10,
        request_id: Optional[str] = None,
    ) -> AsyncIterator[mfn_pb2.SimulationFrame]:
        """
        Stream simulation state in real-time.
        
        Args:
            seed: Random seed
            grid_size: Grid size
            steps: Total simulation steps
            stream_interval: Frames between updates
            request_id: Custom request ID (auto-generated if None)
            
        Yields:
            SimulationFrame for each update
        """
        if self._simulation_stub is None:
            raise RuntimeError("Client not connected")
        
        if request_id is None:
            request_id = self._generate_request_id()
        
        request = mfn_pb2.SimulationStreamRequest(
            request_id=request_id,
            seed=seed,
            grid_size=grid_size,
            steps=steps,
            stream_interval=stream_interval,
        )
        
        metadata = self._create_metadata("/mfn.MFNSimulationService/StreamSimulation")
        
        async for frame in self._simulation_stub.StreamSimulation(
            request,
            metadata=metadata,
            timeout=self.timeout_sec * 10,  # Longer timeout for streaming
        ):
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
        Validate pattern configuration.
        
        Args:
            seed: Random seed
            epochs: Training epochs
            batch_size: Batch size
            grid_size: Grid size
            steps: Simulation steps
            turing_enabled: Enable Turing patterns
            quantum_jitter: Enable quantum noise
            request_id: Custom request ID (auto-generated if None)
            
        Returns:
            ValidationResult with metrics
        """
        if self._validation_stub is None:
            raise RuntimeError("Client not connected")
        
        if request_id is None:
            request_id = self._generate_request_id()
        
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
        
        metadata = self._create_metadata("/mfn.MFNValidationService/ValidatePattern")
        
        async def call():
            return await self._validation_stub.ValidatePattern(
                request,
                metadata=metadata,
                timeout=self.timeout_sec * 5,  # Longer timeout for validation
            )
        
        return await self._retry_call(call)
