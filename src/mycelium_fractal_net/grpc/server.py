"""
gRPC server for MyceliumFractalNet.

Implements async gRPC services:
    - MFNFeaturesService: Feature extraction
    - MFNSimulationService: Field simulation
    - MFNValidationService: Pattern validation
"""

from __future__ import annotations

import asyncio
import signal
from typing import AsyncIterator, Optional

import numpy as np

import grpc
from grpc import aio
from mycelium_fractal_net import (
    estimate_fractal_dimension,
    run_validation,
    simulate_mycelium_field,
)
from mycelium_fractal_net.integration import get_logger
from mycelium_fractal_net.model import ValidationConfig

from . import mfn_pb2, mfn_pb2_grpc
from .config import GRPCConfig, get_grpc_config
from .interceptors import AuditInterceptor, AuthInterceptor, RateLimitInterceptor

logger = get_logger("grpc.server")


class MFNFeaturesServiceServicer(mfn_pb2_grpc.MFNFeaturesServiceServicer):
    """Servicer for feature extraction."""
    
    async def ExtractFeatures(
        self,
        request: mfn_pb2.FeatureRequest,
        context: grpc.aio.ServicerContext,
    ) -> mfn_pb2.FeatureResponse:
        """
        Extract features from simulation data.
        
        Args:
            request: Feature extraction request
            context: gRPC context
            
        Returns:
            FeatureResponse with extracted features
        """
        try:
            # Run simulation
            rng = np.random.default_rng(request.seed)
            field, growth_events = simulate_mycelium_field(
                rng,
                grid_size=request.grid_size,
                steps=request.steps,
                alpha=request.alpha,
                spike_probability=request.spike_probability,
                turing_enabled=request.turing_enabled,
            )
            
            # Extract features
            fractal_dim = estimate_fractal_dimension(field > -0.060)
            
            # Field statistics (convert from V to mV)
            pot_min_mV = float(np.min(field) * 1000)
            pot_max_mV = float(np.max(field) * 1000)
            pot_mean_mV = float(np.mean(field) * 1000)
            pot_std_mV = float(np.std(field) * 1000)
            
            # Build response
            return mfn_pb2.FeatureResponse(
                request_id=request.request_id,
                meta=mfn_pb2.ResponseMeta(
                    meta={
                        "server": "mfn-grpc",
                        "version": "4.1.0",
                    }
                ),
                fractal_dimension=fractal_dim,
                pot_min_mV=pot_min_mV,
                pot_max_mV=pot_max_mV,
                pot_mean_mV=pot_mean_mV,
                pot_std_mV=pot_std_mV,
                growth_events=growth_events,
            )
        
        except Exception as e:
            logger.error(
                f"Feature extraction failed: {e}",
                extra={"request_id": request.request_id},
            )
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Feature extraction failed: {str(e)}",
            )
            raise  # For mypy: abort() never returns
    
    async def StreamFeatures(
        self,
        request: mfn_pb2.FeatureStreamRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[mfn_pb2.FeatureFrame]:
        """
        Stream features during simulation.
        
        Args:
            request: Feature stream request
            context: gRPC context
            
        Yields:
            FeatureFrame for each step interval
        """
        try:
            rng = np.random.default_rng(request.seed)
            steps_per_frame = request.steps_per_frame or 10
            total_growth_events = 0
            
            for step in range(0, request.total_steps, steps_per_frame):
                # Run simulation for this frame
                current_steps = min(steps_per_frame, request.total_steps - step)
                field, growth_events = simulate_mycelium_field(
                    rng,
                    grid_size=request.grid_size,
                    steps=current_steps,
                    alpha=request.alpha,
                    spike_probability=request.spike_probability,
                    turing_enabled=request.turing_enabled,
                )
                
                total_growth_events += growth_events
                
                # Extract features
                fractal_dim = estimate_fractal_dimension(field > -0.060)
                pot_min_mV = float(np.min(field) * 1000)
                pot_max_mV = float(np.max(field) * 1000)
                pot_mean_mV = float(np.mean(field) * 1000)
                pot_std_mV = float(np.std(field) * 1000)
                
                is_final = (step + current_steps >= request.total_steps)
                
                # Yield frame
                yield mfn_pb2.FeatureFrame(
                    request_id=request.request_id,
                    step=step + current_steps,
                    fractal_dimension=fractal_dim,
                    pot_min_mV=pot_min_mV,
                    pot_max_mV=pot_max_mV,
                    pot_mean_mV=pot_mean_mV,
                    pot_std_mV=pot_std_mV,
                    growth_events=total_growth_events,
                    is_final=is_final,
                )
                
                # Allow cancellation
                if context.cancelled():
                    logger.info(
                        f"Stream cancelled at step {step}",
                        extra={"request_id": request.request_id},
                    )
                    return
        
        except Exception as e:
            logger.error(
                f"Feature streaming failed: {e}",
                extra={"request_id": request.request_id},
            )
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Feature streaming failed: {str(e)}",
            )
            raise  # For mypy: abort() never returns


class MFNSimulationServiceServicer(mfn_pb2_grpc.MFNSimulationServiceServicer):
    """Servicer for field simulation."""
    
    async def RunSimulation(
        self,
        request: mfn_pb2.SimulationRequest,
        context: grpc.aio.ServicerContext,
    ) -> mfn_pb2.SimulationResult:
        """
        Run a complete simulation.
        
        Args:
            request: Simulation request
            context: gRPC context
            
        Returns:
            SimulationResult with final state
        """
        try:
            # Run simulation
            rng = np.random.default_rng(request.seed)
            field, growth_events = simulate_mycelium_field(
                rng,
                grid_size=request.grid_size,
                steps=request.steps,
                alpha=request.alpha,
                spike_probability=request.spike_probability,
                turing_enabled=request.turing_enabled,
            )
            
            # Compute features
            fractal_dim = estimate_fractal_dimension(field > -0.060)
            
            # Field statistics (V to mV)
            pot_min_mV = float(np.min(field) * 1000)
            pot_max_mV = float(np.max(field) * 1000)
            pot_mean_mV = float(np.mean(field) * 1000)
            pot_std_mV = float(np.std(field) * 1000)
            
            # Build response
            return mfn_pb2.SimulationResult(
                request_id=request.request_id,
                meta=mfn_pb2.ResponseMeta(
                    meta={
                        "server": "mfn-grpc",
                        "version": "4.1.0",
                    }
                ),
                growth_events=growth_events,
                pot_min_mV=pot_min_mV,
                pot_max_mV=pot_max_mV,
                pot_mean_mV=pot_mean_mV,
                pot_std_mV=pot_std_mV,
                fractal_dimension=fractal_dim,
            )
        
        except Exception as e:
            logger.error(
                f"Simulation failed: {e}",
                extra={"request_id": request.request_id},
            )
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Simulation failed: {str(e)}",
            )
            raise  # For mypy: abort() never returns
    
    async def StreamSimulation(
        self,
        request: mfn_pb2.SimulationStreamRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[mfn_pb2.SimulationFrame]:
        """
        Stream simulation state updates.
        
        Args:
            request: Simulation stream request
            context: gRPC context
            
        Yields:
            SimulationFrame for each step interval
        """
        try:
            rng = np.random.default_rng(request.seed)
            steps_per_frame = request.steps_per_frame or 10
            total_growth_events = 0
            
            for step in range(0, request.total_steps, steps_per_frame):
                # Run simulation for this frame
                current_steps = min(steps_per_frame, request.total_steps - step)
                field, growth_events = simulate_mycelium_field(
                    rng,
                    grid_size=request.grid_size,
                    steps=current_steps,
                    alpha=request.alpha,
                    spike_probability=request.spike_probability,
                    turing_enabled=request.turing_enabled,
                )
                
                total_growth_events += growth_events
                
                # Field statistics
                pot_min_mV = float(np.min(field) * 1000)
                pot_max_mV = float(np.max(field) * 1000)
                pot_mean_mV = float(np.mean(field) * 1000)
                pot_std_mV = float(np.std(field) * 1000)
                
                is_final = (step + current_steps >= request.total_steps)
                
                # Yield frame
                yield mfn_pb2.SimulationFrame(
                    request_id=request.request_id,
                    step=step + current_steps,
                    growth_events=total_growth_events,
                    pot_min_mV=pot_min_mV,
                    pot_max_mV=pot_max_mV,
                    pot_mean_mV=pot_mean_mV,
                    pot_std_mV=pot_std_mV,
                    is_final=is_final,
                )
                
                # Allow cancellation
                if context.cancelled():
                    logger.info(
                        f"Stream cancelled at step {step}",
                        extra={"request_id": request.request_id},
                    )
                    return
        
        except Exception as e:
            logger.error(
                f"Simulation streaming failed: {e}",
                extra={"request_id": request.request_id},
            )
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Simulation streaming failed: {str(e)}",
            )
            raise  # For mypy: abort() never returns


class MFNValidationServiceServicer(mfn_pb2_grpc.MFNValidationServiceServicer):
    """Servicer for pattern validation."""
    
    async def ValidatePattern(
        self,
        request: mfn_pb2.ValidationRequest,
        context: grpc.aio.ServicerContext,
    ) -> mfn_pb2.ValidationResult:
        """
        Validate a pattern using training cycle.
        
        Args:
            request: Validation request
            context: gRPC context
            
        Returns:
            ValidationResult with training metrics
        """
        try:
            # Build config
            cfg = ValidationConfig(
                seed=request.seed,
                epochs=request.epochs,
                batch_size=request.batch_size,
                grid_size=request.grid_size,
                steps=request.steps,
                turing_enabled=request.turing_enabled,
                quantum_jitter=request.quantum_jitter,
            )
            
            # Run validation (blocking, but we're in executor)
            loop = asyncio.get_event_loop()
            metrics = await loop.run_in_executor(None, run_validation, cfg)
            
            # Build response
            return mfn_pb2.ValidationResult(
                request_id=request.request_id,
                meta=mfn_pb2.ResponseMeta(
                    meta={
                        "server": "mfn-grpc",
                        "version": "4.1.0",
                    }
                ),
                loss_start=metrics["loss_start"],
                loss_final=metrics["loss_final"],
                loss_drop=metrics["loss_drop"],
                pot_min_mV=metrics["pot_min_mV"],
                pot_max_mV=metrics["pot_max_mV"],
                example_fractal_dim=metrics["example_fractal_dim"],
                lyapunov_exponent=metrics["lyapunov_exponent"],
                growth_events=metrics["growth_events"],
                nernst_symbolic_mV=metrics["nernst_symbolic_mV"],
                nernst_numeric_mV=metrics["nernst_numeric_mV"],
            )
        
        except Exception as e:
            logger.error(
                f"Validation failed: {e}",
                extra={"request_id": request.request_id},
            )
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Validation failed: {str(e)}",
            )
            raise  # For mypy: abort() never returns


async def serve(
    config: Optional[GRPCConfig] = None,
    interceptors_enabled: bool = True,
) -> aio.Server:
    """
    Start gRPC server.
    
    Args:
        config: Server configuration (uses env vars if None)
        interceptors_enabled: Enable auth/audit/rate-limit interceptors
        
    Returns:
        Running gRPC server
    """
    cfg = config or get_grpc_config()
    
    # Build interceptors
    interceptors = []
    if interceptors_enabled and cfg.auth_enabled:
        interceptors.append(AuthInterceptor())
    interceptors.append(AuditInterceptor())
    if interceptors_enabled:
        interceptors.append(
            RateLimitInterceptor(
                rps_limit=cfg.rate_limit_rps,
                concurrent_limit=cfg.rate_limit_concurrent,
            )
        )
    
    # Create server
    server = aio.server(
        interceptors=interceptors,
        options=[
            ("grpc.max_send_message_length", cfg.max_message_size),
            ("grpc.max_receive_message_length", cfg.max_message_size),
            ("grpc.max_concurrent_streams", cfg.max_concurrent_streams),
            ("grpc.keepalive_time_ms", cfg.keepalive_time_ms),
            ("grpc.keepalive_timeout_ms", cfg.keepalive_timeout_ms),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_time_between_pings_ms", 10000),
            ("grpc.http2.min_ping_interval_without_data_ms", 5000),
        ],
    )
    
    # Add servicers
    mfn_pb2_grpc.add_MFNFeaturesServiceServicer_to_server(
        MFNFeaturesServiceServicer(), server
    )
    mfn_pb2_grpc.add_MFNSimulationServiceServicer_to_server(
        MFNSimulationServiceServicer(), server
    )
    mfn_pb2_grpc.add_MFNValidationServiceServicer_to_server(
        MFNValidationServiceServicer(), server
    )
    
    # Configure TLS if enabled
    if cfg.tls_enabled and cfg.tls_cert_path and cfg.tls_key_path:
        with open(cfg.tls_key_path, "rb") as f:
            private_key = f.read()
        with open(cfg.tls_cert_path, "rb") as f:
            certificate = f.read()
        
        credentials = grpc.ssl_server_credentials(
            [(private_key, certificate)]
        )
        server.add_secure_port(f"[::]:{cfg.port}", credentials)
        logger.info(f"Starting gRPC server with TLS on port {cfg.port}")
    else:
        server.add_insecure_port(f"[::]:{cfg.port}")
        logger.info(f"Starting gRPC server (insecure) on port {cfg.port}")
    
    # Start server
    await server.start()
    
    return server


async def serve_forever(config: Optional[GRPCConfig] = None) -> None:
    """
    Start gRPC server and wait for termination.
    
    Args:
        config: Server configuration
    """
    server = await serve(config)
    
    # Setup signal handlers for graceful shutdown
    async def shutdown() -> None:
        logger.info("Shutting down gRPC server...")
        await server.stop(grace=5.0)
        logger.info("Server stopped")
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
    
    # Wait for server termination
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve_forever())
