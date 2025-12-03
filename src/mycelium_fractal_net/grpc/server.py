"""
gRPC server for MyceliumFractalNet.

Provides high-throughput gRPC API for validation, simulation, and feature extraction.
Uses async servicers for non-blocking I/O operations.

Environment Variables:
    MFN_GRPC_PORT - gRPC server port (default: 50051)
    MFN_GRPC_MAX_WORKERS - Thread pool size (default: 10)
    MFN_GRPC_MAX_MESSAGE_SIZE - Max message size in MB (default: 100)
    MFN_GRPC_MAX_CONCURRENT_STREAMS - Max concurrent streams (default: 1000)
    MFN_GRPC_KEEPALIVE_TIME_MS - Keepalive time (default: 60000)
    MFN_GRPC_KEEPALIVE_TIMEOUT_MS - Keepalive timeout (default: 20000)
    MFN_GRPC_TLS_ENABLED - Enable TLS (default: false)
    MFN_GRPC_TLS_CERT_PATH - TLS certificate path
    MFN_GRPC_TLS_KEY_PATH - TLS private key path

Reference: docs/MFN_GRPC_SPEC.md
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import AsyncIterator, Optional

import grpc
import numpy as np

from mycelium_fractal_net import (
    compute_fractal_features,
    run_validation,
    ValidationConfig,
)

from . import mfn_pb2, mfn_pb2_grpc
from .interceptors import AuditInterceptor, AuthInterceptor, RateLimitInterceptor

logger = logging.getLogger(__name__)

# Default simulation parameters for streaming
DEFAULT_ALPHA = 0.18
DEFAULT_SPIKE_PROBABILITY = 0.25
DEFAULT_TURING_ENABLED = True


class MFNFeaturesServiceServicer(mfn_pb2_grpc.MFNFeaturesServiceServicer):
    """
    gRPC servicer for feature extraction operations.
    
    Provides unary and streaming RPC for fractal feature computation.
    """

    async def ExtractFeatures(
        self,
        request: mfn_pb2.FeatureRequest,
        context: grpc.aio.ServicerContext,
    ) -> mfn_pb2.FeatureResponse:
        """
        Extract features from a simulated field.
        
        Args:
            request: Feature extraction request parameters
            context: gRPC context
            
        Returns:
            FeatureResponse with computed features
        """
        try:
            from mycelium_fractal_net import run_mycelium_simulation, SimulationConfig
            
            # Run simulation with config
            config = SimulationConfig(
                seed=request.seed,
                grid_size=request.grid_size,
                steps=request.steps,
                alpha=request.alpha,
                spike_probability=request.spike_probability,
                turing_enabled=request.turing_enabled,
            )
            
            simulation_result = run_mycelium_simulation(config)
            
            # Extract features
            features = compute_fractal_features(simulation_result)
            
            # Build response
            # Note: MFN features use specific keys. We map them to proto fields:
            # - D_box -> fractal_dimension
            # - D_r2 -> lacunarity (regression fit dimension as proxy)
            # - E_trend -> hurst_exponent (energy trend as proxy)
            # - V_mean, V_std -> spectral energy (voltage statistics)
            # - f_active -> active_nodes (fraction * 1000 for approximate count)
            # - f_active -> edge_density (fraction directly)
            # - max_cluster_size / 1000 -> clustering_coefficient (normalized)
            return mfn_pb2.FeatureResponse(
                request_id=request.request_id,
                meta=mfn_pb2.ResponseMeta(meta={"status": "ok"}),
                fractal_dimension=float(features.values["D_box"]),
                lacunarity=float(features.values.get("D_r2", 0.0)),
                hurst_exponent=float(features.values.get("E_trend", 0.0)),
                spectral_energy_mean=float(features.values.get("V_mean", 0.0)),
                spectral_energy_std=float(features.values.get("V_std", 0.0)),
                active_nodes=int(features.values.get("f_active", 0) * 1000),
                edge_density=float(features.values.get("f_active", 0.0)),
                clustering_coefficient=float(features.values.get("max_cluster_size", 0.0) / 1000.0),
            )
            
        except Exception as e:
            logger.error(f"ExtractFeatures failed: {e}", exc_info=True)
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Feature extraction failed: {str(e)}",
            )

    async def StreamFeatures(
        self,
        request: mfn_pb2.FeatureStreamRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[mfn_pb2.FeatureFrame]:
        """
        Stream features during simulation execution.
        
        Args:
            request: Streaming request parameters
            context: gRPC context
            
        Yields:
            FeatureFrame for each simulation step
        """
        try:
            from mycelium_fractal_net import run_mycelium_simulation, SimulationConfig
            
            # Run simulation with streaming updates
            for step in range(0, request.steps, request.stream_interval):
                if context.cancelled():
                    logger.info("StreamFeatures cancelled by client")
                    return
                
                # Run simulation up to this step
                config = SimulationConfig(
                    seed=request.seed,
                    grid_size=request.grid_size,
                    steps=min(step + request.stream_interval, request.steps),
                    alpha=request.alpha if request.alpha > 0 else DEFAULT_ALPHA,
                    spike_probability=request.spike_probability if request.spike_probability > 0 else DEFAULT_SPIKE_PROBABILITY,
                    turing_enabled=request.turing_enabled if hasattr(request, 'turing_enabled') else DEFAULT_TURING_ENABLED,
                )
                
                simulation_result = run_mycelium_simulation(config)
                features = compute_fractal_features(simulation_result)
                
                yield mfn_pb2.FeatureFrame(
                    request_id=request.request_id,
                    step=step,
                    fractal_dimension=float(features.values["D_box"]),
                    spectral_energy_mean=float(features.values.get("V_mean", 0.0)),
                    active_nodes=int(features.values.get("f_active", 0) * 1000),
                )
                
                # Small delay to prevent overwhelming clients
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"StreamFeatures failed: {e}", exc_info=True)
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Feature streaming failed: {str(e)}",
            )


class MFNSimulationServiceServicer(mfn_pb2_grpc.MFNSimulationServiceServicer):
    """
    gRPC servicer for simulation operations.
    
    Provides unary and streaming RPC for mycelium field simulation.
    """

    async def RunSimulation(
        self,
        request: mfn_pb2.SimulationRequest,
        context: grpc.aio.ServicerContext,
    ) -> mfn_pb2.SimulationResult:
        """
        Run a complete simulation.
        
        Args:
            request: Simulation request parameters
            context: gRPC context
            
        Returns:
            SimulationResult with statistics
        """
        try:
            from mycelium_fractal_net import run_mycelium_simulation, SimulationConfig, estimate_fractal_dimension
            
            # Run simulation with config
            config = SimulationConfig(
                seed=request.seed,
                grid_size=request.grid_size,
                steps=request.steps,
                alpha=request.alpha,
                spike_probability=request.spike_probability,
                turing_enabled=request.turing_enabled,
            )
            
            simulation_result = run_mycelium_simulation(config)
            
            # Extract statistics from result
            field_mv = simulation_result.field * 1000.0  # Convert to mV
            
            # Compute fractal dimension using mean as threshold
            binary = simulation_result.field > np.mean(simulation_result.field)
            fractal_dim = estimate_fractal_dimension(binary)
            
            return mfn_pb2.SimulationResult(
                request_id=request.request_id,
                meta=mfn_pb2.ResponseMeta(meta={"status": "ok"}),
                growth_events=simulation_result.growth_events,
                pot_min_mV=float(np.min(field_mv)),
                pot_max_mV=float(np.max(field_mv)),
                pot_mean_mV=float(np.mean(field_mv)),
                pot_std_mV=float(np.std(field_mv)),
                fractal_dimension=float(fractal_dim),
            )
            
        except Exception as e:
            logger.error(f"RunSimulation failed: {e}", exc_info=True)
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Simulation failed: {str(e)}",
            )

    async def StreamSimulation(
        self,
        request: mfn_pb2.SimulationStreamRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[mfn_pb2.SimulationFrame]:
        """
        Stream simulation state in real-time.
        
        Args:
            request: Streaming request parameters
            context: gRPC context
            
        Yields:
            SimulationFrame for each simulation step
        """
        try:
            from mycelium_fractal_net import run_mycelium_simulation, SimulationConfig, estimate_fractal_dimension
            
            # Run simulation with streaming updates
            for step in range(0, request.steps, request.stream_interval):
                if context.cancelled():
                    logger.info("StreamSimulation cancelled by client")
                    return
                
                # Run simulation up to this step
                config = SimulationConfig(
                    seed=request.seed,
                    grid_size=request.grid_size,
                    steps=min(step + request.stream_interval, request.steps),
                    alpha=request.alpha if request.alpha > 0 else DEFAULT_ALPHA,
                    spike_probability=request.spike_probability if request.spike_probability > 0 else DEFAULT_SPIKE_PROBABILITY,
                    turing_enabled=request.turing_enabled if hasattr(request, 'turing_enabled') else DEFAULT_TURING_ENABLED,
                )
                
                simulation_result = run_mycelium_simulation(config)
                
                # Compute statistics
                field_mv = simulation_result.field * 1000.0
                binary = simulation_result.field > np.mean(simulation_result.field)
                fractal_dim = estimate_fractal_dimension(binary)
                
                yield mfn_pb2.SimulationFrame(
                    request_id=request.request_id,
                    step=step,
                    growth_events=simulation_result.growth_events,
                    pot_mean_mV=float(np.mean(field_mv)),
                    fractal_dimension=float(fractal_dim),
                )
                
                # Small delay
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"StreamSimulation failed: {e}", exc_info=True)
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Simulation streaming failed: {str(e)}",
            )


class MFNValidationServiceServicer(mfn_pb2_grpc.MFNValidationServiceServicer):
    """
    gRPC servicer for validation operations.
    
    Provides RPC for validation cycle execution.
    """

    async def ValidatePattern(
        self,
        request: mfn_pb2.ValidationRequest,
        context: grpc.aio.ServicerContext,
    ) -> mfn_pb2.ValidationResult:
        """
        Validate a pattern configuration.
        
        Args:
            request: Validation request parameters
            context: gRPC context
            
        Returns:
            ValidationResult with metrics
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
            
            # Run validation
            metrics = run_validation(cfg)
            
            return mfn_pb2.ValidationResult(
                request_id=request.request_id,
                meta=mfn_pb2.ResponseMeta(meta={"status": "ok"}),
                loss_start=float(metrics["loss_start"]),
                loss_final=float(metrics["loss_final"]),
                loss_drop=float(metrics["loss_drop"]),
                pot_min_mV=float(metrics["pot_min_mV"]),
                pot_max_mV=float(metrics["pot_max_mV"]),
                example_fractal_dim=float(metrics["example_fractal_dim"]),
                lyapunov_exponent=float(metrics["lyapunov_exponent"]),
                growth_events=float(metrics["growth_events"]),
                nernst_symbolic_mV=float(metrics["nernst_symbolic_mV"]),
                nernst_numeric_mV=float(metrics["nernst_numeric_mV"]),
            )
            
        except Exception as e:
            logger.error(f"ValidatePattern failed: {e}", exc_info=True)
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Validation failed: {str(e)}",
            )


async def serve(
    port: int = 50051,
    tls_cert: Optional[str] = None,
    tls_key: Optional[str] = None,
    enable_auth: bool = True,
    enable_audit: bool = True,
    enable_rate_limit: bool = True,
) -> grpc.aio.Server:
    """
    Start the gRPC server.
    
    Args:
        port: Port to bind (default: 50051)
        tls_cert: Path to TLS certificate (optional)
        tls_key: Path to TLS private key (optional)
        enable_auth: Enable authentication interceptor
        enable_audit: Enable audit logging interceptor
        enable_rate_limit: Enable rate limiting interceptor
        
    Returns:
        Running gRPC server instance
    """
    # Build interceptors chain
    interceptors = []
    if enable_auth:
        interceptors.append(AuthInterceptor())
    if enable_audit:
        interceptors.append(AuditInterceptor())
    if enable_rate_limit:
        interceptors.append(RateLimitInterceptor())
    
    # Server options
    max_message_size = int(os.getenv("MFN_GRPC_MAX_MESSAGE_SIZE", "100")) * 1024 * 1024
    max_concurrent_streams = int(os.getenv("MFN_GRPC_MAX_CONCURRENT_STREAMS", "1000"))
    keepalive_time_ms = int(os.getenv("MFN_GRPC_KEEPALIVE_TIME_MS", "60000"))
    keepalive_timeout_ms = int(os.getenv("MFN_GRPC_KEEPALIVE_TIMEOUT_MS", "20000"))
    
    options = [
        ("grpc.max_send_message_length", max_message_size),
        ("grpc.max_receive_message_length", max_message_size),
        ("grpc.max_concurrent_streams", max_concurrent_streams),
        ("grpc.keepalive_time_ms", keepalive_time_ms),
        ("grpc.keepalive_timeout_ms", keepalive_timeout_ms),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.max_pings_without_data", 0),
    ]
    
    # Create server
    server = grpc.aio.server(
        interceptors=interceptors,
        options=options,
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
    if tls_cert and tls_key:
        with open(tls_cert, "rb") as f:
            cert_chain = f.read()
        with open(tls_key, "rb") as f:
            private_key = f.read()
        
        credentials = grpc.ssl_server_credentials([(private_key, cert_chain)])
        server.add_secure_port(f"[::]:{port}", credentials)
        logger.info(f"Starting gRPC server with TLS on port {port}")
    else:
        server.add_insecure_port(f"[::]:{port}")
        logger.info(f"Starting gRPC server (insecure) on port {port}")
    
    await server.start()
    logger.info("gRPC server started successfully")
    
    return server


async def serve_forever() -> None:
    """
    Run the gRPC server until interrupted.
    
    Reads configuration from environment variables.
    """
    port = int(os.getenv("MFN_GRPC_PORT", "50051"))
    tls_enabled = os.getenv("MFN_GRPC_TLS_ENABLED", "false").lower() == "true"
    tls_cert = os.getenv("MFN_GRPC_TLS_CERT_PATH") if tls_enabled else None
    tls_key = os.getenv("MFN_GRPC_TLS_KEY_PATH") if tls_enabled else None
    
    server = await serve(
        port=port,
        tls_cert=tls_cert,
        tls_key=tls_key,
    )
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        await server.stop(grace=5.0)
        logger.info("gRPC server stopped")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(serve_forever())
