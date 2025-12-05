"""
gRPC server for MyceliumFractalNet v4.1.

Provides high-performance gRPC interface for fractal neural network operations.
Supports both unary and streaming RPCs for real-time simulation updates.

Usage:
    python grpc_server.py --port 50051

Environment Variables:
    MFN_GRPC_PORT - gRPC server port (default: 50051)
    MFN_GRPC_MAX_WORKERS - Maximum thread pool workers (default: 10)
    MFN_GRPC_AUTH_REQUIRED - Whether to require authentication (default: false)
    MFN_GRPC_API_KEY - API key for authentication

Reference: docs/MFN_INTEGRATION_GAPS.md#mfn-api-grpc
"""

import argparse
import os
import time
from concurrent import futures
from typing import Iterator

import grpc
import numpy as np

from mycelium_fractal_net import (
    compute_nernst_potential,
    estimate_fractal_dimension,
    run_mycelium_simulation_with_history,
    run_validation,
)
from mycelium_fractal_net.grpc.protos import mycelium_pb2, mycelium_pb2_grpc
from mycelium_fractal_net.integration import get_logger
from mycelium_fractal_net.types import SimulationConfig

logger = get_logger("grpc_server")


class MyceliumServicer(mycelium_pb2_grpc.MyceliumServiceServicer):
    """gRPC servicer implementation for MyceliumFractalNet."""

    def __init__(self, require_auth: bool = False, api_key: str = ""):
        """
        Initialize the servicer.

        Args:
            require_auth: Whether to require API key authentication.
            api_key: Expected API key for authentication.
        """
        self.require_auth = require_auth
        self.api_key = api_key
        self.start_time = time.time()
        logger.info(f"MyceliumServicer initialized (auth_required={require_auth})")

    def _check_auth(self, context: grpc.ServicerContext) -> bool:
        """
        Check authentication metadata.

        Args:
            context: gRPC servicer context.

        Returns:
            True if authenticated or auth not required.
        """
        if not self.require_auth:
            return True

        metadata = dict(context.invocation_metadata())
        client_key = metadata.get("x-api-key", "")

        if client_key != self.api_key:
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details("Invalid or missing API key")
            return False

        return True

    def HealthCheck(
        self, request: mycelium_pb2.HealthCheckRequest, context: grpc.ServicerContext
    ) -> mycelium_pb2.HealthCheckResponse:
        """Health check endpoint."""
        uptime = int(time.time() - self.start_time)
        return mycelium_pb2.HealthCheckResponse(
            status="healthy", version="4.1.0", uptime_seconds=uptime
        )

    def Validate(
        self, request: mycelium_pb2.ValidateRequest, context: grpc.ServicerContext
    ) -> mycelium_pb2.ValidateResponse:
        """Run validation cycle."""
        if not self._check_auth(context):
            return mycelium_pb2.ValidateResponse()

        try:
            logger.info(f"Validation request: seed={request.seed}, epochs={request.epochs}")

            # Run validation
            result = run_validation(
                seed=request.seed, epochs=request.epochs, grid_size=request.grid_size or 32
            )

            response = mycelium_pb2.ValidateResponse(
                loss_start=result.get("loss_start", 0.0),
                loss_final=result.get("loss_final", 0.0),
                loss_drop=result.get("loss_drop", 0.0),
                pot_min_mV=result.get("pot_min_mV", 0.0),
                pot_max_mV=result.get("pot_max_mV", 0.0),
                lyapunov_exponent=result.get("lyapunov_exponent", 0.0),
                nernst_symbolic_mV=result.get("nernst_symbolic_mV", 0.0),
            )

            logger.info(f"Validation completed: loss_drop={response.loss_drop:.3f}")
            return response

        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return mycelium_pb2.ValidateResponse()

    def Simulate(
        self, request: mycelium_pb2.SimulateRequest, context: grpc.ServicerContext
    ) -> mycelium_pb2.SimulateResponse:
        """Simulate mycelium field."""
        if not self._check_auth(context):
            return mycelium_pb2.SimulateResponse()

        try:
            logger.info(
                f"Simulation request: seed={request.seed}, "
                f"grid={request.grid_size}, steps={request.steps}"
            )

            # Create simulation config
            config = SimulationConfig(
                seed=request.seed,
                grid_size=request.grid_size or 64,
                steps=request.steps or 100,
                turing_enabled=request.turing_enabled,
                alpha=request.alpha if request.alpha > 0 else 0.1,
                gamma=request.gamma if request.gamma > 0 else 0.8,
            )

            # Run simulation
            result = run_mycelium_simulation_with_history(config)
            field_history = result.field_history

            # Calculate statistics
            field_mean = [float(np.mean(field)) for field in field_history]
            field_std = [float(np.std(field)) for field in field_history]

            # Compute fractal dimension
            binary = field_history[-1] > -0.060
            fractal_dim = estimate_fractal_dimension(binary)

            # Field stats for final state
            final_field = field_history[-1]
            field_stats = mycelium_pb2.FieldStats(
                mean=float(np.mean(final_field)),
                std=float(np.std(final_field)),
                min=float(np.min(final_field)),
                max=float(np.max(final_field)),
            )

            response = mycelium_pb2.SimulateResponse(
                field_mean=field_mean,
                field_std=field_std,
                growth_events=result.growth_events,
                fractal_dimension=fractal_dim,
                field_stats=field_stats,
            )

            logger.info(
                f"Simulation completed: growth_events={response.growth_events}, "
                f"fractal_dim={fractal_dim:.3f}"
            )
            return response

        except Exception as e:
            logger.error(f"Simulation failed: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return mycelium_pb2.SimulateResponse()

    def SimulateStream(
        self, request: mycelium_pb2.SimulateRequest, context: grpc.ServicerContext
    ) -> Iterator[mycelium_pb2.SimulationUpdate]:
        """Stream simulation updates in real-time."""
        if not self._check_auth(context):
            return

        try:
            logger.info(
                f"Streaming simulation request: seed={request.seed}, "
                f"grid={request.grid_size}, steps={request.steps}"
            )

            # Create simulation config
            config = SimulationConfig(
                seed=request.seed,
                grid_size=request.grid_size or 64,
                steps=request.steps or 100,
                turing_enabled=request.turing_enabled,
                alpha=request.alpha if request.alpha > 0 else 0.1,
                gamma=request.gamma if request.gamma > 0 else 0.8,
            )

            # Run simulation with history
            result = run_mycelium_simulation_with_history(config)
            field_history = result.field_history
            total_steps = len(field_history)

            # Stream updates for each step
            for step, field in enumerate(field_history):
                if context.is_active():
                    update = mycelium_pb2.SimulationUpdate(
                        step=step,
                        total_steps=total_steps,
                        pot_mean_mV=float(np.mean(field) * 1000),
                        pot_std_mV=float(np.std(field) * 1000),
                        growth_events=result.growth_events if step == total_steps - 1 else 0,
                        completed=(step == total_steps - 1),
                    )
                    yield update
                else:
                    logger.info("Client disconnected during streaming")
                    break

            logger.info(f"Streaming simulation completed: {total_steps} updates sent")

        except Exception as e:
            logger.error(f"Streaming simulation failed: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

    def ComputeNernst(
        self, request: mycelium_pb2.NernstRequest, context: grpc.ServicerContext
    ) -> mycelium_pb2.NernstResponse:
        """Compute Nernst potential."""
        if not self._check_auth(context):
            return mycelium_pb2.NernstResponse()

        try:
            potential = compute_nernst_potential(
                z_valence=request.z_valence,
                concentration_out_molar=request.concentration_out_molar,
                concentration_in_molar=request.concentration_in_molar,
                temperature_k=request.temperature_k,
            )

            return mycelium_pb2.NernstResponse(potential_mV=potential * 1000)

        except Exception as e:
            logger.error(f"Nernst computation failed: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return mycelium_pb2.NernstResponse()

    def AggregateFederated(
        self, request: mycelium_pb2.FederatedAggregateRequest, context: grpc.ServicerContext
    ) -> mycelium_pb2.FederatedAggregateResponse:
        """Aggregate federated learning gradients."""
        if not self._check_auth(context):
            return mycelium_pb2.FederatedAggregateResponse()

        try:
            # Import federated aggregation
            from mycelium_fractal_net.model import hierarchical_krum

            # Convert protobuf gradients to numpy arrays
            gradients = [np.array(g.values) for g in request.gradients]

            # Perform aggregation
            aggregated = hierarchical_krum(
                gradients=gradients,
                num_clusters=request.num_clusters or 10,
                byzantine_fraction=request.byzantine_fraction or 0.2,
            )

            response = mycelium_pb2.FederatedAggregateResponse(
                aggregated_gradient=aggregated.tolist(),
                num_accepted=len(gradients),
                num_rejected=0,
            )

            logger.info(f"Federated aggregation completed: {len(gradients)} gradients")
            return response

        except Exception as e:
            logger.error(f"Federated aggregation failed: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return mycelium_pb2.FederatedAggregateResponse()


def serve(port: int = 50051, max_workers: int = 10, require_auth: bool = False, api_key: str = ""):
    """
    Start the gRPC server.

    Args:
        port: Port to listen on.
        max_workers: Maximum number of thread pool workers.
        require_auth: Whether to require API key authentication.
        api_key: API key for authentication.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = MyceliumServicer(require_auth=require_auth, api_key=api_key)
    mycelium_pb2_grpc.add_MyceliumServiceServicer_to_server(servicer, server)

    server.add_insecure_port(f"[::]:{port}")
    server.start()

    logger.info(
        f"gRPC server started on port {port} "
        f"(max_workers={max_workers}, auth={require_auth})"
    )
    print(f"MyceliumFractalNet gRPC server listening on port {port}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("gRPC server shutting down...")
        server.stop(grace=5)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MyceliumFractalNet gRPC Server")
    parser.add_argument("--port", type=int, default=50051, help="Port to listen on")
    parser.add_argument("--max-workers", type=int, default=10, help="Max thread pool workers")
    parser.add_argument("--auth", action="store_true", help="Require API key authentication")
    parser.add_argument("--api-key", type=str, default="", help="API key for authentication")

    args = parser.parse_args()

    # Override with environment variables if set
    port = int(os.getenv("MFN_GRPC_PORT", args.port))
    max_workers = int(os.getenv("MFN_GRPC_MAX_WORKERS", args.max_workers))
    require_auth = os.getenv("MFN_GRPC_AUTH_REQUIRED", "false").lower() == "true" or args.auth
    api_key = os.getenv("MFN_GRPC_API_KEY", args.api_key)

    serve(port=port, max_workers=max_workers, require_auth=require_auth, api_key=api_key)


if __name__ == "__main__":
    main()
