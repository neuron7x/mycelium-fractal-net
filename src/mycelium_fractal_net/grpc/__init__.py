"""
gRPC API for MyceliumFractalNet.

Provides high-throughput gRPC services for features extraction,
simulation, and validation with streaming support.

Services:
    - MFNFeaturesService: Feature extraction (unary + streaming)
    - MFNSimulationService: Field simulation (unary + streaming)
    - MFNValidationService: Pattern validation
    
Reference: docs/MFN_GRPC_SPEC.md
"""

from .client import MFNClient
from .server import serve

__all__ = [
    "MFNClient",
    "serve",
]
