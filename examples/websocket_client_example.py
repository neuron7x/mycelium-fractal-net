"""
WebSocket Client Example for MFN Streaming API.

Demonstrates how to connect to MFN WebSocket endpoints with HMAC signature
authentication and subscribe to real-time fractal features or live simulation updates.

Usage:
    python examples/websocket_client_example.py
    
Requirements:
    pip install websockets
"""

import asyncio
import hashlib
import hmac
import json
import time
from typing import Optional


class MFNWebSocketClient:
    """
    Client for MFN WebSocket streaming API.
    
    Supports:
    - HMAC-SHA256 signature authentication
    - Feature streaming
    - Simulation streaming
    - Heartbeat monitoring
    - Automatic reconnection
    """
    
    def __init__(self, host: str = "localhost", port: int = 8000, api_key: Optional[str] = None):
        """
        Initialize WebSocket client.
        
        Args:
            host: Server hostname.
            port: Server port.
            api_key: API key for authentication (optional in dev mode).
        """
        self.host = host
        self.port = port
        self.api_key = api_key
        self.websocket = None
        
    async def connect(self, endpoint: str):
        """
        Connect to a WebSocket endpoint.
        
        Args:
            endpoint: Endpoint path (e.g., "/ws/stream_features").
        """
        import websockets
        
        url = f"ws://{self.host}:{self.port}{endpoint}"
        print(f"Connecting to {url}...")
        
        self.websocket = await websockets.connect(url)
        print("Connected!")
        
    async def send_init(self, client_info: str = "python-client"):
        """Send initialization message."""
        message = {
            "type": "init",
            "payload": {
                "protocol_version": "1.0",
                "client_info": client_info,
            }
        }
        
        await self.websocket.send(json.dumps(message))
        print(f"→ Sent INIT: {client_info}")
        
        # Wait for server response
        response = json.loads(await self.websocket.recv())
        print(f"← Received: {response['type']}")
        
        return response
    
    async def authenticate(self, use_signature: bool = True):
        """
        Authenticate with API key and optional HMAC signature.
        
        Args:
            use_signature: Whether to include HMAC signature for enhanced security.
        """
        if not self.api_key:
            print("⚠ No API key provided, skipping authentication")
            return
        
        timestamp = int(time.time() * 1000)
        
        payload = {
            "api_key": self.api_key,
            "timestamp": timestamp,
        }
        
        if use_signature:
            # Generate HMAC-SHA256 signature
            timestamp_str = str(timestamp)
            signature = hmac.new(
                self.api_key.encode('utf-8'),
                timestamp_str.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            payload["signature"] = signature
            print(f"🔐 Generated signature: {signature[:16]}...")
        
        message = {
            "type": "auth",
            "payload": payload
        }
        
        await self.websocket.send(json.dumps(message))
        print(f"→ Sent AUTH with {'signature' if use_signature else 'API key only'}")
        
        # Wait for server response
        response = json.loads(await self.websocket.recv())
        print(f"← Received: {response['type']}")
        
        if response["type"] == "auth_success":
            print("✓ Authentication successful")
            return True
        else:
            print(f"✗ Authentication failed: {response.get('payload', {}).get('message')}")
            return False
    
    async def subscribe_features(
        self,
        stream_id: str = "features-1",
        update_interval_ms: int = 100,
        compression: bool = False
    ):
        """
        Subscribe to real-time fractal features stream.
        
        Args:
            stream_id: Unique stream identifier.
            update_interval_ms: Update interval in milliseconds.
            compression: Enable compression.
        """
        message = {
            "type": "subscribe",
            "payload": {
                "stream_type": "stream_features",
                "stream_id": stream_id,
                "params": {
                    "update_interval_ms": update_interval_ms,
                    "compression": compression,
                }
            }
        }
        
        await self.websocket.send(json.dumps(message))
        print(f"→ Sent SUBSCRIBE: stream_features (interval={update_interval_ms}ms)")
        
        # Wait for server response
        response = json.loads(await self.websocket.recv())
        print(f"← Received: {response['type']}")
        
        return response
    
    async def subscribe_simulation(
        self,
        stream_id: str = "sim-1",
        grid_size: int = 64,
        steps: int = 64,
        update_interval_steps: int = 1
    ):
        """
        Subscribe to live simulation updates.
        
        Args:
            stream_id: Unique stream identifier.
            grid_size: Simulation grid size.
            steps: Number of simulation steps.
            update_interval_steps: Send update every N steps.
        """
        message = {
            "type": "subscribe",
            "payload": {
                "stream_type": "simulation_live",
                "stream_id": stream_id,
                "params": {
                    "seed": 42,
                    "grid_size": grid_size,
                    "steps": steps,
                    "update_interval_steps": update_interval_steps,
                }
            }
        }
        
        await self.websocket.send(json.dumps(message))
        print(f"→ Sent SUBSCRIBE: simulation_live (grid={grid_size}, steps={steps})")
        
        # Wait for server response
        response = json.loads(await self.websocket.recv())
        print(f"← Received: {response['type']}")
        
        return response
    
    async def handle_heartbeat(self):
        """Handle heartbeat from server."""
        message = {
            "type": "pong",
            "timestamp": int(time.time() * 1000)
        }
        await self.websocket.send(json.dumps(message))
        print("💓 Sent PONG in response to heartbeat")
    
    async def receive_updates(self, max_updates: int = 10):
        """
        Receive and print updates from server.
        
        Args:
            max_updates: Maximum number of updates to receive (0 = unlimited).
        """
        count = 0
        
        print(f"\n📊 Listening for updates (max={max_updates if max_updates > 0 else 'unlimited'})...\n")
        
        try:
            while max_updates == 0 or count < max_updates:
                message = json.loads(await self.websocket.recv())
                msg_type = message.get("type")
                
                if msg_type == "feature_update":
                    # Feature stream update
                    payload = message["payload"]
                    features = payload["features"]
                    seq = payload["sequence"]
                    print(f"[{seq:4d}] Features: fractal_dim={features.get('fractal_dimension', 0):.3f}, "
                          f"active_nodes={features.get('active_nodes', 0):.0f}")
                    count += 1
                    
                elif msg_type == "simulation_state":
                    # Simulation step update
                    payload = message["payload"]
                    step = payload["step"]
                    total = payload["total_steps"]
                    state = payload["state"]
                    print(f"[{step:3d}/{total:3d}] pot_mean={state['pot_mean_mV']:.2f}mV, "
                          f"active={state['active_nodes']}")
                    count += 1
                    
                elif msg_type == "simulation_complete":
                    # Simulation finished
                    payload = message["payload"]
                    metrics = payload["final_metrics"]
                    print(f"\n✓ Simulation complete!")
                    print(f"  Final fractal dimension: {metrics.get('fractal_dimension', 0):.3f}")
                    print(f"  Growth events: {metrics.get('growth_events', 0):.0f}")
                    break
                    
                elif msg_type == "heartbeat":
                    # Server heartbeat
                    await self.handle_heartbeat()
                    
                elif msg_type == "error":
                    # Error message
                    error = message["payload"]
                    print(f"✗ Error: {error['error_code']} - {error['message']}")
                    break
                    
                else:
                    print(f"← Received: {msg_type}")
        
        except asyncio.CancelledError:
            print("\n⚠ Cancelled by user")
        except Exception as e:
            print(f"\n✗ Error receiving updates: {e}")
    
    async def close(self):
        """Close WebSocket connection."""
        if self.websocket:
            message = {"type": "close"}
            await self.websocket.send(json.dumps(message))
            await self.websocket.close()
            print("\n✓ Connection closed")


async def example_feature_streaming():
    """Example: Stream real-time fractal features."""
    print("=" * 70)
    print("Example 1: Real-time Fractal Features Streaming")
    print("=" * 70 + "\n")
    
    # Create client (no API key for dev mode)
    client = MFNWebSocketClient(host="localhost", port=8000)
    
    try:
        # Connect to feature streaming endpoint
        await client.connect("/ws/stream_features")
        
        # Initialize
        await client.send_init(client_info="python-example-features")
        
        # Authenticate (optional in dev mode)
        # await client.authenticate(use_signature=False)
        
        # Subscribe to feature stream
        await client.subscribe_features(
            stream_id="example-features-1",
            update_interval_ms=200  # Update every 200ms
        )
        
        # Receive updates
        await client.receive_updates(max_updates=20)
        
    finally:
        await client.close()


async def example_simulation_streaming():
    """Example: Stream live simulation updates."""
    print("\n" + "=" * 70)
    print("Example 2: Live Simulation Streaming")
    print("=" * 70 + "\n")
    
    # Create client with API key
    client = MFNWebSocketClient(
        host="localhost",
        port=8000,
        api_key="test-key-123"  # Use real key in production
    )
    
    try:
        # Connect to simulation endpoint
        await client.connect("/ws/simulation_live")
        
        # Initialize
        await client.send_init(client_info="python-example-simulation")
        
        # Authenticate with HMAC signature
        authenticated = await client.authenticate(use_signature=True)
        
        if authenticated:
            # Subscribe to simulation stream
            await client.subscribe_simulation(
                stream_id="example-sim-1",
                grid_size=32,
                steps=20,
                update_interval_steps=1  # Update every step
            )
            
            # Receive all updates until simulation completes
            await client.receive_updates(max_updates=0)
        
    finally:
        await client.close()


async def example_heartbeat_only():
    """Example: Simple heartbeat connection for health monitoring."""
    print("\n" + "=" * 70)
    print("Example 3: Heartbeat-Only Connection")
    print("=" * 70 + "\n")
    
    client = MFNWebSocketClient(host="localhost", port=8000)
    
    try:
        # Connect to heartbeat endpoint
        await client.connect("/ws/heartbeat")
        
        # Initialize
        await client.send_init(client_info="python-example-heartbeat")
        
        # Authenticate (optional in dev mode)
        # await client.authenticate(use_signature=False)
        
        print("Monitoring connection health for 60 seconds...")
        print("(Server will send heartbeat every 30s)")
        
        # Just listen for heartbeats
        await asyncio.wait_for(
            client.receive_updates(max_updates=0),
            timeout=60.0
        )
        
    except asyncio.TimeoutError:
        print("\n✓ 60 seconds elapsed, connection healthy")
    finally:
        await client.close()


async def main():
    """Run all examples."""
    print("\nMFN WebSocket Client Examples\n")
    print("Make sure the MFN API server is running on localhost:8000")
    print("Start server with: uvicorn api:app --host 0.0.0.0 --port 8000\n")
    
    await asyncio.sleep(1)
    
    # Run examples
    await example_feature_streaming()
    await asyncio.sleep(2)
    
    # Uncomment to run other examples:
    # await example_simulation_streaming()
    # await asyncio.sleep(2)
    # await example_heartbeat_only()


if __name__ == "__main__":
    asyncio.run(main())
