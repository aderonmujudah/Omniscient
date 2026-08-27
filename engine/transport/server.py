import asyncio
import json
import logging
import websockets
from typing import Set, Optional
from engine.sources.base import GazeSource
from engine.sources.recorder import _serialize_sample

logger = logging.getLogger(__name__)

class WebsocketPublisher:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server = None
        self.loop = None

    async def _handler(self, websocket, path):
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        try:
            # Just keep the connection open to send data
            async for _ in websocket:
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            logger.info(f"Client disconnected: {websocket.remote_address}")

    async def _start_server(self):
        self.server = await websockets.serve(self._handler, self.host, self.port)
        logger.info(f"WebSocket server listening on ws://{self.host}:{self.port}")

    def start(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.loop.create_task(self._start_server())

    async def _broadcast(self, message: str):
        if not self.clients:
            return
        # Create tasks for all sends
        tasks = [asyncio.create_task(client.send(message)) for client in self.clients]
        # Ignore errors from disconnected clients during broadcast
        await asyncio.gather(*tasks, return_exceptions=True)

    def publish_event(self, event_dict: dict) -> None:
        """Broadcast one InteractionEvent to every connected client.

        Samples and events share the socket. They are distinguished by `event_type`, which
        only an event carries, so a client can route on its presence.
        """
        if not self.loop or not self.clients:
            return
        asyncio.run_coroutine_threadsafe(
            self._broadcast(json.dumps(event_dict)), self.loop
        )

    def publish_sample(self, sample):
        if not self.loop or not self.clients:
            return
        message = _serialize_sample(sample)
        # Schedule the broadcast in the event loop
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self.loop)

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
