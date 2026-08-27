import asyncio
import json
import logging
import websockets
import jsonschema
from pathlib import Path
from typing import Set, Optional
from engine.sources.recorder import _serialize_sample
from engine.input.base import InputBackend

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent.parent.parent / "protocol" / "schema.json"
with open(SCHEMA_PATH, "r") as f:
    PROTOCOL_SCHEMA = json.load(f)

class WebsocketPublisher:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, input_backend: Optional[InputBackend] = None):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server = None
        self.loop = None
        self.input_backend = input_backend

    async def _handler(self, websocket, path):
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    jsonschema.validate(instance=data, schema=PROTOCOL_SCHEMA)
                    
                    if "action" in data and self.input_backend:
                        action = data["action"]
                        if action == "CLICK":
                            self.input_backend.click(data["x"], data["y"], data["button"], data["count"])
                        elif action == "DRAG":
                            self.input_backend.drag(data["phase"], data["x"], data["y"])
                        elif action == "SCROLL":
                            self.input_backend.scroll(data["x"], data["y"], data["dy"])
                        elif action == "KEY":
                            self.input_backend.key(data.get("text"), data.get("keycode"))
                except json.JSONDecodeError:
                    logger.error("Malformed JSON received from client")
                except jsonschema.ValidationError as e:
                    logger.error(f"ActionRequest schema validation failed: {e.message}")
                except Exception as e:
                    logger.error(f"Error processing ActionRequest: {e}")
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
        tasks = [asyncio.create_task(client.send(message)) for client in self.clients]
        await asyncio.gather(*tasks, return_exceptions=True)

    def publish_event(self, event_dict: dict) -> None:
        if not self.loop or not self.clients:
            return
        asyncio.run_coroutine_threadsafe(
            self._broadcast(json.dumps(event_dict)), self.loop
        )

    def publish_sample(self, sample):
        if not self.loop or not self.clients:
            return
        message = _serialize_sample(sample)
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self.loop)

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
