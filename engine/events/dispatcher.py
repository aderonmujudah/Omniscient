import logging
from typing import Any, Callable, Dict, List, Optional
from engine.events.interaction import InteractionEvent

logger = logging.getLogger(__name__)

class EventDispatcher:
    """Fans out InteractionEvents to registered subscribers and WebSocket."""

    def __init__(self) -> None:
        self._subscribers: List[Callable[[InteractionEvent], None]] = []
        self._ws_broadcast: Optional[Callable[[Dict[str, Any]], None]] = None

    def subscribe(self, callback: Callable[[InteractionEvent], None]) -> None:
        self._subscribers.append(callback)

    def set_ws_broadcast(self, broadcast_fn: Callable[[Dict[str, Any]], None]) -> None:
        """Set the WebSocket broadcast function."""
        self._ws_broadcast = broadcast_fn

    def dispatch(self, event: InteractionEvent) -> None:
        """Dispatch an event to all subscribers and WebSocket."""
        event_dict = event.to_dict()
        
        for subscriber in self._subscribers:
            try:
                subscriber(event)
            except Exception:
                logger.exception("Subscriber raised an exception")
        
        if self._ws_broadcast is not None:
            try:
                self._ws_broadcast(event_dict)
            except Exception:
                logger.exception("WebSocket broadcast failed")

    def dispatch_many(self, events: List[InteractionEvent]) -> None:
        for event in events:
            self.dispatch(event)
