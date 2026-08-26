"""Gesture detection package.

Public API: Role, GestureEvent, GestureRegistry.
Individual detector classes are internal to this package.
"""

from engine.events.gestures.base import Role, GestureEvent
from engine.events.gestures.registry import GestureRegistry

__all__ = ["Role", "GestureEvent", "GestureRegistry"]
