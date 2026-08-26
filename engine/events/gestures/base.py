import enum
from typing import Protocol, Optional
from dataclasses import dataclass
from engine.sources.base import GazeSample

class Role(enum.Enum):
    """Interaction roles resolved at runtime from gesture assignments."""
    ENGAGE = "ENGAGE"
    CANCEL = "CANCEL"
    MENU = "MENU"

@dataclass(frozen=True)
class GestureEvent:
    """Emitted when a gesture fires, carrying only the assigned role."""
    role: Role
    timestamp: float
    gaze_x: float
    gaze_y: float

class GestureDetector(Protocol):
    """Protocol for all gesture detectors."""

    @property
    def name(self) -> str:
        """Unique identifier for this gesture type."""
        ...

    def process_sample(self, sample: GazeSample) -> Optional[str]:
        """Process a gaze sample. Returns gesture name if detected, else None."""
        ...

    def reset(self) -> None:
        """Reset internal state."""
        ...

    @property
    def latched_position(self) -> tuple[float, float]:
        """Returns the gaze position to use for event reporting.
        
        For closure-based gestures, this is the position at closure onset.
        For gaze-position gestures, this is the position at detection time.
        """
        ...
        
    def update_gaze_position(self, x: float, y: float) -> None:
        """Update the last known gaze position from calibrated coordinates."""
        ...
