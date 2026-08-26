from __future__ import annotations
import enum
from dataclasses import dataclass
from typing import Optional, Any, Dict


class EventType(enum.Enum):
    GAZE_MOVE = "GAZE_MOVE"
    FIXATION_START = "FIXATION_START"
    FIXATION_END = "FIXATION_END"
    SACCADE = "SACCADE"
    BLINK = "BLINK"
    DWELL_START = "DWELL_START"
    DWELL_PROGRESS = "DWELL_PROGRESS"
    DWELL_COMPLETE = "DWELL_COMPLETE"
    DWELL_CANCEL = "DWELL_CANCEL"
    GESTURE = "GESTURE"
    TRACKING_LOST = "TRACKING_LOST"
    TRACKING_RESUMED = "TRACKING_RESUMED"
    STATE_CHANGE = "STATE_CHANGE"
    CALIBRATION_START = "CALIBRATION_START"
    CALIBRATION_POINT = "CALIBRATION_POINT"
    CALIBRATION_COMPLETE = "CALIBRATION_COMPLETE"
    CALIBRATION_FAILED = "CALIBRATION_FAILED"


@dataclass(frozen=True)
class InteractionEvent:
    event_type: str
    timestamp: float
    x: Optional[float] = None
    y: Optional[float] = None
    duration_ms: Optional[float] = None
    start_x: Optional[float] = None
    start_y: Optional[float] = None
    end_x: Optional[float] = None
    end_y: Optional[float] = None
    zone_id: Optional[str] = None
    progress: Optional[float] = None
    role: Optional[str] = None
    from_state: Optional[str] = None
    to_state: Optional[str] = None
    point_index: Optional[int] = None
    profile_id: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict, excluding None fields."""
        result: Dict[str, Any] = {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
        }
        for field_name in ("x", "y", "duration_ms", "start_x", "start_y", "end_x", "end_y",
                          "zone_id", "progress", "role", "from_state", "to_state",
                          "point_index", "profile_id", "reason"):
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        return result
