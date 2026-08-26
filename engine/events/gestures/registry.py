"""Gesture registry — resolves enabled detectors and role assignments from the calibration profile.

The registry is the only place that knows which gesture is assigned to which role.
Nothing outside the gesture package sees gesture names.
"""

from typing import Dict, List, Optional
from engine.sources.base import GazeSample
from engine.events.gestures.base import Role, GestureEvent
from engine.events.gestures.long_blink import LongBlinkDetector
from engine.events.gestures.extended_closure import ExtendedClosureDetector
from engine.events.gestures.off_screen_glance import OffScreenGlanceDetector
from engine.events.gestures.smooth_pursuit import SmoothPursuitDetector
from engine.events.gestures.gaze_stroke import GazeStrokeDetector
from engine.events.gestures.reserved_zone_dwell import ReservedZoneDwellDetector


_DETECTOR_FACTORIES = {
    "long_blink": lambda kw: LongBlinkDetector(ear_threshold=kw.get("ear_threshold", 0.2)),
    "extended_closure": lambda kw: ExtendedClosureDetector(ear_threshold=kw.get("ear_threshold", 0.2)),
    "off_screen_glance": lambda kw: OffScreenGlanceDetector(
        screen_w=kw["screen_w"], screen_h=kw["screen_h"]
    ),
    "smooth_pursuit": lambda kw: SmoothPursuitDetector(),
    "gaze_stroke": lambda kw: GazeStrokeDetector(),
}


class GestureRegistry:
    """Resolves gesture detectors to interaction roles using profile configuration."""

    def __init__(
        self,
        role_assignment: Dict[str, str],
        screen_w: int,
        screen_h: int,
        reserved_zones: Dict[str, dict],
        ear_threshold: float = 0.2,
    ) -> None:
        self._gesture_detectors: list = []
        self._gesture_role_map: Dict[int, Role] = {}
        self._zone_detector: Optional[ReservedZoneDwellDetector] = None

        factory_kwargs = {
            "screen_w": screen_w,
            "screen_h": screen_h,
            "ear_threshold": ear_threshold,
        }

        zone_roles: Dict[Role, dict] = {}

        for role_str, gesture_name in role_assignment.items():
            try:
                role = Role(role_str)
            except ValueError:
                continue

            if gesture_name in ("reserved_zone_dwell", "corner_dwell"):
                if role_str in reserved_zones:
                    zone_roles[role] = reserved_zones[role_str]
            elif gesture_name in _DETECTOR_FACTORIES:
                det = _DETECTOR_FACTORIES[gesture_name](factory_kwargs)
                det_id = id(det)
                self._gesture_detectors.append(det)
                self._gesture_role_map[det_id] = role

        if zone_roles:
            self._zone_detector = ReservedZoneDwellDetector(
                zones=zone_roles,
                screen_w=screen_w,
                screen_h=screen_h,
            )

    def process_sample(
        self, sample: GazeSample, gaze_x: float, gaze_y: float
    ) -> List[GestureEvent]:
        """Process a sample through all active detectors.

        Args:
            sample: The current gaze sample.
            gaze_x: Calibrated horizontal screen coordinate (pixels).
            gaze_y: Calibrated vertical screen coordinate (pixels).

        Returns:
            List of GestureEvents (typically 0 or 1).
        """
        events: List[GestureEvent] = []

        for det in self._gesture_detectors:
            det.update_gaze_position(gaze_x, gaze_y)
            result = det.process_sample(sample)
            if result is not None:
                lx, ly = det.latched_position
                role = self._gesture_role_map[id(det)]
                events.append(GestureEvent(
                    role=role, timestamp=sample.t, gaze_x=lx, gaze_y=ly
                ))

        if self._zone_detector is not None:
            self._zone_detector.update_gaze_position(gaze_x, gaze_y)
            result = self._zone_detector.process_sample(sample)
            if result is not None:
                lx, ly = self._zone_detector.latched_position
                fired_role = self._zone_detector.last_fired_role
                if fired_role is not None:
                    events.append(GestureEvent(
                        role=fired_role, timestamp=sample.t, gaze_x=lx, gaze_y=ly
                    ))

        return events

    def reset(self) -> None:
        for det in self._gesture_detectors:
            det.reset()
        if self._zone_detector is not None:
            self._zone_detector.reset()
