"""Gesture registry — resolves enabled detectors and role assignments from the calibration profile.

The registry is the only place that knows which gesture is assigned to which role.
Nothing outside the gesture package sees gesture names.
"""

import logging
from typing import Dict, List, Optional
from engine.sources.base import GazeSample
from engine.events.gestures.base import Role, GestureEvent
from engine.events.gestures.long_blink import LongBlinkDetector
from engine.events.gestures.extended_closure import ExtendedClosureDetector
from engine.events.gestures.off_screen_glance import OffScreenGlanceDetector
from engine.events.gestures.gaze_stroke import GazeStrokeDetector
from engine.events.gestures.reserved_zone_dwell import ReservedZoneDwellDetector

logger = logging.getLogger(__name__)


def _closure_bounds(kw: dict) -> dict:
    """The EAR bounds a closure detector is built with, carrying only what was measured.

    An unmeasured bound is left out rather than defaulted here, so the detector's own value
    applies and there is one place holding it rather than two that can disagree.
    """
    return {name: kw[name] for name in ("ear_threshold", "ear_reopen")
            if kw.get(name) is not None}


_DETECTOR_FACTORIES = {
    "long_blink": lambda kw: LongBlinkDetector(
        **_closure_bounds(kw),
        **({"closure_min_s": kw["threshold_ms"] / 1000.0} if kw.get("threshold_ms") else {}),
    ),
    "extended_closure": lambda kw: ExtendedClosureDetector(**_closure_bounds(kw)),
    "off_screen_glance": lambda kw: OffScreenGlanceDetector(
        screen_w=kw["screen_w"], screen_h=kw["screen_h"]
    ),
    "gaze_stroke": lambda kw: GazeStrokeDetector(),
}


def _parse_role(name: str) -> Optional[Role]:
    """Resolve a role name as the calibration profile records it.

    The profile writes roles in lower case, per the persisted profile format, while the
    role enum is upper case. Matching them exactly would leave every role unfilled, which
    silently removes the user's ability to act rather than failing visibly.
    """
    try:
        return Role(str(name).upper())
    except ValueError:
        return None


class GestureRegistry:
    """Resolves gesture detectors to interaction roles using profile configuration."""

    def __init__(
        self,
        role_assignment: Dict[str, str],
        screen_w: int,
        screen_h: int,
        reserved_zones: Dict[str, dict],
        ear_threshold: Optional[float] = None,
        *,
        gesture_params: Dict[str, dict],
        closure_threshold_ms: Optional[float] = None,
        ear_reopen: Optional[float] = None,
    ) -> None:
        """
        Args:
            gesture_params: Per-gesture parameters measured during calibration, keyed by
                gesture name. Required rather than optional: a detector constructed
                without its measured parameters silently falls back to a generic band,
                discarding the measurement the assessment recorded.
            closure_threshold_ms: The user's measured deliberate-closure threshold. Passed as
                a measurement rather than as a named gesture's parameter, so that callers
                outside this package never need to know which detectors consume it.
            ear_threshold: The user's measured closing bound, and ear_reopen the bound a
                closure already under way must clear to be treated as ended. Both are
                properties of a person's eyelids rather than of the product, and either left
                unset leaves the detector's own default in place.
        """
        self._gesture_detectors: list = []
        self._gesture_role_map: Dict[int, Role] = {}
        self._zone_detector: Optional[ReservedZoneDwellDetector] = None

        self._gesture_params = gesture_params

        zone_roles: Dict[Role, dict] = {}
        zone_rects = {str(name).upper(): rect for name, rect in reserved_zones.items()}

        for role_str, gesture_name in role_assignment.items():
            role = _parse_role(role_str)
            if role is None:
                logger.warning("Unrecognised role %r in the profile; it stays unfilled.", role_str)
                continue

            if gesture_name in ("reserved_zone_dwell", "corner_dwell"):
                if role.value in zone_rects:
                    zone_roles[role] = zone_rects[role.value]
            elif gesture_name in _DETECTOR_FACTORIES:
                factory_kwargs = {
                    "screen_w": screen_w,
                    "screen_h": screen_h,
                    "ear_threshold": ear_threshold,
                    "ear_reopen": ear_reopen,
                }
                if closure_threshold_ms is not None:
                    factory_kwargs["threshold_ms"] = closure_threshold_ms
                factory_kwargs.update(gesture_params.get(gesture_name, {}))
                det = _DETECTOR_FACTORIES[gesture_name](factory_kwargs)
                if not det.can_fire:
                    logger.warning(
                        "Gesture %s cannot fire and was not assigned to role %s.",
                        gesture_name, role_str,
                    )
                    continue
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
