"""Reserved-zone dwell gesture detector — the universal fallback.

Detects dwell in reserved screen zones assigned to interaction roles.
Always available regardless of user motor capability; requires only
the ability to direct gaze.

Dwell duration default: 0.8s. Sourced assumption from Majaranta and
Räihä, "Twenty Years of Eye Typing", 2002. Range in literature: 600–1000 ms.
"""

from typing import Optional, Dict
from engine.sources.base import GazeSample
from engine.events.gestures.base import Role


class ReservedZoneDwellDetector:

    def __init__(
        self,
        zones: Dict[Role, dict],
        screen_w: int,
        screen_h: int,
        dwell_duration_s: float = 0.8,
    ) -> None:
        self._zones = zones
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._dwell_duration_s = dwell_duration_s

        self._current_zone: Optional[Role] = None
        self._zone_entry_t: Optional[float] = None
        self._latched_x: float = 0.0
        self._latched_y: float = 0.0
        self._last_gaze_x: float = 0.0
        self._last_gaze_y: float = 0.0
        self._last_fired_role: Optional[Role] = None

    @property
    def name(self) -> str:
        return "reserved_zone_dwell"

    @property
    def requires_gaze_position(self) -> bool:
        return True

    @property
    def can_fire(self) -> bool:
        return True

    @property
    def latched_position(self) -> tuple[float, float]:
        return (self._latched_x, self._latched_y)

    @property
    def last_fired_role(self) -> Optional[Role]:
        """The role of the last fired zone dwell event."""
        return self._last_fired_role

    def process_sample(self, sample: GazeSample) -> Optional[str]:
        if not sample.ok:
            return None

        active_zone = self._get_active_zone(self._last_gaze_x, self._last_gaze_y)

        if active_zone != self._current_zone:
            self._current_zone = active_zone
            self._zone_entry_t = sample.t if active_zone is not None else None

        if self._current_zone is not None and self._zone_entry_t is not None:
            # Menu role requires a 2s dwell per spec, others use default
            duration_needed = 2.0 if self._current_zone == Role.MENU else self._dwell_duration_s
            if sample.t - self._zone_entry_t >= duration_needed:
                self._latched_x = self._last_gaze_x
                self._latched_y = self._last_gaze_y
                self._last_fired_role = self._current_zone
                self._current_zone = None
                self._zone_entry_t = None
                return self.name

        return None

    def update_gaze_position(self, x: float, y: float) -> None:
        self._last_gaze_x = x
        self._last_gaze_y = y

    def _get_active_zone(self, x: float, y: float) -> Optional[Role]:
        nx = x / self._screen_w if self._screen_w > 0 else 0.0
        ny = y / self._screen_h if self._screen_h > 0 else 0.0

        for role, rect in self._zones.items():
            rx, ry = rect["x"], rect["y"]
            rw, rh = rect["w"], rect["h"]
            if rx <= nx <= rx + rw and ry <= ny <= ry + rh:
                return role
        return None

    def reset(self) -> None:
        self._current_zone = None
        self._zone_entry_t = None
        self._latched_x = 0.0
        self._latched_y = 0.0
        self._last_fired_role = None
