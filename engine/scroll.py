from typing import Dict, Optional

from engine.input.base import InputBackend

# Fraction of screen height at the top and bottom that scrolls, per the interaction spec.
ZONE_FRACTION = 0.08

# How long the gaze must rest in a zone before scrolling begins, in seconds. Long enough that
# glancing across the edge of the screen does not move the content under the user.
ARM_TIME_S = 0.4

# Scroll units per second at the extreme edge of a zone. The rate ramps from zero at the inner
# boundary to this value at the screen edge.
MAX_RATE_UNITS_PER_S = 1000.0


class ScrollController:
    """Edge-zone scrolling.

    The scroll bands span the full width of the screen, so without an exclusion they cover every
    reserved zone: the corners hold ENGAGE, CANCEL and MENU, and MENU is the recovery hatch that a
    user with a failed calibration relies on. Scrolling the content out from under someone while
    they dwell to reach that hatch is the failure this exclusion exists to prevent.
    """

    def __init__(self, screen_h: int, input_backend: InputBackend,
                 screen_w: int = 0, reserved_zones: Optional[Dict[str, Dict[str, float]]] = None):
        self.screen_h = screen_h
        self.screen_w = screen_w
        self.input_backend = input_backend
        self.zone_h = int(screen_h * ZONE_FRACTION)
        self.reserved_zones = reserved_zones or {}

        self.active_zone = None
        self.zone_entry_time = None
        self.last_scroll_time = None

    def _in_reserved_zone(self, x: float, y: float) -> bool:
        if not self.reserved_zones or self.screen_w <= 0:
            return False
        for zone in self.reserved_zones.values():
            zx = zone["x"] * self.screen_w
            zy = zone["y"] * self.screen_h
            if zx <= x <= zx + zone["w"] * self.screen_w and zy <= y <= zy + zone["h"] * self.screen_h:
                return True
        return False

    def _reset(self):
        self.active_zone = None
        self.zone_entry_time = None
        self.last_scroll_time = None

    def update(self, x: float, y: float, timestamp: float, paused: bool):
        if not self.input_backend or paused or self._in_reserved_zone(x, y):
            self._reset()
            return

        current_zone = None
        proximity = 0.0

        if y < self.zone_h:
            current_zone = "top"
            proximity = 1.0 - (y / self.zone_h)
        elif y > self.screen_h - self.zone_h:
            current_zone = "bottom"
            proximity = 1.0 - ((self.screen_h - y) / self.zone_h)

        if current_zone != self.active_zone:
            self.active_zone = current_zone
            self.zone_entry_time = timestamp if current_zone else None
            self.last_scroll_time = None

        if not self.active_zone or self.zone_entry_time is None:
            return
        if timestamp - self.zone_entry_time < ARM_TIME_S:
            return

        if self.last_scroll_time is None:
            self.last_scroll_time = timestamp
            return

        dt = timestamp - self.last_scroll_time
        if dt <= 0:
            return
        dy = proximity * MAX_RATE_UNITS_PER_S * dt
        self.input_backend.scroll(x, y, dy if self.active_zone == "top" else -dy)
        self.last_scroll_time = timestamp
