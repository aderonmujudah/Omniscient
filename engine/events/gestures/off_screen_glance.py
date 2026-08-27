"""Off-screen glance gesture detector.

Detects a deliberate look off the calibrated screen area and back.
This is a gaze-position gesture that does not depend on eye closure,
satisfying the requirement for at least one non-closure candidate.

Conceptual source: Drewes and Schmidt, "Interacting with the Computer
Using Gaze Gestures", INTERACT 2007.

Parameters min_off_duration_s and max_return_window_s are UNTUNED
pending recorded human gaze data.
"""

from typing import Optional
from engine.sources.base import GazeSample


class OffScreenGlanceDetector:

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        min_off_duration_s: float = 0.15,
        max_return_window_s: float = 1.0,
        margin_fraction: float = 0.05,
    ) -> None:
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._min_off_duration_s = min_off_duration_s
        self._max_return_window_s = max_return_window_s

        self._left_bound = -margin_fraction * screen_w
        self._right_bound = (1.0 + margin_fraction) * screen_w
        self._top_bound = -margin_fraction * screen_h
        self._bottom_bound = (1.0 + margin_fraction) * screen_h

        self._off_screen_start_t: Optional[float] = None
        self._was_off_screen: bool = False
        self._latched_x: float = 0.0
        self._latched_y: float = 0.0
        self._last_gaze_x: float = 0.0
        self._last_gaze_y: float = 0.0

    @property
    def name(self) -> str:
        return "off_screen_glance"

    @property
    def requires_gaze_position(self) -> bool:
        return True

    @property
    def can_fire(self) -> bool:
        return True

    @property
    def latched_position(self) -> tuple[float, float]:
        return (self._latched_x, self._latched_y)

    def process_sample(self, sample: GazeSample) -> Optional[str]:
        if not sample.ok:
            return None

        off_screen = self._is_off_screen(self._last_gaze_x, self._last_gaze_y)

        if off_screen and not self._was_off_screen:
            # Gaze just left the screen — record departure time and latch position
            self._off_screen_start_t = sample.t
            self._latched_x = self._last_gaze_x
            self._latched_y = self._last_gaze_y

        elif not off_screen and self._was_off_screen:
            # Gaze returned to screen — check if it qualifies as a glance
            if self._off_screen_start_t is not None:
                duration_s = sample.t - self._off_screen_start_t
                self._off_screen_start_t = None

                if self._min_off_duration_s <= duration_s <= self._max_return_window_s:
                    self._was_off_screen = False
                    return self.name

        elif off_screen and self._was_off_screen:
            # Still off screen — check for timeout
            if self._off_screen_start_t is not None:
                if sample.t - self._off_screen_start_t > self._max_return_window_s:
                    self._off_screen_start_t = None

        self._was_off_screen = off_screen
        return None

    def update_gaze_position(self, x: float, y: float) -> None:
        self._last_gaze_x = x
        self._last_gaze_y = y

    def _is_off_screen(self, x: float, y: float) -> bool:
        return (x < self._left_bound or x > self._right_bound or
                y < self._top_bound or y > self._bottom_bound)

    def reset(self) -> None:
        self._off_screen_start_t = None
        self._was_off_screen = False
        self._latched_x = 0.0
        self._latched_y = 0.0
