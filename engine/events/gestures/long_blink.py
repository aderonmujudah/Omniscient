"""Deliberate long blink detector.

A closure is classified as a long blink when its duration falls inside a band whose
lower bound is the per-user threshold measured during calibration. The band is closed
at both ends, so a closure is classified only once the eyes reopen: until then a longer
closure is still possible and would belong to a different gesture.

Source: Schiffman, H.R. (2001). Sensation and Perception. Involuntary blinks fall well
below the lower bound; the bound itself is measured per user rather than assumed.
"""

from typing import Optional
from engine.sources.base import GazeSample

# Upper bound of the long-blink band in seconds. This is the boundary with an extended
# closure, not a tuned value: a closure longer than this is a different gesture.
DEFAULT_CLOSURE_MAX_S = 0.8

# Lower bound used when no per-user threshold has been measured. UNTUNED.
DEFAULT_CLOSURE_MIN_S = 0.3


class LongBlinkDetector:

    def __init__(
        self,
        ear_threshold: float = 0.2,
        closure_min_s: float = DEFAULT_CLOSURE_MIN_S,
        closure_max_s: float = DEFAULT_CLOSURE_MAX_S,
    ) -> None:
        self._ear_threshold = ear_threshold
        self._closure_min_s = closure_min_s
        self._closure_max_s = closure_max_s
        self._closure_start_t: Optional[float] = None
        self._latched_x: float = 0.0
        self._latched_y: float = 0.0
        self._was_closed: bool = False
        self._last_gaze_x: float = 0.0
        self._last_gaze_y: float = 0.0

    @property
    def name(self) -> str:
        return "long_blink"

    @property
    def closure_min_s(self) -> float:
        return self._closure_min_s

    @property
    def closure_max_s(self) -> float:
        return self._closure_max_s

    @property
    def requires_gaze_position(self) -> bool:
        return False

    @property
    def can_fire(self) -> bool:
        """False when the measured threshold leaves no room below the extended-closure
        boundary, in which case this user's long blink cannot be distinguished."""
        return self._closure_min_s < self._closure_max_s

    @property
    def latched_position(self) -> tuple[float, float]:
        return (self._latched_x, self._latched_y)

    def process_sample(self, sample: GazeSample) -> Optional[str]:
        closed = self._is_closed(sample)

        if closed and not self._was_closed:
            self._closure_start_t = sample.t
            self._latched_x = self._last_gaze_x
            self._latched_y = self._last_gaze_y

        if not closed and self._was_closed and self._closure_start_t is not None:
            duration_s = sample.t - self._closure_start_t
            self._closure_start_t = None
            self._was_closed = False

            if self._closure_min_s <= duration_s <= self._closure_max_s:
                return self.name
            return None

        self._was_closed = closed
        return None

    def update_gaze_position(self, x: float, y: float) -> None:
        """Update the last known gaze position from calibrated coordinates."""
        self._last_gaze_x = x
        self._last_gaze_y = y

    def _is_closed(self, sample: GazeSample) -> bool:
        if not sample.ok or sample.ear is None:
            return False
        return sample.ear["left"] < self._ear_threshold and sample.ear["right"] < self._ear_threshold

    def reset(self) -> None:
        self._closure_start_t = None
        self._was_closed = False
        self._latched_x = 0.0
        self._latched_y = 0.0
