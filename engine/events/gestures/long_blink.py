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
#
# Raised from 0.8 s on 2026-08-27. Measured on the target subject, a deliberate long blink runs
# 1964 to 2600 ms while ordinary blinking ceils at 836 ms, so the former bound sat below every
# deliberate closure the subject produced and above every threshold that would separate the two
# populations: can_fire went False and the gesture was withheld from a subject who performs it
# reliably. Measured from recordings 02 and 06, n=44 ordinary blinks and n=13 cued closures.
DEFAULT_CLOSURE_MAX_S = 3.1
# Closing bound. Measured on the target subject over four minutes of ordinary blinking and
# thirteen cued two-second closures: at 0.18 the two populations separate by 452 ms, against
# 174 ms at the 0.2 of Soukupova and Cech (2016), whose figure is a population mean rather
# than a per-user value. INTERIM: this belongs in the profile, not in a default.
DEFAULT_EAR_CLOSE = 0.18

# EAR above which a closure already under way is treated as ended. Held above the closing bound
# so that a closure is not ended by the EAR wandering across a single value: measured on the
# target subject, a held closure sits between 0.14 and 0.21 and crossed a lone 0.2 bound
# repeatedly, splitting six of thirteen cued holds into as many as six runs.
DEFAULT_EAR_REOPEN = 0.24

# Lower bound used when no per-user threshold has been measured. UNTUNED.
DEFAULT_CLOSURE_MIN_S = 0.3


class LongBlinkDetector:

    def __init__(
        self,
        ear_threshold: float = DEFAULT_EAR_CLOSE,
        closure_min_s: float = DEFAULT_CLOSURE_MIN_S,
        closure_max_s: float = DEFAULT_CLOSURE_MAX_S,
        ear_reopen: float = DEFAULT_EAR_REOPEN,
    ) -> None:
        self._ear_threshold = ear_threshold
        self._ear_reopen = max(ear_reopen, ear_threshold)
        self._closure_min_s = closure_min_s
        self._closure_max_s = closure_max_s
        self._closure_start_t: Optional[float] = None
        self._latched_x: float = 0.0
        self._latched_y: float = 0.0
        self._was_closed: bool = False
        self.in_fixation: bool = True
        import collections
        self._gaze_history = collections.deque(maxlen=4)

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
            if not self.in_fixation:
                closed = False
            else:
                self._closure_start_t = sample.t
                if self._gaze_history:
                    self._latched_x, self._latched_y = self._gaze_history[-1]
                else:
                    self._latched_x, self._latched_y = 0.0, 0.0

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
        self._gaze_history.append((x, y))

    def _is_closed(self, sample: GazeSample) -> bool:
        if not sample.ok:
            return False
            
        bound = self._ear_reopen if self._was_closed else self._ear_threshold
        
        if sample.blink_score is not None:
            return sample.blink_score > max(0.5, 1.0 - bound)
            
        if sample.ear is None:
            return False
        return sample.ear["left"] < bound and sample.ear["right"] < bound

    def reset(self) -> None:
        self._closure_start_t = None
        self._was_closed = False
        self._latched_x = 0.0
        self._latched_y = 0.0
        self._gaze_history.clear()
