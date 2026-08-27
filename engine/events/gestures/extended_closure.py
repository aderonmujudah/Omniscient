from typing import Optional
from engine.sources.base import GazeSample

class ExtendedClosureDetector:
    # Bounds moved from [0.8, 2.0] on 2026-08-27 to stay disjoint from the long-blink band,
    # which now reaches 3.0 s against a measured deliberate closure of 1964 to 2600 ms. Under
    # the former bounds both detectors accepted the same closure and fired on it together.
    #
    # UNMEASURED. No closure longer than the cued two-second hold was recorded, so only the
    # lower edge of this band derives from an observation of the subject.
    CLOSURE_MIN_S: float = 3.1
    CLOSURE_MAX_S: float = 6.0

    # Held above the closing bound for the reason given on LongBlinkDetector: a lone bound is
    # crossed repeatedly by the EAR of a closure that is genuinely held.
    DEFAULT_EAR_REOPEN: float = 0.24

    # Measured on the target subject; see LongBlinkDetector.
    DEFAULT_EAR_CLOSE: float = 0.18

    def __init__(self, ear_threshold: float = DEFAULT_EAR_CLOSE,
                 ear_reopen: float = DEFAULT_EAR_REOPEN) -> None:
        self._ear_threshold = ear_threshold
        self._ear_reopen = max(ear_reopen, ear_threshold)
        self._closure_start_t: Optional[float] = None
        self._latched_x: float = 0.0
        self._latched_y: float = 0.0
        self._was_closed: bool = False
        import collections
        self._gaze_history = collections.deque(maxlen=4)

    @property
    def name(self) -> str:
        return "extended_closure"

    @property
    def requires_gaze_position(self) -> bool:
        return False

    @property
    def can_fire(self) -> bool:
        return self.CLOSURE_MIN_S < self.CLOSURE_MAX_S

    @property
    def latched_position(self) -> tuple[float, float]:
        return (self._latched_x, self._latched_y)

    def process_sample(self, sample: GazeSample) -> Optional[str]:
        closed = self._is_closed(sample)
        
        if closed and not self._was_closed:
            self._closure_start_t = sample.t
            if self._gaze_history:
                self._latched_x, self._latched_y = self._gaze_history[0]
            else:
                self._latched_x, self._latched_y = 0.0, 0.0
        
        if not closed and self._was_closed and self._closure_start_t is not None:
            duration_s = sample.t - self._closure_start_t
            self._closure_start_t = None
            self._was_closed = False
            
            if self.CLOSURE_MIN_S <= duration_s <= self.CLOSURE_MAX_S:
                return self.name
            return None
        
        self._was_closed = closed
        return None

    def update_gaze_position(self, x: float, y: float) -> None:
        self._gaze_history.append((x, y))

    def _is_closed(self, sample: GazeSample) -> bool:
        if not sample.ok or sample.ear is None:
            return False
        # A closure under way is held open to the reopen bound, so the eye must be
        # demonstrably open again to end it rather than merely ambiguous.
        bound = self._ear_reopen if self._was_closed else self._ear_threshold
        return sample.ear["left"] < bound and sample.ear["right"] < bound

    def reset(self) -> None:
        self._closure_start_t = None
        self._was_closed = False
        self._latched_x = 0.0
        self._latched_y = 0.0
        self._gaze_history.clear()
