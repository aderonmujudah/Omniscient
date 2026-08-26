from typing import Optional
from engine.sources.base import GazeSample

class ExtendedClosureDetector:
    CLOSURE_MIN_S: float = 0.8
    CLOSURE_MAX_S: float = 2.0

    def __init__(self, ear_threshold: float = 0.2) -> None:
        self._ear_threshold = ear_threshold
        self._closure_start_t: Optional[float] = None
        self._latched_x: float = 0.0
        self._latched_y: float = 0.0
        self._was_closed: bool = False
        self._last_gaze_x: float = 0.0
        self._last_gaze_y: float = 0.0

    @property
    def name(self) -> str:
        return "extended_closure"

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
            
            if self.CLOSURE_MIN_S <= duration_s <= self.CLOSURE_MAX_S:
                return self.name
            return None
        
        self._was_closed = closed
        return None

    def update_gaze_position(self, x: float, y: float) -> None:
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
