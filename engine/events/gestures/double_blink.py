from typing import Optional
from engine.sources.base import GazeSample

class DoubleBlinkDetector:
    def __init__(self, interval_max_s: float = 0.5, ear_threshold: float = 0.18, ear_reopen: float = 0.24) -> None:
        self.interval_max_s = interval_max_s
        self._ear_threshold = ear_threshold
        self._ear_reopen = max(ear_reopen, ear_threshold)
        
        self._was_closed = False
        self._last_reopen_t: Optional[float] = None
        
        self._latched_x: float = 0.0
        self._latched_y: float = 0.0
        import collections
        self._gaze_history = collections.deque(maxlen=4)

    @property
    def name(self) -> str:
        return "double_blink"

    @property
    def requires_gaze_position(self) -> bool:
        return False

    @property
    def can_fire(self) -> bool:
        return True

    @property
    def latched_position(self) -> tuple[float, float]:
        return (self._latched_x, self._latched_y)

    def process_sample(self, sample: GazeSample) -> Optional[str]:
        closed = self._is_closed(sample)

        if closed and not self._was_closed:
            # Blink onset
            if self._gaze_history:
                self._latched_x, self._latched_y = self._gaze_history[-1]
            else:
                self._latched_x, self._latched_y = 0.0, 0.0
                
            if self._last_reopen_t is not None:
                if sample.t - self._last_reopen_t <= self.interval_max_s:
                    self._last_reopen_t = None
                    self._was_closed = True
                    return self.name

        if not closed and self._was_closed:
            self._last_reopen_t = sample.t

        self._was_closed = closed
        return None

    def update_gaze_position(self, x: float, y: float) -> None:
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
        self._was_closed = False
        self._last_reopen_t = None
        self._latched_x = 0.0
        self._latched_y = 0.0
        self._gaze_history.clear()
