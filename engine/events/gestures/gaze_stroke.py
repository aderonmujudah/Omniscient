from typing import Optional, List, Tuple
from engine.sources.base import GazeSample

class GazeStrokeDetector:
    def __init__(
        self,
        min_displacement_px: float = 200.0,
        max_duration_s: float = 0.5,
    ) -> None:
        """
        A stroke is a displacement of at least min_displacement_px completed within
        max_duration_s. Direction is not discriminated, so the detector reports a single
        undifferentiated stroke rather than one gesture per direction.
        """
        self._min_displacement_px = min_displacement_px
        self._max_duration_s = max_duration_s
        
        self._history: List[Tuple[float, float, float]] = []
        self._latched_x: float = 0.0
        self._latched_y: float = 0.0
        self._last_gaze_x: float = 0.0
        self._last_gaze_y: float = 0.0

    @property
    def name(self) -> str:
        return "gaze_stroke"

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
            
        self._history.append((sample.t, self._last_gaze_x, self._last_gaze_y))
        
        while self._history and self._history[0][0] < sample.t - self._max_duration_s:
            self._history.pop(0)
            
        if len(self._history) < 2:
            return None
            
        dx = self._history[-1][1] - self._history[0][1]
        dy = self._history[-1][2] - self._history[0][2]
        dist = (dx**2 + dy**2)**0.5
        
        if dist >= self._min_displacement_px:
            self._latched_x = self._last_gaze_x
            self._latched_y = self._last_gaze_y
            self._history.clear()
            return self.name
            
        return None

    def update_gaze_position(self, x: float, y: float) -> None:
        self._last_gaze_x = x
        self._last_gaze_y = y

    def reset(self) -> None:
        self._history.clear()
        self._latched_x = 0.0
        self._latched_y = 0.0
