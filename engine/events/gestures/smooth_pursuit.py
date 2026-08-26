from typing import Optional, List, Tuple
from engine.sources.base import GazeSample

class SmoothPursuitDetector:
    def __init__(
        self,
        correlation_threshold: float = 0.8,
        window_duration_s: float = 1.0,
        min_velocity_px_s: float = 50.0
    ) -> None:
        self._correlation_threshold = correlation_threshold
        self._window_duration_s = window_duration_s
        self._min_velocity_px_s = min_velocity_px_s
        self._history: List[Tuple[float, float, float]] = []
        self._latched_x: float = 0.0
        self._latched_y: float = 0.0
        self._last_gaze_x: float = 0.0
        self._last_gaze_y: float = 0.0

    @property
    def name(self) -> str:
        return "smooth_pursuit"

    @property
    def latched_position(self) -> tuple[float, float]:
        return (self._latched_x, self._latched_y)

    def process_sample(self, sample: GazeSample) -> Optional[str]:
        if not sample.ok:
            return None
            
        self._history.append((sample.t, self._last_gaze_x, self._last_gaze_y))
        
        while self._history and self._history[0][0] < sample.t - self._window_duration_s:
            self._history.pop(0)
            
        if len(self._history) < 2:
            return None
            
        dt = self._history[-1][0] - self._history[0][0]
        if dt < self._window_duration_s * 0.8:
            return None
            
        dx = self._history[-1][1] - self._history[0][1]
        dy = self._history[-1][2] - self._history[0][2]
        dist = (dx**2 + dy**2)**0.5
        vel = dist / dt if dt > 0 else 0.0
        
        if vel > self._min_velocity_px_s:
            # Requires a target to correlate against, which is UNTUNED/missing right now.
            pass
            
        return None

    def update_gaze_position(self, x: float, y: float) -> None:
        self._last_gaze_x = x
        self._last_gaze_y = y

    def reset(self) -> None:
        self._history.clear()
        self._latched_x = 0.0
        self._latched_y = 0.0
