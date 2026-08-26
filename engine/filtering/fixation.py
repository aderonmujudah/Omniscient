from dataclasses import dataclass
from typing import Optional, Tuple
from collections import deque

@dataclass(frozen=True)
class Fixation:
    """A detected fixation with centroid and duration."""
    x: float
    y: float
    start_t: float
    end_t: float
    duration_s: float

class FixationDetector:
    def __init__(self, dispersion_threshold: float = 50.0, min_fixation_duration_s: float = 0.1) -> None:
        # UNTUNED: Parameters from literature (Salvucci and Goldberg 2000), not tuned against recorded human gaze data.
        self.dispersion_threshold = dispersion_threshold
        self.min_fixation_duration_s = min_fixation_duration_s
        self.window: deque = deque()
        self.is_fixing: bool = False
        self._active_centroid: Optional[Tuple[float, float]] = None

    def process(self, x: float, y: float, timestamp: float) -> Optional[Fixation]:
        """Process a new gaze point. Returns a Fixation when one ends."""
        self.window.append((x, y, timestamp))
        completed_fixation: Optional[Fixation] = None

        while len(self.window) > 0:
            xs = [pt[0] for pt in self.window]
            ys = [pt[1] for pt in self.window]
            dispersion = (max(xs) - min(xs)) + (max(ys) - min(ys))
            duration = self.window[-1][2] - self.window[0][2]

            if dispersion <= self.dispersion_threshold:
                if duration >= self.min_fixation_duration_s:
                    self.is_fixing = True
                    self._active_centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
                break
            else:
                if self.is_fixing:
                    # Emit fixation before this exceeding point
                    fix_xs = xs[:-1]
                    fix_ys = ys[:-1]
                    if len(fix_xs) > 0:
                        fix_start = self.window[0][2]
                        fix_end = self.window[-2][2]
                        fix_dur = fix_end - fix_start
                        cx = sum(fix_xs) / len(fix_xs)
                        cy = sum(fix_ys) / len(fix_ys)
                        completed_fixation = Fixation(
                            x=cx, 
                            y=cy, 
                            start_t=fix_start, 
                            end_t=fix_end, 
                            duration_s=fix_dur
                        )
                    
                    self.is_fixing = False
                    self._active_centroid = None
                    
                    # Keep only the last point to start a new window
                    last_pt = self.window[-1]
                    self.window.clear()
                    self.window.append(last_pt)
                    break
                else:
                    self.window.popleft()

        return completed_fixation

    @property
    def active_fixation_centroid(self) -> Optional[Tuple[float, float]]:
        """Returns the centroid of the current active fixation, if any."""
        return self._active_centroid

    def reset(self) -> None:
        self.window.clear()
        self.is_fixing = False
        self._active_centroid = None
