import math
from typing import Tuple, Optional

class OneEuroFilter:
    def __init__(self, rate: float, min_cutoff: float = 0.01, beta: float = 0.005, d_cutoff: float = 1.0) -> None:
        self.rate = rate
        # Tuned against held-fixation recording (recordings/s2b/04_deliberate_held_1080p.jsonl)
        # and saccade recordings to eliminate jitter while minimizing lag.
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.prev_timestamp: Optional[float] = None
        
        self.prev_x: Optional[float] = None
        self.prev_y: Optional[float] = None
        
        self.x_filt_prev: Optional[float] = None
        self.dx_filt_prev: Optional[float] = None
        self.y_filt_prev: Optional[float] = None
        self.dy_filt_prev: Optional[float] = None

    def _alpha(self, rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)

    def _lowpass(self, x: float, alpha: float, prev: Optional[float]) -> float:
        if prev is None:
            return x
        return alpha * x + (1.0 - alpha) * prev

    def __call__(self, x: float, y: float, timestamp: float) -> Tuple[float, float]:
        if self.prev_timestamp is None:
            self.prev_timestamp = timestamp
            self.prev_x = x
            self.prev_y = y
            self.x_filt_prev = x
            self.y_filt_prev = y
            self.dx_filt_prev = 0.0
            self.dy_filt_prev = 0.0
            return x, y

        dt = timestamp - self.prev_timestamp
        if dt > 0.0:
            rate = 1.0 / dt
        else:
            rate = self.rate

        if self.prev_x is None or self.prev_y is None:
            dx = 0.0
            dy = 0.0
        else:
            dx = (x - self.prev_x) * rate
            dy = (y - self.prev_y) * rate

        alpha_d = self._alpha(rate, self.d_cutoff)
        edx = self._lowpass(dx, alpha_d, self.dx_filt_prev)
        edy = self._lowpass(dy, alpha_d, self.dy_filt_prev)
        self.dx_filt_prev = edx
        self.dy_filt_prev = edy

        cutoff_x = self.min_cutoff + self.beta * abs(edx)
        cutoff_y = self.min_cutoff + self.beta * abs(edy)

        alpha_x = self._alpha(rate, cutoff_x)
        alpha_y = self._alpha(rate, cutoff_y)

        filtered_x = self._lowpass(x, alpha_x, self.x_filt_prev)
        filtered_y = self._lowpass(y, alpha_y, self.y_filt_prev)
        
        self.x_filt_prev = filtered_x
        self.y_filt_prev = filtered_y

        self.prev_timestamp = timestamp
        self.prev_x = x
        self.prev_y = y

        return filtered_x, filtered_y

    def reset(self) -> None:
        self.prev_timestamp = None
        self.prev_x = None
        self.prev_y = None
        self.x_filt_prev = None
        self.dx_filt_prev = None
        self.y_filt_prev = None
        self.dy_filt_prev = None
