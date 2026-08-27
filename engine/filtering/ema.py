import math
from typing import Optional, Tuple

class EMAFilter:
    """1D Exponential Moving Average filter, time-aware."""
    def __init__(self, rate: float, cutoff: float = 1.0) -> None:
        self.rate = rate
        self.cutoff = cutoff
        self.prev_timestamp: Optional[float] = None
        self.val_filt_prev: Optional[float] = None

    def _alpha(self, rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)

    def __call__(self, val: float, timestamp: float) -> float:
        if self.prev_timestamp is None or self.val_filt_prev is None:
            self.prev_timestamp = timestamp
            self.val_filt_prev = val
            return val

        dt = timestamp - self.prev_timestamp
        rate = 1.0 / dt if dt > 0.0 else self.rate

        alpha = self._alpha(rate, self.cutoff)
        filtered_val = alpha * val + (1.0 - alpha) * self.val_filt_prev
        
        self.val_filt_prev = filtered_val
        self.prev_timestamp = timestamp
        return filtered_val

    def reset(self) -> None:
        self.prev_timestamp = None
        self.val_filt_prev = None

class EMAFilter2D:
    """2D Exponential Moving Average filter, time-aware, for gaze cascade."""
    def __init__(self, rate: float, cutoff: float = 5.0) -> None:
        self.rate = rate
        self.cutoff = cutoff
        self.prev_timestamp: Optional[float] = None
        self.x_filt_prev: Optional[float] = None
        self.y_filt_prev: Optional[float] = None

    def _alpha(self, rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, y: float, timestamp: float) -> Tuple[float, float]:
        if self.prev_timestamp is None or self.x_filt_prev is None or self.y_filt_prev is None:
            self.prev_timestamp = timestamp
            self.x_filt_prev = x
            self.y_filt_prev = y
            return x, y

        dt = timestamp - self.prev_timestamp
        rate = 1.0 / dt if dt > 0.0 else self.rate

        alpha = self._alpha(rate, self.cutoff)
        filtered_x = alpha * x + (1.0 - alpha) * self.x_filt_prev
        filtered_y = alpha * y + (1.0 - alpha) * self.y_filt_prev
        
        self.x_filt_prev = filtered_x
        self.y_filt_prev = filtered_y
        self.prev_timestamp = timestamp
        return filtered_x, filtered_y

    def reset(self) -> None:
        self.prev_timestamp = None
        self.x_filt_prev = None
        self.y_filt_prev = None
