import time
from typing import Optional
from engine.sources.base import GazeSample
from .base import GestureDetector

class LongBlinkDetector(GestureDetector):
    def __init__(self, threshold_ms: float = 450.0):
        self.threshold_s = threshold_ms / 1000.0
        self.closure_start_t = None
        self.fired_this_closure = False

    def process_sample(self, sample: GazeSample) -> Optional[str]:
        is_closed = False
        if sample.ok and sample.ear:
            if sample.ear["left"] < 0.2 and sample.ear["right"] < 0.2:
                is_closed = True

        if is_closed:
            if self.closure_start_t is None:
                self.closure_start_t = sample.t
                self.fired_this_closure = False
            else:
                if not self.fired_this_closure:
                    duration = sample.t - self.closure_start_t
                    if duration >= self.threshold_s:
                        self.fired_this_closure = True
                        return "long_blink"
        else:
            self.closure_start_t = None
            self.fired_this_closure = False
            
        return None

    def reset(self) -> None:
        self.closure_start_t = None
        self.fired_this_closure = False
