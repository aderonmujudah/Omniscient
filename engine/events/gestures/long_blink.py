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
        if not sample.ok: # Assuming ok=False implies closure or loss
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
