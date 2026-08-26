import time
from typing import Optional
from engine.sources.base import GazeSample
from .base import GestureDetector

class ExtendedClosureDetector(GestureDetector):
    def __init__(self, threshold_ms: float = 800.0):
        self.threshold_s = threshold_ms / 1000.0
        self.closure_start_t = None
        self.fired_this_closure = False

    def process_sample(self, sample: GazeSample) -> Optional[str]:
        if not sample.ok:
            if self.closure_start_t is None:
                self.closure_start_t = sample.t
                self.fired_this_closure = False
            else:
                if not self.fired_this_closure:
                    duration = sample.t - self.closure_start_t
                    if duration >= self.threshold_s:
                        self.fired_this_closure = True
                        return "extended_closure"
        else:
            self.closure_start_t = None
            self.fired_this_closure = False
            
        return None

    def reset(self) -> None:
        self.closure_start_t = None
        self.fired_this_closure = False
