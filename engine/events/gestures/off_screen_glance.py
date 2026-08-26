import time
from typing import Optional
from engine.sources.base import GazeSample
from .base import GestureDetector

class OffScreenGlanceDetector(GestureDetector):
    def __init__(self, threshold_ms: float = 300.0):
        """
        Detects when the user looks explicitly off-screen.
        Implementation requires calibrated screen coordinates to function, which are not yet available to the detector.
        """
        self.threshold_s = threshold_ms / 1000.0
        self.glance_start_t = None
        self.fired = False
        
    def process_sample(self, sample: GazeSample) -> Optional[str]:
        return None

    def reset(self) -> None:
        self.glance_start_t = None
        self.fired = False
