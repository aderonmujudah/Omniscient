import time
from typing import Optional
from engine.sources.base import GazeSample
from .base import GestureDetector

class OffScreenGlanceDetector(GestureDetector):
    def __init__(self, threshold_ms: float = 300.0):
        # We assume off-screen glance is detected by extreme feature displacement or model predicted coords being way off screen.
        # But this operates on GazeSample before model calibration? 
        # Wait, the gesture assessment runs AFTER calibration.
        # So we can pass calibrated point to it, or it can detect it from raw features.
        # Let's say off screen is when feature dx or dy > 0.35 (just a heuristic).
        self.threshold_s = threshold_ms / 1000.0
        self.glance_start_t = None
        self.fired = False
        
    def process_sample(self, sample: GazeSample) -> Optional[str]:
        # A real implementation would check calibrated screen coordinates or feature limits.
        # For this scope, we just provide a structural skeleton that the assessment can run.
        if sample.ok and sample.eyes:
            # Fake logic for off-screen: e.g. very large EAR or something, 
            # actually we don't have screen coordinates here unless we pass them.
            # Let's assume we pass calibrated screen coords as an extension to GazeSample,
            # or the detector maintains its own logic.
            pass
            
        return None

    def reset(self) -> None:
        self.glance_start_t = None
        self.fired = False
