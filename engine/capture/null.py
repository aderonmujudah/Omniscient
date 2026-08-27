import numpy as np
from engine.capture.base import ScreenCapture

class NullCapture(ScreenCapture):
    """
    Headless capture backend for tests.
    Returns a blank image.
    """
    
    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
        self.capture_count = 0

    def capture(self) -> np.ndarray:
        self.capture_count += 1
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def close(self) -> None:
        pass
