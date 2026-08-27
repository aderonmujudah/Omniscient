import numpy as np
from engine.capture.base import ScreenCapture

class NullCapture(ScreenCapture):
    """Headless capture backend.

    Each call returns a different image. A backend that returned the same pixels every time would
    make any assertion that a frozen image stayed frozen pass whether the freeze held or not, since
    a fresh capture would encode identically to the stale one.
    """

    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
        self.capture_count = 0

    def capture(self) -> np.ndarray:
        self.capture_count += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # A block large enough to survive JPEG compression, so the difference is visible to a
        # comparison made on the encoded form rather than on the array.
        frame[: self.height // 2, : self.width // 2] = (self.capture_count * 37) % 256
        return frame

    def close(self) -> None:
        pass
