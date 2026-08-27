from typing import Protocol
import numpy as np

class ScreenCapture(Protocol):
    """
    Interface for capturing the screen.
    """
    
    def capture(self) -> np.ndarray:
        """
        Capture the current screen.
        
        Returns:
            A numpy array representing the screen image (BGR or BGRA format),
            with the overlay window excluded from the capture.
        """
        pass

    def close(self) -> None:
        """
        Release any resources used by the capture backend.
        """
        pass
