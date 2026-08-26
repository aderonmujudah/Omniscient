from typing import Protocol, Optional
from engine.sources.base import GazeSample

class GestureDetector(Protocol):
    def process_sample(self, sample: GazeSample) -> Optional[str]:
        """
        Process a new sample.
        Returns the gesture ID if the gesture was detected on this frame, else None.
        """
        ...
        
    def reset(self) -> None:
        """Reset detector state."""
        ...
