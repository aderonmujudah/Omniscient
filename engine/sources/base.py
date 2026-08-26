from typing import Protocol, Iterator, Optional, Dict
from dataclasses import dataclass

@dataclass
class Point2D:
    x: float
    y: float

@dataclass
class EyeGeometry:
    iris: Point2D
    inner: Point2D
    outer: Point2D
    top: Point2D
    bottom: Point2D

@dataclass
class GazeSample:
    t: float
    seq: int
    ok: bool
    condition: Optional[str] = None
    eyes: Optional[Dict[str, EyeGeometry]] = None
    ear: Optional[Dict[str, float]] = None
    ipd_px: Optional[float] = None
    frame_width: Optional[int] = None
    conf: Optional[float] = None

class GazeSource(Protocol):
    def start(self) -> None:
        """Initialize the source."""
        ...

    def stop(self) -> None:
        """Stop the source and release resources."""
        ...

    def iter_samples(self) -> Iterator[GazeSample]:
        """Yield GazeSample objects continually."""
        ...
