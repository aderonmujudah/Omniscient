import json
import logging
from threading import Thread
from typing import Iterator
from .base import GazeSource, GazeSample

logger = logging.getLogger(__name__)

def _serialize_sample(sample: GazeSample) -> str:
    data = {
        "t": sample.t,
        "seq": sample.seq,
        "ok": sample.ok
    }
    
    if sample.condition is not None:
        data["condition"] = sample.condition
    
    if sample.eyes:
        data["eyes"] = {
            "left": {
                "iris": [sample.eyes["left"].iris.x, sample.eyes["left"].iris.y],
                "inner": [sample.eyes["left"].inner.x, sample.eyes["left"].inner.y],
                "outer": [sample.eyes["left"].outer.x, sample.eyes["left"].outer.y],
                "top": [sample.eyes["left"].top.x, sample.eyes["left"].top.y],
                "bottom": [sample.eyes["left"].bottom.x, sample.eyes["left"].bottom.y]
            },
            "right": {
                "iris": [sample.eyes["right"].iris.x, sample.eyes["right"].iris.y],
                "inner": [sample.eyes["right"].inner.x, sample.eyes["right"].inner.y],
                "outer": [sample.eyes["right"].outer.x, sample.eyes["right"].outer.y],
                "top": [sample.eyes["right"].top.x, sample.eyes["right"].top.y],
                "bottom": [sample.eyes["right"].bottom.x, sample.eyes["right"].bottom.y]
            }
        }
    if sample.ear:
        data["ear"] = sample.ear
    if sample.ipd_px is not None:
        data["ipd_px"] = sample.ipd_px
    # The capture width must survive a round trip. Viewing distance cannot be derived
    # without it, and a replayed session that omits it yields no distance at all.
    if sample.frame_width is not None:
        data["frame_width"] = sample.frame_width
    if sample.conf is not None:
        data["conf"] = sample.conf
        
    return json.dumps(data)


class RecorderSource(GazeSource):
    """Wraps an underlying source and records its output to disk."""
    def __init__(self, inner: GazeSource, filepath: str):
        self.inner = inner
        self.filepath = filepath
        self.file = None

    def start(self) -> None:
        self.inner.start()
        self.file = open(self.filepath, 'w')
        logger.info(f"RecorderSource started writing to {self.filepath}")

    def stop(self) -> None:
        self.inner.stop()
        if self.file:
            self.file.close()
        logger.info("RecorderSource stopped.")

    def iter_samples(self) -> Iterator[GazeSample]:
        for sample in self.inner.iter_samples():
            if self.file:
                self.file.write(_serialize_sample(sample) + "\n")
                self.file.flush()
            yield sample
