import json
import time
import logging
from typing import Iterator
from .base import GazeSource, GazeSample, EyeGeometry, Point2D

logger = logging.getLogger(__name__)

class ReplaySource(GazeSource):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None
        self.is_running = False

    def start(self) -> None:
        logger.info(f"Starting ReplaySource from {self.filepath}")
        self.file = open(self.filepath, 'r')
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False
        if self.file:
            self.file.close()
        logger.info("ReplaySource stopped.")

    def iter_samples(self) -> Iterator[GazeSample]:
        if not self.file:
            return

        first_sample_t = None
        first_real_t = None

        for line in self.file:
            if not self.is_running:
                break
            
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            
            # Reconstruct GazeSample
            eyes = None
            if 'eyes' in data and data['eyes']:
                eyes = {}
                for eye_key, eye_data in data['eyes'].items():
                    eyes[eye_key] = EyeGeometry(
                        iris=Point2D(*eye_data['iris']),
                        inner=Point2D(*eye_data['inner']),
                        outer=Point2D(*eye_data['outer']),
                        top=Point2D(*eye_data['top']),
                        bottom=Point2D(*eye_data['bottom'])
                    )
            
            sample = GazeSample(
                t=data['t'],
                seq=data['seq'],
                ok=data['ok'],
                condition=data.get('condition'),
                eyes=eyes,
                ear=data.get('ear'),
                ipd_px=data.get('ipd_px'),
                conf=data.get('conf')
            )

            # Timing alignment
            if first_sample_t is None:
                first_sample_t = sample.t
                first_real_t = time.monotonic()
            
            target_time = first_real_t + (sample.t - first_sample_t)
            now = time.monotonic()
            if target_time > now:
                time.sleep(target_time - now)

            # Update sample timestamp to reflect replay capture time?
            # Instructions: "replays a captured GazeSample stream from disk at its original timing."
            # We keep the original t or update it? "A session can be recorded and replayed with inter-sample timing within 5 ms of the original."
            # Let's yield it exactly as recorded.
            yield sample
