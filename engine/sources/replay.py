import json
import time
import logging
from typing import Iterator, Optional
from .base import GazeSource, GazeSample, EyeGeometry, Point2D, LABEL_KEY

logger = logging.getLogger(__name__)

class ReplaySource(GazeSource):
    def __init__(self, filepath: str, realtime: bool = True):
        """
        realtime reproduces the original inter-sample delays, which is required when the
        replay drives a live consumer. Offline analysis passes False: a parameter sweep reads
        the same recording once per candidate value, so sleeping through every pass costs the
        recording's duration multiplied by the number of candidates.
        """
        self.filepath = filepath
        self.realtime = realtime
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

    def _parse_sample(self, data: dict) -> GazeSample:
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

        # Replays are deterministic and must use original captured timestamps.
        return GazeSample(
            t=data['t'],
            seq=data['seq'],
            ok=data['ok'],
            condition=data.get('condition'),
            eyes=eyes,
            ear=data.get('ear'),
            ipd_px=data.get('ipd_px'),
            frame_width=data.get('frame_width'),
            conf=data.get('conf')
        )

    def iter_labeled_samples(self) -> Iterator[tuple[GazeSample, Optional[dict]]]:
        """
        Yields each sample paired with the label recorded alongside it, or None where the
        recording carries none. The label is yielded separately rather than attached to
        GazeSample because it describes the display, not the sensor.
        """
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
            sample = self._parse_sample(data)

            if self.realtime:
                if first_sample_t is None:
                    first_sample_t = sample.t
                    first_real_t = time.monotonic()
                target_time = first_real_t + (sample.t - first_sample_t)
                now = time.monotonic()
                if target_time > now:
                    time.sleep(target_time - now)

            yield sample, data.get(LABEL_KEY)

    def iter_samples(self) -> Iterator[GazeSample]:
        for sample, _ in self.iter_labeled_samples():
            yield sample
