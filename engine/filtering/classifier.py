"""Per-sample state classifier.

Combines fixation detection, blink detection, and tracking status
to assign one of four labels to each gaze sample.

Priority order, highest first:
1. LOST — face or gaze not detected
2. BLINK — eyes closed (EAR below threshold)
3. FIXATION — gaze within an active fixation
4. SACCADE — gaze moving between fixations
"""

import enum
from typing import Optional, Tuple
from engine.filtering.fixation import FixationDetector, Fixation


class SampleState(enum.Enum):
    FIXATION = "FIXATION"
    SACCADE = "SACCADE"
    BLINK = "BLINK"
    LOST = "LOST"


class SampleClassifier:

    def __init__(
        self,
        fixation_detector: FixationDetector,
        ear_threshold: float = 0.2,
    ) -> None:
        self._fixation_detector = fixation_detector
        self._ear_threshold = ear_threshold

    def classify(
        self,
        ok: bool,
        ear_left: Optional[float],
        ear_right: Optional[float],
        gaze_x: float,
        gaze_y: float,
        timestamp: float,
    ) -> Tuple[SampleState, Optional[Fixation]]:
        """Classify a sample given its tracking status and calibrated gaze.

        Args:
            ok: Whether face/gaze tracking succeeded.
            ear_left: Left eye aspect ratio, or None if unavailable.
            ear_right: Right eye aspect ratio, or None if unavailable.
            gaze_x: Calibrated horizontal screen coordinate in pixels.
            gaze_y: Calibrated vertical screen coordinate in pixels.
            timestamp: Sample timestamp in seconds.

        Returns:
            Tuple of (state label, completed fixation if one just ended).
        """
        if not ok:
            return SampleState.LOST, None

        if ear_left is not None and ear_right is not None:
            if ear_left < self._ear_threshold and ear_right < self._ear_threshold:
                return SampleState.BLINK, None

        completed = self._fixation_detector.process(gaze_x, gaze_y, timestamp)

        if self._fixation_detector.active_fixation_centroid is not None:
            return SampleState.FIXATION, completed

        return SampleState.SACCADE, completed

    def reset(self) -> None:
        self._fixation_detector.reset()
