"""Smooth pursuit detector.

Smooth pursuit is identified by correlating gaze motion against the motion of a moving
stimulus the user is asked to follow. No component yet produces that stimulus signal, so
this detector has no reachable path that can fire and reports `can_fire` as False. It is
therefore never offered as a candidate gesture: a user asked to perform it would have
every attempt recorded as a failure they caused.

The interface is retained so that the detector can be completed once a pursuit stimulus
exists, without changing any consumer.

Source: Vidal, M., Bulling, A. and Gellersen, H. (2013). Pursuits: Spontaneous
Interaction with Displays based on Smooth Pursuit Eye Movement and Moving Targets.
"""

from typing import Optional
from engine.sources.base import GazeSample


class SmoothPursuitDetector:

    def __init__(self) -> None:
        self._latched_x: float = 0.0
        self._latched_y: float = 0.0
        self._last_gaze_x: float = 0.0
        self._last_gaze_y: float = 0.0

    @property
    def name(self) -> str:
        return "smooth_pursuit"

    @property
    def requires_gaze_position(self) -> bool:
        return True

    @property
    def can_fire(self) -> bool:
        return False

    @property
    def latched_position(self) -> tuple[float, float]:
        return (self._latched_x, self._latched_y)

    def process_sample(self, sample: GazeSample) -> Optional[str]:
        """Always returns None. Correlation requires a stimulus target signal that no
        component currently produces."""
        return None

    def update_gaze_position(self, x: float, y: float) -> None:
        self._last_gaze_x = x
        self._last_gaze_y = y

    def reset(self) -> None:
        self._latched_x = 0.0
        self._latched_y = 0.0
