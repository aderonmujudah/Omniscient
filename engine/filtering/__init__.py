"""Filtering package — noise suppression and fixation detection."""

from engine.filtering.one_euro import OneEuroFilter
from engine.filtering.kalman import KalmanGazeFilter
from engine.filtering.ema import EMAFilter, EMAFilter2D
from engine.filtering.fixation import FixationDetector, Fixation
from engine.filtering.classifier import SampleClassifier, SampleState

__all__ = ["OneEuroFilter", "KalmanGazeFilter", "EMAFilter", "EMAFilter2D", "FixationDetector", "Fixation", "SampleClassifier", "SampleState"]
