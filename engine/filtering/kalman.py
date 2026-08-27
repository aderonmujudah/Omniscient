import cv2
import numpy as np
from typing import Tuple, Optional

class KalmanGazeFilter:
    def __init__(self, rate: float, process_noise: float = 1e-4, measurement_noise: float = 1e-2) -> None:
        self.rate = rate
        self.kf = cv2.KalmanFilter(4, 2)
        
        dt = 1.0 / rate
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.prev_timestamp: Optional[float] = None
        self.initialized = False

    def __call__(self, x: float, y: float, timestamp: float) -> Tuple[float, float]:
        measurement = np.array([[np.float32(x)], [np.float32(y)]], dtype=np.float32)

        if not self.initialized or self.prev_timestamp is None:
            self.kf.statePre = np.array([[np.float32(x)], [np.float32(y)], [0.0], [0.0]], dtype=np.float32)
            self.kf.statePost = np.array([[np.float32(x)], [np.float32(y)], [0.0], [0.0]], dtype=np.float32)
            self.initialized = True
            self.prev_timestamp = timestamp
            return x, y

        dt = timestamp - self.prev_timestamp
        if dt > 0.0:
            self.kf.transitionMatrix[0, 2] = np.float32(dt)
            self.kf.transitionMatrix[1, 3] = np.float32(dt)

        self.kf.predict()
        estimated = self.kf.correct(measurement)

        self.prev_timestamp = timestamp
        return float(estimated[0, 0]), float(estimated[1, 0])

    def reset(self) -> None:
        self.prev_timestamp = None
        self.initialized = False
