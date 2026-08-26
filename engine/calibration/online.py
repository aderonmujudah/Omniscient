from typing import Tuple
from engine.calibration.model import CalibrationModel

class OnlineRecalibrator:
    def __init__(self, model: CalibrationModel):
        self.model = model
        self.activation_history = []

    def hook_successful_activation(self, feature: Tuple[float, float], screen_point: Tuple[float, float]) -> None:
        """
        Records a successful activation (feature vector and known screen point)
        to slowly adapt the calibration model online.
        """
        self.activation_history.append((feature, screen_point))
        # Future implementation will maintain a sliding window of points
        # to blend a local affine correction or refit the polynomial.
