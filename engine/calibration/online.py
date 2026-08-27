from typing import Tuple, Optional
from engine.calibration.model import CalibrationModel
import math

class OnlineRecalibrator:
    def __init__(self, model: CalibrationModel, max_shift_px: float = 100.0):
        self.model = model
        self.max_shift_px = max_shift_px
        self.activation_history = []
        
        # State for current correction
        self.dx = 0.0
        self.dy = 0.0

    def hook_successful_activation(self, feature: Tuple[float, float], screen_point: Tuple[float, float]) -> None:
        """
        Records a successful activation (feature vector and known screen point)
        to slowly adapt the calibration model online.
        """
        self.activation_history.append((feature, screen_point))
        
        # Use a simple moving average of recent errors (predicted vs actual)
        # to shift the model. Bounded to max_shift_px.
        recent = self.activation_history[-5:]
        sum_err_x, sum_err_y = 0.0, 0.0
        
        # Disable our own correction when making predictions to find raw error
        raw_predict = self.model.predict
        if hasattr(self.model, '_raw_predict'):
            raw_predict = self.model._raw_predict
            
        for f, s in recent:
            px, py = raw_predict(f[0], f[1])
            sum_err_x += (s[0] - px)
            sum_err_y += (s[1] - py)
            
        avg_err_x = sum_err_x / len(recent)
        avg_err_y = sum_err_y / len(recent)
        
        # Bound the correction
        dist = math.hypot(avg_err_x, avg_err_y)
        if dist > self.max_shift_px:
            scale = self.max_shift_px / dist
            avg_err_x *= scale
            avg_err_y *= scale
            
        self.dx = avg_err_x
        self.dy = avg_err_y
        
        # Inject our correction into the model
        if not hasattr(self.model, '_raw_predict'):
            self.model._raw_predict = self.model.predict
            
        def corrected_predict(fx, fy):
            px, py = self.model._raw_predict(fx, fy)
            return px + self.dx, py + self.dy
            
        self.model.predict = corrected_predict

    def revert(self) -> None:
        """Revert any online correction."""
        if hasattr(self.model, '_raw_predict'):
            self.model.predict = self.model._raw_predict
            del self.model._raw_predict
        self.dx = 0.0
        self.dy = 0.0
        self.activation_history.clear()
