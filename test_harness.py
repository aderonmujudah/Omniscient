from engine.calibration.harness import run_accuracy_harness
import engine.calibration.session
engine.calibration.session.DISPERSION_THRESHOLD = 0.15

import engine.calibration.model
def linear_get_terms(self, fx, fy): return [1.0, fx, fy]
engine.calibration.model.CalibrationModel._get_terms = linear_get_terms

print("Linear Accuracy:", run_accuracy_harness("recordings/s2b/01_0506.jsonl", 1920, 1080, 597.0)["mean_error_deg"])
