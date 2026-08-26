import sys
import json
import logging
from engine.sources.replay import ReplaySource
from engine.calibration.session import CalibrationSession
from engine.calibration.model import CalibrationModel
from engine.calibration.validation import validate_calibration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_accuracy_harness(fixture_path: str, screen_w: int, screen_h: int, diag_mm: float):
    source = ReplaySource(fixture_path)
    session = CalibrationSession(screen_w, screen_h)
    
    source.start()
    session.start()
    
    for sample in source.iter_samples():
        event = session.process_sample(sample)
        if event and event["type"] in ("CALIBRATION_DONE", "CALIBRATION_FAILED"):
            break
            
    source.stop()
    
    if session.state != "DONE":
        logger.error("Calibration did not complete successfully.")
        return None
        
    fit_features, fit_targets = session.get_fit_data()
    val_features, val_targets = session.get_val_data()
    
    model = CalibrationModel()
    model.fit(fit_features, fit_targets)
    
    avg_ipd = session.get_avg_ipd()
    
    mean_err, worst_err, points = validate_calibration(
        model, val_features, val_targets, avg_ipd, screen_w, screen_h, diag_mm
    )
    
    logger.info(f"Accuracy Harness Results:")
    logger.info(f"Mean Error: {mean_err:.2f} degrees")
    logger.info(f"Worst Error: {worst_err:.2f} degrees")
    
    return {
        "mean_error_deg": mean_err,
        "worst_error_deg": worst_err,
        "points": points,
        "model_dict": model.to_dict(),
        "ipd_px": avg_ipd,
        "long_blink_ms": session.get_long_blink_threshold_ms()
    }

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python harness.py <fixture.jsonl> <screen_w> <screen_h> <diag_mm>")
        sys.exit(1)
    run_accuracy_harness(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]))
