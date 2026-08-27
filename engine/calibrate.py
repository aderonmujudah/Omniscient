import os
import sys
import logging
import argparse
from engine.calibration.present_window import run_calibration
from engine.calibration.harness import run_labeled_harness
from engine.calibration.assessment import GestureAssessment
from engine.calibration.store import save_profile
from engine.calibration.validation import estimate_viewing_distance_mm

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="End-to-end calibration")
    parser.add_argument("--record-file", required=True, help="Path to recording")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index")
    parser.add_argument("--dispersion-threshold", type=float, default=None)
    parser.add_argument("--screen-w", type=int, default=1920)
    parser.add_argument("--screen-h", type=int, default=1080)
    parser.add_argument("--diag-mm", type=float, default=597.0) # 24-inch monitor typical
    args = parser.parse_args()

    logger.info("Starting calibration presentation...")
    run_calibration(args.record_file, args.camera_index, args.dispersion_threshold)
    
    if not os.path.exists(args.record_file):
        logger.error("Recording was not created. Aborting.")
        sys.exit(1)

    logger.info("Fitting model to labeled session...")
    result = run_labeled_harness(args.record_file, args.screen_w, args.screen_h, args.diag_mm, args.dispersion_threshold)
    
    if not result:
        logger.error("Harness failed to produce a valid model.")
        sys.exit(1)
        
    mean_err = result["mean_error_deg"]
    worst_err = result["worst_error_deg"]
    has_measured = result["has_measured_distance"]
    
    logger.info("Validation Mean Error: %.2f deg, Worst Error: %.2f deg", mean_err, worst_err)
    
    # Assess gestures
    # Compute per-user blink thresholds from the recording
    from engine.sources.replay import ReplaySource
    from engine.calibration.session import CalibrationSession
    import json
    
    offline_session = CalibrationSession(args.screen_w, args.screen_h, args.dispersion_threshold)
    source = ReplaySource(args.record_file, realtime=False)
    source.start()
    offline_session.start()
    for sample in source.iter_samples():
        offline_session.process_sample(sample)
    source.stop()
    
    long_blink_ms = offline_session.get_long_blink_threshold_ms()
    blink_thresh = offline_session.get_blink_thresholds()
    ear_threshold = blink_thresh.get('ear_close', 0.18)
    ear_reopen = blink_thresh.get('ear_reopen', 0.24)
    
    assessment = GestureAssessment(
        long_blink_threshold_ms=long_blink_ms,
        screen_w=args.screen_w,
        screen_h=args.screen_h,
        gaze_position_available=True,
        ear_threshold=ear_threshold,
        ear_reopen=ear_reopen
    )
    
    # "with reserved-zone dwell filling any role no optional gesture earned."
    roles = assessment.assign_roles()
    
    dist_mm = 600.0
    if has_measured:
        # Re-calculate viewing distance for the profile
        dist_mm = estimate_viewing_distance_mm(result["ipd_px"])
    else:
        logger.error("Calibration did not result in a measured viewing distance.")
        # We continue to let save_profile do the rejection (which is strict)
    
    profile = {
        "model": result["model_dict"],
        "screen": {
            "w": args.screen_w,
            "h": args.screen_h,
            "diag_mm": args.diag_mm,
            "viewing_dist_mm": dist_mm
        },
        "validation": {
            "mean_error_deg": mean_err,
            "worst_error_deg": worst_err
        },
        "blink": {
            "long_threshold_ms": long_blink_ms,
            "ear_close": ear_threshold,
            "ear_reopen": ear_reopen
        },
        "gestures": {
            "roles": roles
        }
    }
    
    # Save the profile
    # Get standard omniscient profile path
    profile_path = os.path.expanduser("~/.omniscient/profile.json")
    os.makedirs(os.path.dirname(profile_path), exist_ok=True)
    
    logger.info("Writing profile to %s", profile_path)
    
    # save_profile has its own guard against mean_err > 5.0 and not has_measured
    # The prompt explicitly says: "save_profile refuses above 5.0 deg mean error and refuses an assumed viewing distance. Neither refusal may be loosened, removed, bypassed, or forced. If a run is refused, that is the finding; report it and stop."
    saved = save_profile(profile, profile_path, mean_error_deg=mean_err, has_measured_distance=has_measured)
    
    if not saved:
        logger.error("save_profile rejected the profile. See validation errors above.")
        sys.exit(1)
        
    logger.info("Profile successfully saved.")

if __name__ == "__main__":
    main()
