import sys
import logging
from typing import Optional
from engine.sources.replay import ReplaySource
from engine.calibration.session import CalibrationSession
from engine.calibration.model import CalibrationModel
from engine.calibration.validation import validate_calibration
from engine.calibration.labels import read_labeled_session, select_accepted, split_fit_val
from engine.calibration.session import DISPERSION_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_accuracy_harness(fixture_path: str, screen_w: int, screen_h: int, diag_mm: float):
    """
    Re-runs the state machine over a recording and analyses what it collects.

    This is the right way to exercise the state machine itself, and the wrong way to tune it:
    the pairing between a collected window and a target comes from the machine's own index, so
    it is only trustworthy while the acceptance constants match those the recording was made
    at. Use run_labeled_harness for anything that varies a constant.
    """
    source = ReplaySource(fixture_path, realtime=False)
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
    frame_width = session.get_frame_width()
    
    mean_err, worst_err, points, has_measured_distance = validate_calibration(
        model, val_features, val_targets, avg_ipd, frame_width, screen_w, screen_h, diag_mm
    )
    
    logger.info(f"Accuracy Harness Results:")
    logger.info(f"Mean Error: {mean_err:.2f} degrees")
    logger.info(f"Worst Error: {worst_err:.2f} degrees")
    if not has_measured_distance:
        logger.warning("Warning: Assumed distance was used.")
    
    return {
        "mean_error_deg": mean_err,
        "worst_error_deg": worst_err,
        "points": points,
        "model_dict": model.to_dict(),
        "ipd_px": avg_ipd,
        "has_measured_distance": has_measured_distance,
        "long_blink_ms": session.get_long_blink_threshold_ms()
    }


def _resolve_threshold(requested: Optional[float], recorded: Optional[float]) -> float:
    """
    Settles which acceptance threshold an analysis runs at, and says so.

    A recording made under a loosened threshold contains only windows that threshold admitted.
    Analysing it under the built-in default silently rejects every one of them and reports too
    few windows, without naming the value that did the rejecting. The recorded value is
    therefore what an analysis defaults to; a caller sweeping candidates passes its own.
    """
    if requested is not None:
        return requested
    if recorded is not None:
        logger.info(
            "No dispersion threshold supplied. Analysing at %.4f, the value the recording was "
            "captured under.", recorded,
        )
        return recorded
    logger.warning(
        "Recording carries no acceptance threshold and none was supplied. Falling back to the "
        "built-in %.4f, which is not necessarily the value the recording was made under.",
        DISPERSION_THRESHOLD,
    )
    return DISPERSION_THRESHOLD


def run_labeled_harness(fixture_path: str, screen_w: int, screen_h: int, diag_mm: float,
                        dispersion_threshold: Optional[float] = None):
    """
    Analyses a recording using the targets it recorded, never the state machine's index.

    Every tuning run uses this path: the pairing survives a changed acceptance constant
    because it was written down while the target was on screen.

    No blink threshold is reported. A calibration yields on the order of ten blinks, over which
    a high percentile is simply the longest one observed, so that figure comes from a separate
    and longer recording.

    dispersion_threshold selects the acceptance threshold to re-apply. Left unset it takes the
    one the recording was captured under, which is the only value every window in the recording
    was actually judged against.
    """
    source = ReplaySource(fixture_path, realtime=False)
    source.start()
    labeled = read_labeled_session(source.iter_labeled_samples())
    source.stop()

    if labeled.labeled_count == 0:
        logger.error(
            "Recording carries no target labels, so what was on screen while it was captured "
            "cannot be recovered. Refusing to analyse it as though it were ground truth."
        )
        return None

    if labeled.screen_conflicts:
        logger.error(
            "Recording reports more than one screen size across its samples, so the coordinate "
            "space of its targets is ambiguous. Refusing to analyse it."
        )
        return None

    if labeled.screen_w is None:
        logger.warning(
            "Recording carries no screen dimensions. Falling back to the supplied %dx%d, which "
            "cannot be checked against what the targets were actually placed on.",
            screen_w, screen_h,
        )
    elif (labeled.screen_w, labeled.screen_h) != (screen_w, screen_h):
        # Nothing downstream would notice: the fit succeeds and the error is reported in degrees
        # scaled by the ratio between the two, which is the shape of a plausible wrong answer.
        logger.error(
            "Recording placed its targets on a %dx%d screen but %dx%d was supplied. Refusing to "
            "report an accuracy figure over a coordinate space the recording did not use.",
            labeled.screen_w, labeled.screen_h, screen_w, screen_h,
        )
        return None
    else:
        screen_w, screen_h = labeled.screen_w, labeled.screen_h

    threshold = _resolve_threshold(dispersion_threshold, labeled.dispersion_threshold)

    accepted, diverged = select_accepted(labeled.windows, threshold)
    fit, val = split_fit_val(accepted)

    if len(fit) < 5 or not val:
        logger.error(
            "Too few accepted windows to analyse at dispersion threshold %.4f: %d fit, %d "
            "validation, out of %d windows in the recording. The threshold is the gate: a "
            "recording whose windows are all wider than it yields nothing to fit.",
            threshold, len(fit), len(val), len(labeled.windows),
        )
        return None

    model = CalibrationModel()
    model.fit([w.mean_feature() for w in fit], [w.target for w in fit])

    ipds = [w.mean_ipd() for w in accepted if w.mean_ipd() > 0.0]
    avg_ipd = sum(ipds) / len(ipds) if ipds else 0.0

    mean_err, worst_err, points, has_measured_distance = validate_calibration(
        model, [w.mean_feature() for w in val], [w.target for w in val],
        avg_ipd, labeled.frame_width, screen_w, screen_h, diag_mm
    )

    if diverged:
        logger.warning(
            f"{diverged} of {len(accepted)} targets were judged differently than at capture "
            "time. The recording cannot show what the person would have done next in that case."
        )
    if not has_measured_distance:
        logger.warning("Warning: Assumed distance was used.")

    return {
        "mean_error_deg": mean_err,
        "worst_error_deg": worst_err,
        "points": points,
        "model_dict": model.to_dict(),
        "ipd_px": avg_ipd,
        "has_measured_distance": has_measured_distance,
        "paired_by": "recorded_label",
        "dispersion_threshold": threshold,
        "fit_points": len(fit),
        "val_points": len(val),
        "windows_diverged": diverged,
        "fit_targets": [w.target for w in fit],
        "val_targets": [w.target for w in val],
    }


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python harness.py <recording.jsonl> <screen_w> <screen_h> <diag_mm> [--labeled]")
        sys.exit(1)
    path, w, h, diag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    if "--labeled" in sys.argv[5:]:
        run_labeled_harness(path, w, h, diag)
    else:
        run_accuracy_harness(path, w, h, diag)
