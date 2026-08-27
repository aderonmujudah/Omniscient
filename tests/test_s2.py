import pytest
import os
import json
from engine.sources.replay import ReplaySource
from engine.calibration.session import CalibrationSession
from engine.calibration.model import CalibrationModel
from engine.calibration.validation import validate_calibration, estimate_viewing_distance_mm
from engine.calibration.assessment import GestureAssessment
from engine.calibration.store import save_profile, load_profile
from engine.calibration.harness import run_accuracy_harness
from engine.calibration.online import OnlineRecalibrator
from engine.sources.base import GazeSample

# The suite runs both inside the test container, where the repository is mounted at /app, and
# directly against a virtualenv on a development host. Resolving the fixture from this file
# rather than from an absolute mount point keeps both paths working.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_PATH = os.path.join(REPO_ROOT, "fixtures", "s2_calibration.jsonl")

def test_calibration_end_to_end_and_validation(tmp_path):
    source = ReplaySource(FIXTURE_PATH)
    session = CalibrationSession(1920, 1080)
    
    source.start()
    session.start()
    
    events = []
    for sample in source.iter_samples():
        event = session.process_sample(sample)
        if event:
            events.append(event)
        if session.state == "DONE":
            break
            
    assert session.state == "DONE"
    
    fit_features, fit_targets = session.get_fit_data()
    val_features, val_targets = session.get_val_data()
    
    # Regression fitted and evaluated on disjoint point sets
    assert len(fit_features) == 9
    assert len(val_features) == 4
    
    # Validation points never appear in fit
    for vt in val_targets:
        assert vt not in fit_targets
        
    model = CalibrationModel()
    model.fit(fit_features, fit_targets)
    
    avg_ipd = session.get_avg_ipd()
    
    mean_err, worst_err, points, has_measured = validate_calibration(
        model, val_features, val_targets, avg_ipd, session.get_frame_width(), 1920, 1080, 597.0
    )
    
    assert mean_err > 0.0 # Error should be reported
    assert worst_err > 0.0
    
    # Profile should save since synthetic error is very low
    p = str(tmp_path / "profile.json")
    saved = save_profile({"model": model.to_dict()}, p, mean_error_deg=mean_err,
                         has_measured_distance=has_measured)
    assert saved
    assert os.path.exists(p)

def test_deliberately_corrupted_calibration(tmp_path):
    fit_features = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.0, 0.1), (0.1, 0.1)]
    fit_targets = [(100, 100), (200, 100), (300, 100), (100, 200), (200, 200)]
    
    model = CalibrationModel()
    model.fit(fit_features, fit_targets)
    
    # Corrupt validation targets
    val_features = [(0.3, 0.0)]
    val_targets = [(9999, 9999)] 
    
    mean_err, worst_err, points, has_measured = validate_calibration(
        model, val_features, val_targets, 118.4, 640, 1920, 1080, 597.0
    )
    
    assert mean_err > 50.0 # Huge error in degrees
    
    p = str(tmp_path / "corrupted_profile.json")
    saved = save_profile({"model": model.to_dict()}, p, mean_error_deg=mean_err,
                         has_measured_distance=has_measured)
    
    assert not saved # Profile storage is gated and rejected
    assert not os.path.exists(p)

def test_dispersion_rejection():
    session = CalibrationSession(1920, 1080)
    session.start()
    
    from engine.sources.base import EyeGeometry, Point2D
    
    # Send highly dispersed samples
    t = 1000.0
    # Settling 0.3s
    for i in range(18):
        geom = {"left": EyeGeometry(Point2D(0,0), Point2D(0,0), Point2D(100,0), Point2D(0,-20), Point2D(0,20)),
                "right": EyeGeometry(Point2D(0,0), Point2D(0,0), Point2D(100,0), Point2D(0,-20), Point2D(0,20))}
        s = GazeSample(t=t, seq=i, ok=True, eyes=geom, ipd_px=100)
        session.process_sample(s)
        t += 1/60.0
        
    # Collecting 0.7s
    for i in range(43):
        # alternate extreme gaze
        fx = 0.5 if i % 2 == 0 else -0.5
        iris_x = fx * 100
        geom = {"left": EyeGeometry(Point2D(iris_x,0), Point2D(0,0), Point2D(100,0), Point2D(0,-20), Point2D(0,20)),
                "right": EyeGeometry(Point2D(iris_x,0), Point2D(0,0), Point2D(100,0), Point2D(0,-20), Point2D(0,20))}
        s = GazeSample(t=t, seq=18+i, ok=True, eyes=geom, ipd_px=100)
        event = session.process_sample(s)
        t += 1/60.0
        
    assert session.retries == 1

def test_profile_persistence(tmp_path):
    model = CalibrationModel()
    model.coeffs_x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    model.coeffs_y = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    
    results = [
        {"id": "long_blink", "success": 0.9, "false_positive": 0.05, "enabled": True, "declined_by_user": False, "params": {"threshold_ms": 450.0}}
    ]
    roles = {"engage": "long_blink", "cancel": "reserved_zone_dwell", "menu": "corner_dwell"}
    
    profile = {
        "model": model.to_dict(),
        "assessment_results": results,
        "roles": roles
    }
    
    p = str(tmp_path / "profile.json")
    save_profile(profile, p, mean_error_deg=1.0, has_measured_distance=True)
    
    loaded = load_profile(p)
    model2 = CalibrationModel()
    model2.load_dict(loaded["model"])
    
    assert model.coeffs_x == model2.coeffs_x
    assert model.coeffs_y == model2.coeffs_y
    
    px, py = model.predict(0.5, 0.5)
    px2, py2 = model2.predict(0.5, 0.5)
    
    assert px == px2
    assert py == py2
    
    assert loaded["assessment_results"] == results
    assert loaded["roles"] == roles

def test_blink_threshold():
    session = CalibrationSession(1920, 1080)
    # fake natural blink of 250ms
    t = 1000.0
    session.process_sample(GazeSample(t=t, seq=1, ok=True, ear={"left": 0.1, "right": 0.1}))
    from engine.sources.base import EyeGeometry, Point2D
    geom = {"left": EyeGeometry(Point2D(0,0), Point2D(0,0), Point2D(100,0), Point2D(0,-20), Point2D(0,20)),
            "right": EyeGeometry(Point2D(0,0), Point2D(0,0), Point2D(100,0), Point2D(0,-20), Point2D(0,20))}
    session.process_sample(GazeSample(t=t+0.25, seq=2, ok=True, eyes=geom, ear={"left": 0.3, "right": 0.3}, ipd_px=100))
    
    val = session.get_long_blink_threshold_ms()
    assert val >= 400.0 # Floor should apply since 250 < 400
    
def test_gesture_assessment_success():
    assess = GestureAssessment(gaze_position_available=False)
    assess.start()
    
    assert assess.state == "EXPLAIN"
    assess.user_ready()
    assert assess.state == "TEST_ACTIVE"
    
    # The measured threshold is 450 ms, so a 500 ms closure qualifies. The closure is
    # classified when the eyes reopen, since a longer closure would be a different gesture.
    t = 1000.0
    assess.process_sample(GazeSample(t=t, seq=1, ok=True, ear={"left": 0.1, "right": 0.1}), gaze_x=None, gaze_y=None)
    t += 0.5
    assess.process_sample(GazeSample(t=t, seq=2, ok=True, ear={"left": 0.3, "right": 0.3}), gaze_x=None, gaze_y=None)
    
    assert assess.successes == 1

def test_gesture_assessment_decline_returns_fallback_strings():
    """
    Asserts that declining all optional gestures returns the correct fallback strings.
    Actual instantiation and behavioral tests of reserved_zone_dwell are deferred until implemented.
    """
    assess = GestureAssessment(gaze_position_available=False)
    assess.start()
    
    # Decline everything
    while assess.state != "DONE":
        assess.user_declines()
        
    roles = assess.assign_roles()
    
    assert roles["engage"] == "reserved_zone_dwell"
    assert roles["cancel"] == "reserved_zone_dwell"
    assert roles["menu"] == "corner_dwell"
    
def test_gesture_assessment_failure():
    assess = GestureAssessment(gaze_position_available=False)
    assess.start()
    assess.user_ready()
    
    # Don't trigger the gesture, just timeout 3 times
    t = 1000.0
    for _ in range(4):
        t += 3.1
        assess.process_sample(GazeSample(t=t, seq=1, ok=True), gaze_x=None, gaze_y=None)
        
    assert assess.successes == 0
    assert assess.state == "TEST_CONTROL"
    
    # complete control window (10.0s)
    t += 10.1
    assess.process_sample(GazeSample(t=t, seq=2, ok=True), gaze_x=None, gaze_y=None)
    
    roles = assess.assign_roles()
    assert assess.results[0]["enabled"] == False
    assert roles["engage"] == "reserved_zone_dwell"
    assert roles["cancel"] == "reserved_zone_dwell"

def test_declined_gesture_stays_disabled_with_high_reliability():
    assess = GestureAssessment(gaze_position_available=False)
    assess.start()
    assess.user_ready()
    
    # Measure a high success rate (3 successes)
    t = 1000.0
    for _ in range(3):
        assess.process_sample(GazeSample(t=t, seq=1, ok=True, ear={"left": 0.1, "right": 0.1}), gaze_x=None, gaze_y=None)
        t += 0.5
        assess.process_sample(GazeSample(t=t, seq=2, ok=True, ear={"left": 0.3, "right": 0.3}), gaze_x=None, gaze_y=None)
        t += 1.0

    assert assess.successes == 3
    assert assess.state == "TEST_CONTROL"
    
    # User declines during the control period despite the perfect successes
    assess.user_declines(current_t=t)
    
    res = assess.results[0]
    assert res["declined_by_user"] == True
    assert res["enabled"] == False
    assert res["success"] == 1.0 # The high success rate was recorded before declining
    
    # Decline the rest
    while assess.state != "DONE":
        assess.user_declines(current_t=t)
        
    roles = assess.assign_roles()
    assert roles["engage"] == "reserved_zone_dwell" # The declined highly reliable gesture is absent from roles

def test_harness_runs():
    res = run_accuracy_harness(FIXTURE_PATH, 1920, 1080, 597.0)
    assert res is not None
    assert "mean_error_deg" in res

def test_focal_length_resolution_independence():
    # If a person's physical IPD is constant, 
    # capturing at 1280x720 means the pixel IPD is double that at 640x360.
    # The estimated viewing distance should remain exactly the same.
    dist_640 = estimate_viewing_distance_mm(ipd_px=50, frame_width_px=640)
    dist_1280 = estimate_viewing_distance_mm(ipd_px=100, frame_width_px=1280)
    assert abs(dist_640 - dist_1280) < 0.1

def test_online_recalibrator():
    model = CalibrationModel()
    model.coeffs_x = [1.0] * 6
    model.coeffs_y = [1.0] * 6
    recalibrator = OnlineRecalibrator(model)
    recalibrator.hook_successful_activation((0.1, 0.1), (100, 100))
    assert len(recalibrator.activation_history) == 1

def test_assessment_standalone_rerun():
    assess = GestureAssessment(gaze_position_available=False)
    assess.start()
    
    # Drive to completion by declining all
    while assess.state != "DONE":
        assess.user_declines()
        
    # Without a gaze position only the closure gestures can be presented.
    # Previously 2 (long_blink, extended_closure), now 3 with double_blink.
    assert len(assess.results) == 3
    
    # Rerun
    assess.start()
    assert assess.state == "EXPLAIN"
    assert assess.current_idx == 0
    assert len(assess.results) == 0
    assert assess.attempts == 0
    assert assess.successes == 0
    assert assess.false_positives == 0

def test_focal_length_resolution_independence_real_path():
    from engine.calibration.session import CalibrationSession
    from engine.calibration.model import CalibrationModel
    from engine.calibration.validation import validate_calibration
    from engine.sources.base import GazeSample, EyeGeometry, Point2D

    # Create two sessions with identical geometries, but one captured at 640 width with IPD 50,
    # and one at 1280 width with IPD 100.
    
    def run_session(width, ipd):
        session = CalibrationSession(1920, 1080)
        session.start()
        
        # Settle
        t = 1000.0
        session.process_sample(GazeSample(t=t, seq=1, ok=True, frame_width=width))
        t += 0.4
        
        # Provide one target's worth of data, enough to finish point 0
        for i in range(12):
            geom = {"left": EyeGeometry(Point2D(0,0), Point2D(0,0), Point2D(100,0), Point2D(0,-20), Point2D(0,20)),
                    "right": EyeGeometry(Point2D(0,0), Point2D(0,0), Point2D(100,0), Point2D(0,-20), Point2D(0,20))}
            session.process_sample(GazeSample(t=t, seq=2+i, ok=True, eyes=geom, ipd_px=ipd, frame_width=width))
            t += 1/60.0
            
        fit_f, fit_t = session.get_fit_data()
        model = CalibrationModel()
        # Just manually set model since fit needs more points
        model.coeffs_x = [1.0] * 6
        model.coeffs_y = [1.0] * 6
        
        val_f = [(0.5, 0.5)]
        val_t = [(100, 100)]
        
        mean_err, worst_err, points, has_measured = validate_calibration(model, val_f, val_t, session.get_avg_ipd(), session.get_frame_width(), 1920, 1080, 597.0)
        return mean_err

    err_640 = run_session(640, 50)
    err_1280 = run_session(1280, 100)
    
    assert abs(err_640 - err_1280) < 0.1


def test_absent_ipd_is_refused(tmp_path):
    from engine.calibration.session import CalibrationSession
    from engine.calibration.model import CalibrationModel
    from engine.calibration.validation import validate_calibration
    from engine.calibration.store import save_profile
    from engine.sources.base import GazeSample, EyeGeometry, Point2D

    session = CalibrationSession(1920, 1080)
    session.start()
    
    # Settle
    t = 1000.0
    session.process_sample(GazeSample(t=t, seq=1, ok=True, frame_width=640))
    t += 0.4
    
    # Collect with NO ipd_px
    for i in range(12):
        geom = {"left": EyeGeometry(Point2D(0,0), Point2D(0,0), Point2D(100,0), Point2D(0,-20), Point2D(0,20)),
                "right": EyeGeometry(Point2D(0,0), Point2D(0,0), Point2D(100,0), Point2D(0,-20), Point2D(0,20))}
        session.process_sample(GazeSample(t=t, seq=2+i, ok=True, eyes=geom, frame_width=640))
        t += 1/60.0
        
    model = CalibrationModel()
    model.coeffs_x = [1.0] * 6
    model.coeffs_y = [1.0] * 6
    
    val_f = [(0.5, 0.5)]
    val_t = [(100, 100)]
    
    mean_err, worst_err, points, has_measured = validate_calibration(model, val_f, val_t, session.get_avg_ipd(), session.get_frame_width(), 1920, 1080, 597.0)
    
    assert has_measured is False
    
    p = str(tmp_path / "absent_profile.json")
    saved = save_profile({"model": model.to_dict()}, p, mean_error_deg=1.0, has_measured_distance=has_measured)
    
    assert saved is False
    import os
    assert not os.path.exists(p)


def test_absent_frame_width_is_refused(tmp_path):
    """
    A session whose samples never carried a capture width cannot yield a viewing
    distance, so its accuracy figure is not a measurement and must not be stored.
    This is the case for any recording made before the width was captured.
    """
    from engine.calibration.session import CalibrationSession
    from engine.calibration.model import CalibrationModel
    from engine.calibration.validation import validate_calibration
    from engine.calibration.store import save_profile
    from engine.sources.base import GazeSample, EyeGeometry, Point2D

    session = CalibrationSession(1920, 1080)
    session.start()

    t = 1000.0
    session.process_sample(GazeSample(t=t, seq=1, ok=True))
    t += 0.4

    # A healthy IPD throughout, but no sample declares the width it was captured at.
    for i in range(12):
        geom = {"left": EyeGeometry(Point2D(0,0), Point2D(0,0), Point2D(100,0), Point2D(0,-20), Point2D(0,20)),
                "right": EyeGeometry(Point2D(0,0), Point2D(0,0), Point2D(100,0), Point2D(0,-20), Point2D(0,20))}
        session.process_sample(GazeSample(t=t, seq=2+i, ok=True, eyes=geom, ipd_px=120.0))
        t += 1/60.0

    # The session saw a usable IPD on every sample but never learned the capture width.
    assert session.get_frame_width() is None

    model = CalibrationModel()
    model.coeffs_x = [1.0] * 6
    model.coeffs_y = [1.0] * 6

    # A healthy IPD cannot rescue an unknown width: both are needed for a distance.
    mean_err, worst_err, points, has_measured = validate_calibration(
        model, [(0.5, 0.5)], [(100, 100)], 120.0,
        session.get_frame_width(), 1920, 1080, 597.0
    )

    assert has_measured is False

    p = str(tmp_path / "no_width_profile.json")
    assert save_profile({"model": model.to_dict()}, p, mean_error_deg=1.0,
                        has_measured_distance=has_measured) is False
    assert not os.path.exists(p)


def test_harness_reports_measured_distance_from_fixture():
    """
    The replay fixture declares the width it was captured at, so the harness must
    report a genuine measurement rather than falling back to an assumed distance.
    """
    res = run_accuracy_harness(FIXTURE_PATH, 1920, 1080, 597.0)
    assert res is not None
    assert res["has_measured_distance"] is True
