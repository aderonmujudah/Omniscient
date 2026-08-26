import pytest
import os
import json
from engine.sources.replay import ReplaySource
from engine.calibration.session import CalibrationSession
from engine.calibration.model import CalibrationModel
from engine.calibration.validation import validate_calibration
from engine.calibration.assessment import GestureAssessment
from engine.calibration.store import save_profile, load_profile
from engine.sources.base import GazeSample

FIXTURE_PATH = "/app/fixtures/s2_calibration.jsonl"

def test_calibration_end_to_end_and_validation():
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
    
    mean_err, worst_err, points = validate_calibration(
        model, val_features, val_targets, avg_ipd, 1920, 1080, 597.0
    )
    
    assert mean_err > 0.0 # Error should be reported
    assert worst_err > 0.0
    
def test_deliberately_corrupted_calibration():
    fit_features = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.0, 0.1), (0.1, 0.1)]
    fit_targets = [(100, 100), (200, 100), (300, 100), (100, 200), (200, 200)]
    
    model = CalibrationModel()
    model.fit(fit_features, fit_targets)
    
    # Corrupt validation targets
    val_features = [(0.3, 0.0)]
    val_targets = [(9999, 9999)] 
    
    mean_err, worst_err, points = validate_calibration(
        model, val_features, val_targets, 118.4, 1920, 1080, 597.0
    )
    
    assert mean_err > 50.0 # Huge error in degrees

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
    
    profile = {
        "model": model.to_dict()
    }
    
    p = str(tmp_path / "profile.json")
    save_profile(profile, p)
    
    loaded = load_profile(p)
    model2 = CalibrationModel()
    model2.load_dict(loaded["model"])
    
    assert model.coeffs_x == model2.coeffs_x
    assert model.coeffs_y == model2.coeffs_y
    
    px, py = model.predict(0.5, 0.5)
    px2, py2 = model2.predict(0.5, 0.5)
    
    assert px == px2
    assert py == py2

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
    assess = GestureAssessment()
    assess.start()
    
    assert assess.state == "EXPLAIN"
    assess.user_ready()
    assert assess.state == "TEST_ACTIVE"
    
    # We test long blink. Threshold is 450ms.
    # Send blink ok=False for 500ms
    t = 1000.0
    assess.process_sample(GazeSample(t=t, seq=1, ok=True, ear={"left": 0.1, "right": 0.1}))
    t += 0.5
    assess.process_sample(GazeSample(t=t, seq=2, ok=True, ear={"left": 0.1, "right": 0.1}))
    
    assert assess.successes == 1

def test_gesture_assessment_decline_returns_fallback_strings():
    """
    Asserts that declining all optional gestures returns the correct fallback strings.
    Actual instantiation and behavioral tests of reserved_zone_dwell belong in Scope 3.
    """
    assess = GestureAssessment()
    assess.start()
    
    # Decline everything
    while assess.state != "DONE":
        assess.user_declines()
        
    roles = assess.assign_roles()
    
    assert roles["engage"] == "reserved_zone_dwell"
    assert roles["cancel"] == "reserved_zone_dwell"
    assert roles["menu"] == "corner_dwell"
    
def test_gesture_assessment_failure():
    assess = GestureAssessment()
    assess.start()
    assess.user_ready()
    
    # Don't trigger the gesture, just timeout 3 times
    t = 1000.0
    for _ in range(4):
        t += 3.1
        assess.process_sample(GazeSample(t=t, seq=1, ok=True))
        
    assert assess.successes == 0
    assert assess.state == "TEST_CONTROL"
    
    t += 3.1
    assess.process_sample(GazeSample(t=t, seq=2, ok=True))
    
    roles = assess.assign_roles()
    assert assess.results[0]["enabled"] == False
