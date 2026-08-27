import os
import pytest
from engine.sources.replay import ReplaySource
from engine.main import build_emitter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_PATH = os.path.join(REPO_ROOT, "recordings", "s2b", "02_quiet_ordinary_1080p.jsonl")

def test_reading_inertness():
    """
    Replays 240s of ordinary blinking through the assembled engine.
    Asserts zero state transitions and zero interaction events carrying a role.
    """
    if not os.path.exists(FIXTURE_PATH):
        pytest.skip(f"Fixture not found: {FIXTURE_PATH}")

    # Use the measured bounds for this subject
    profile = {
        "model": {
            "kind": "poly2",
            "coeffs_x": [0,0,0,0,0,0],
            "coeffs_y": [0,0,0,0,0,0]
        },
        "screen": {
            "w": 1920,
            "h": 1080,
            "diag_mm": 597.0,
            "viewing_dist_mm": 600.0
        },
        "blink": {
            "long_threshold_ms": 1750.0,
            "ear_close": 0.18,
            "ear_reopen": 0.24
        },
        "gestures": {
            "roles": {
                "engage": "reserved_zone_dwell",
                "cancel": "reserved_zone_dwell",
                "menu": "corner_dwell"
            }
        }
    }

    # Actually wait! If we have a long blink that is 1750ms, it shouldn't fire during ordinary reading.
    # The gesture map says NO optional gesture was earned, so long_blink isn't even mapped to a role!
    # But wait, does it emit state transitions? "zero state transitions"
    
    class DummyDispatcher:
        def __init__(self):
            self.events = []
        def subscribe(self, handler):
            pass
        def publish(self, event):
            self.events.append(event)
        def dispatch_many(self, events):
            self.events.extend(events)
            
    dispatcher = DummyDispatcher()
    
    # We need to build the emitter.
    emitter, model, machine = build_emitter(
        profile, dispatcher, rate=30.0, capture_backend=None, input_backend=None
    )
    
    source = ReplaySource(FIXTURE_PATH, realtime=False)
    source.start()
    
    for sample in source.iter_samples():
        # process_sample returns nothing, publishes to dispatcher
        # But wait! We need to map raw to screen!
        px, py = None, None
        if sample.ok and sample.eyes:
            from engine.features.eye_features import extract_features
            f = extract_features(sample.eyes["left"], sample.eyes["right"])
            px, py = model.predict(*f)
        emitter.process_sample(sample, px, py)
                
    source.stop()
    
    role_events = 0
    state_transitions = 0
    
    for e in dispatcher.events:
        # e is an InteractionEvent if from emitter, or dictionary if state machine?
        # InteractionEvent has type and role attributes
        if hasattr(e, "type"):
            if e.type.name == "STATE_TRANSITION":
                state_transitions += 1
            if e.role is not None:
                role_events += 1
        elif isinstance(e, dict):
            if e.get("type") == "STATE_TRANSITION":
                state_transitions += 1
            if e.get("role"):
                role_events += 1
                
    assert state_transitions == 0, f"Expected 0 state transitions, got {state_transitions}"
    assert role_events == 0, f"Expected 0 role events, got {role_events}"
