from unittest.mock import MagicMock

try:
    import numpy
except ImportError:
    import sys
    from unittest.mock import MagicMock
    sys.modules["numpy"] = MagicMock()

import pytest
import math
from engine.machine import StateMachine, Rect
from engine.capture.null import NullCapture
from engine.input.null import NullInput
from engine.events.interaction import InteractionEvent, EventType
from engine.calibration.online import OnlineRecalibrator
from engine.calibration.model import CalibrationModel

class MockDispatcher:
    def __init__(self):
        self.events = []
    def dispatch(self, ev):
        self.events.append(ev)

def test_radial_menu_selection():
    dispatcher = MockDispatcher()
    capture = NullCapture(1920, 1080)
    input_backend = NullInput()
    machine = StateMachine(1920, 1080, 597.0, 2.0, 600.0, dispatcher, capture, input_backend=input_backend)
    
    # Go to ZOOM1
    machine.process_event(InteractionEvent(event_type=EventType.GESTURE.value, timestamp=1.0, role="ENGAGE"))
    machine.process_event(InteractionEvent(event_type=EventType.DWELL_COMPLETE.value, timestamp=2.0, x=10.0, y=10.0))
    
    # ENGAGE in ZOOM1 -> RADIAL
    machine.process_event(InteractionEvent(event_type=EventType.GESTURE.value, timestamp=3.0, role="ENGAGE", x=960.0, y=540.0))
    assert machine.state == "RADIAL"
    
    # Dwell on drag_start
    machine.process_event(InteractionEvent(event_type=EventType.DWELL_COMPLETE.value, timestamp=4.0, zone_id="radial_drag_start"))
    
    assert machine.state == "IDLE"
    assert len(input_backend.mouse_injections) > 0

def test_implicit_recalibration():
    dispatcher = MockDispatcher()
    capture = NullCapture(1920, 1080)
    input_backend = NullInput()
    
    model = CalibrationModel()
    # Mock predict
    model.predict = MagicMock(return_value=(960.0, 540.0))
    
    recalibrator = OnlineRecalibrator(model)
    machine = StateMachine(1920, 1080, 597.0, 2.0, 600.0, dispatcher, capture, input_backend=input_backend, recalibrator=recalibrator)
    
    # Setup some fake features
    machine.latest_features = (0.5, 0.5)
    machine.resolved_point = (1000.0, 600.0) # The user clicked here
    
    # execute action (simulate DWELL_COMPLETE in ZOOM2)
    machine.zoom2_rect = Rect(800, 400, 200, 200)
    machine.state = "ZOOM2"
    machine.process_event(InteractionEvent(event_type=EventType.DWELL_COMPLETE.value, timestamp=1.0, x=1920.0, y=1080.0))
    # This will map event.x=1920, event.y=1080 -> 1.0, 1.0 -> orig_x = 800+200=1000, orig_y=400+200=600
    
    assert len(recalibrator.activation_history) == 1
    
    # The error is (1000-960) = 40, (600-540) = 60
    # Because it's the only point, avg err is 40, 60
    # Next predict should add 40, 60
    model._raw_predict = MagicMock(return_value=(960.0, 540.0))
    px, py = model.predict(0.5, 0.5)
    assert px == 1000.0
    assert py == 600.0
    
    # Revert
    recalibrator.revert()
    px, py = model.predict(0.5, 0.5)
    assert px == 960.0
    assert py == 540.0

def test_cancel_returns_to_idle():
    dispatcher = MockDispatcher()
    capture = NullCapture(1920, 1080)
    machine = StateMachine(1920, 1080, 597.0, 2.0, 600.0, dispatcher, capture)
    
    machine.process_event(InteractionEvent(event_type=EventType.GESTURE.value, timestamp=1.0, role="ENGAGE"))
    machine.process_event(InteractionEvent(event_type=EventType.DWELL_COMPLETE.value, timestamp=2.0, x=10.0, y=10.0))
    machine.process_event(InteractionEvent(event_type=EventType.GESTURE.value, timestamp=3.0, role="ENGAGE", x=960.0, y=540.0))
    assert machine.state == "RADIAL"
    
    machine.process_event(InteractionEvent(event_type=EventType.DWELL_COMPLETE.value, timestamp=4.0, zone_id="radial_cancel"))
    assert machine.state == "IDLE"


def test_adversarial_recalibration_bound():
    model = CalibrationModel()
    model.predict = MagicMock(return_value=(960.0, 540.0))
    recalibrator = OnlineRecalibrator(model, max_shift_px=100.0)
    
    # Send a bunch of activations that are WRONG by 500 pixels!
    # They should be capped by the 100.0 limit.
    for _ in range(10):
        recalibrator.hook_successful_activation((0.5, 0.5), (1460.0, 540.0))
    
    model._raw_predict = MagicMock(return_value=(960.0, 540.0))
    px, py = model.predict(0.5, 0.5)
    
    # 1460 - 960 = +500 x error. Bounded to 100 max shift.
    # So the model should return 960 + 100 = 1060.
    assert px == 1060.0
    assert py == 540.0
    
    # Revert it
    recalibrator.revert()
    px, py = model.predict(0.5, 0.5)
    assert px == 960.0
    assert py == 540.0

