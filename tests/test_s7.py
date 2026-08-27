try:
    import numpy
except ImportError:
    import sys
    from unittest.mock import MagicMock
    sys.modules["numpy"] = MagicMock()

from engine.machine import StateMachine
from engine.capture.null import NullCapture
from engine.input.null import NullInput
from engine.events.interaction import InteractionEvent, EventType
from unittest.mock import MagicMock

class MockDispatcher:
    def __init__(self):
        self.events = []
    def dispatch(self, ev):
        self.events.append(ev)

def test_system_menu_and_pause():
    dispatcher = MockDispatcher()
    capture = NullCapture(1920, 1080)
    input_backend = NullInput()
    
    machine = StateMachine(1920, 1080, 597.0, 2.0, 600.0, dispatcher, capture, input_backend)
    assert machine.state == "IDLE"
    assert not machine.is_paused
    
    # 1. Trigger menu role
    machine.process_event(InteractionEvent(event_type=EventType.GESTURE.value, timestamp=1.0, role="MENU"))
    assert machine.state == "SYSTEM_MENU"
    
    # 2. Trigger pause
    machine.process_event(InteractionEvent(event_type=EventType.DWELL_COMPLETE.value, timestamp=2.0, zone_id="sys_pause"))
    assert machine.state == "IDLE"
    assert machine.is_paused
    
    # 3. Verify that normal gestures are ignored while paused
    machine.process_event(InteractionEvent(event_type=EventType.GESTURE.value, timestamp=3.0, role="ENGAGE"))
    assert machine.state == "IDLE" # Ignored
    
    # 4. Trigger menu again
    machine.process_event(InteractionEvent(event_type=EventType.GESTURE.value, timestamp=4.0, role="MENU"))
    assert machine.state == "SYSTEM_MENU"
    
    # 5. Trigger resume
    machine.process_event(InteractionEvent(event_type=EventType.DWELL_COMPLETE.value, timestamp=5.0, zone_id="sys_resume"))
    assert machine.state == "IDLE"
    assert not machine.is_paused

def test_recalibrate_dispatches_event():
    dispatcher = MockDispatcher()
    capture = NullCapture(1920, 1080)
    machine = StateMachine(1920, 1080, 597.0, 2.0, 600.0, dispatcher, capture)
    
    machine.process_event(InteractionEvent(event_type=EventType.GESTURE.value, timestamp=1.0, role="MENU"))
    assert machine.state == "SYSTEM_MENU"
    
    # Trigger recalibrate
    machine.process_event(InteractionEvent(event_type=EventType.DWELL_COMPLETE.value, timestamp=2.0, zone_id="sys_recalibrate"))
    
    # Check if CALIBRATION_START was dispatched
    cal_events = [e for e in dispatcher.events if e.event_type == EventType.CALIBRATION_START.value]
    assert len(cal_events) == 1

def test_scroll_zones():
    from engine.scroll import ScrollController
    input_backend = NullInput()
    sc = ScrollController(1080, input_backend)

    sc.update(960.0, 50.0, 1.0, False)
    assert sc.active_zone == "top"

    sc.update(960.0, 50.0, 1.2, False)
    assert len(input_backend.mouse_injections) == 0, "scrolled before the arm time elapsed"

    # The arm time is reached here, which starts the ramp rather than emitting on the same tick.
    sc.update(960.0, 50.0, 1.5, False)
    assert len(input_backend.mouse_injections) == 0

    sc.update(960.0, 50.0, 1.6, False)
    assert len(input_backend.mouse_injections) > 0


def test_reserved_zones_do_not_scroll():
    """The corners hold ENGAGE, CANCEL and MENU, and the scroll bands span the full screen width.

    MENU is the recovery hatch for a user whose calibration has failed. Without this exclusion,
    dwelling on it scrolls the content underneath at the maximum rate, because the top band's ramp
    peaks at the very edge where that corner sits.
    """
    from engine.main import RESERVED_ZONES
    from engine.scroll import ScrollController

    screen_w, screen_h = 1920, 1080

    for name, zone in RESERVED_ZONES.items():
        input_backend = NullInput()
        sc = ScrollController(screen_h, input_backend,
                              screen_w=screen_w, reserved_zones=RESERVED_ZONES)
        # The centre of the reserved zone, which is where a dwell on it lands.
        x = (zone["x"] + zone["w"] / 2) * screen_w
        y = (zone["y"] + zone["h"] / 2) * screen_h

        for step in range(20):
            sc.update(x, y, 1.0 + step * 0.1, False)

        assert sc.active_zone is None, f"{name} zone armed the scroll ramp"
        assert input_backend.mouse_injections == [], f"dwelling on the {name} zone scrolled"

    # A point on the same band but clear of every reserved zone still scrolls.
    input_backend = NullInput()
    sc = ScrollController(screen_h, input_backend,
                          screen_w=screen_w, reserved_zones=RESERVED_ZONES)
    for step in range(20):
        sc.update(screen_w / 2, 40.0, 1.0 + step * 0.1, False)
    assert input_backend.mouse_injections, "the exclusion suppressed scrolling everywhere"
    
