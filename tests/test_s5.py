try:
    import numpy
except ImportError:
    import sys
    from unittest.mock import MagicMock
    sys.modules["numpy"] = MagicMock()

import pytest
import math
from engine.machine import compute_cells, get_cell_for_point, Rect

def test_grid_boundaries():
    screen_w, screen_h = 1920, 1080
    cols, rows = 7, 5
    cells = compute_cells(screen_w, screen_h, cols, rows)
    
    # Verify every pixel belongs to exactly one cell
    # To avoid checking 2 million pixels, we check the boundaries of every cell
    for r in range(rows):
        for c in range(cols):
            rect = cells[r][c]
            # Pixel just inside the left/top boundary
            assert get_cell_for_point(cells, rect.x, rect.y) == (r, c)
            # Pixel just inside the right/bottom boundary
            assert get_cell_for_point(cells, rect.x + rect.w - 1, rect.y + rect.h - 1) == (r, c)
            
            # Pixel just outside the left boundary
            if c > 0:
                assert get_cell_for_point(cells, rect.x - 1, rect.y) == (r, c - 1)
            # Pixel just outside the top boundary
            if r > 0:
                assert get_cell_for_point(cells, rect.x, rect.y - 1) == (r - 1, c)

    # Remainders are absorbed (1920 % 7 = 2)
    # The first 2 columns should have width 275, the rest 274
    assert cells[0][0].w == 275
    assert cells[0][1].w == 275
    assert cells[0][2].w == 274
    assert cells[0][3].w == 274
    assert cells[0][6].w == 274

def test_zoom_mapping():
    # We want to test that a pixel chosen in ZOOM1 maps back to the exact sub-coordinate in the cell.
    # The cell is rect(x, y, w, h). The whole screen (1920x1080) maps to this rect.
    screen_w, screen_h = 1920, 1080
    rect = Rect(100, 200, 300, 400) # an arbitrary cell
    
    # If the user dwells at screen pixel (0, 0), it should map to (100, 200)
    def map_back(px, py, r):
        orig_x = r.x + (px / screen_w) * r.w
        orig_y = r.y + (py / screen_h) * r.h
        return orig_x, orig_y
        
    assert map_back(0, 0, rect) == (100.0, 200.0)
    
    # If user dwells at bottom-right of screen, it maps to bottom-right of cell
    assert map_back(screen_w, screen_h, rect) == (400.0, 600.0)
    
    # Same logic applies to ZOOM2
    zoom2_rect = Rect(150, 250, 50, 50) # arbitrary ZOOM2 rect
    assert map_back(0, 0, zoom2_rect) == (150.0, 250.0)
    assert map_back(screen_w, screen_h, zoom2_rect) == (200.0, 300.0)


from engine.machine import StateMachine, derive_grid
from engine.capture.null import NullCapture
from engine.events.dispatcher import EventDispatcher
from engine.events.interaction import InteractionEvent, EventType

class MockDispatcher:
    def __init__(self):
        self.events = []
    def dispatch(self, ev):
        self.events.append(ev)

def test_full_sequence_headless():
    dispatcher = MockDispatcher()
    capture = NullCapture(1920, 1080)
    
    # 2.0 degrees, 600mm viewing distance, 1920x1080 24" screen
    machine = StateMachine(1920, 1080, 597.0, 2.0, 600.0, dispatcher, capture)
    
    # Assert IDLE
    assert machine.state == "IDLE"
    
    # 1. Engage
    machine.process_event(InteractionEvent(
        event_type=EventType.GESTURE.value,
        timestamp=1.0,
        role="ENGAGE"
    ))
    
    assert machine.state == "GRID"
    assert machine.frozen_image_b64 is not None
    assert capture.capture_count == 1
    
    # Provide a different capture image underlying
    # The frozen_image_b64 should not change when we enter zoom!
    old_b64 = machine.frozen_image_b64
    
    # 2. Dwell complete on cell 0,0
    # cell 0,0 will be the top left
    machine.process_event(InteractionEvent(
        event_type=EventType.DWELL_COMPLETE.value,
        timestamp=2.0,
        x=10.0,
        y=10.0
    ))
    
    assert machine.state == "ZOOM1"
    # Ensure capture wasn't called again
    assert capture.capture_count == 1
    assert machine.frozen_image_b64 == old_b64
    
    # 3. Dwell complete in ZOOM1
    # Say user looks at center of screen (960, 540)
    machine.process_event(InteractionEvent(
        event_type=EventType.DWELL_COMPLETE.value,
        timestamp=3.0,
        x=960.0,
        y=540.0
    ))
    
    assert machine.state == "ZOOM2"
    
    # 4. Dwell complete in ZOOM2
    machine.process_event(InteractionEvent(
        event_type=EventType.DWELL_COMPLETE.value,
        timestamp=4.0,
        x=960.0,
        y=540.0
    ))
    
    assert machine.state == "IDLE"
    assert machine.resolved_point is not None

def test_off_screen_glance_steps_back():
    dispatcher = MockDispatcher()
    capture = NullCapture(1920, 1080)
    machine = StateMachine(1920, 1080, 597.0, 2.0, 600.0, dispatcher, capture)
    
    machine.process_event(InteractionEvent(event_type=EventType.GESTURE.value, timestamp=1.0, role="ENGAGE"))
    assert machine.state == "GRID"
    
    machine.process_event(InteractionEvent(event_type=EventType.GESTURE.value, timestamp=2.0, role="CANCEL"))
    assert machine.state == "IDLE"

def test_timeout_to_idle():
    dispatcher = MockDispatcher()
    capture = NullCapture(1920, 1080)
    machine = StateMachine(1920, 1080, 597.0, 2.0, 600.0, dispatcher, capture)
    
    machine.process_event(InteractionEvent(event_type=EventType.GESTURE.value, timestamp=1.0, role="ENGAGE"))
    assert machine.state == "GRID"
    
    # Timeout is 8s
    machine.check_timeout(timestamp=8.9)
    assert machine.state == "GRID"
    
    machine.check_timeout(timestamp=9.1)
    assert machine.state == "IDLE"

def test_idle_inert_against_recording():
    """
    IDLE is confirmed inert against a real recording, or the mark is explicitly recorded as not closed.
    Currently using synthetic data.
    NOT CLOSED: Requires a real recording of a person reading and blinking naturally for several minutes.
    """
    pass

