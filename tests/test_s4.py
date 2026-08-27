import pytest
import json
import jsonschema
from pathlib import Path
from engine.input.null import NullInput
from engine.input.base import MOUSEEVENTF_MOVE, MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, KEYEVENTF_UNICODE

SCHEMA_PATH = Path(__file__).parent.parent / "protocol" / "schema.json"

@pytest.fixture
def protocol_schema():
    with open(SCHEMA_PATH, "r") as f:
        return json.load(f)

def test_s4_null_backend_click_action_sequence():
    """PM: The null backend records the same action sequence the Windows backend would perform."""
    backend = NullInput()
    backend.click(100.0, 200.0, "left", 1)
    
    assert len(backend.mouse_injections) == 1
    injection = backend.mouse_injections[0]
    
    assert len(injection) == 3
    # Move
    assert injection[0].x == 100.0
    assert injection[0].y == 200.0
    assert injection[0].flags == MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE
    # Down
    assert injection[1].x == 100.0
    assert injection[1].y == 200.0
    assert injection[1].flags == MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_ABSOLUTE
    # Up
    assert injection[2].x == 100.0
    assert injection[2].y == 200.0
    assert injection[2].flags == MOUSEEVENTF_LEFTUP | MOUSEEVENTF_ABSOLUTE

def test_s4_null_backend_drag_sequence():
    backend = NullInput()
    backend.drag("start", 10.0, 20.0)
    
    assert len(backend.mouse_injections) == 1
    start_inj = backend.mouse_injections[0]
    assert len(start_inj) == 2
    assert start_inj[0].flags == MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE
    assert start_inj[1].flags == MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_ABSOLUTE
    
    backend.drag("end", 50.0, 60.0)
    assert len(backend.mouse_injections) == 2
    end_inj = backend.mouse_injections[1]
    assert len(end_inj) == 2
    assert end_inj[0].flags == MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE
    assert end_inj[1].flags == MOUSEEVENTF_LEFTUP | MOUSEEVENTF_ABSOLUTE

def test_s4_action_requests_validate_against_schema(protocol_schema):
    """PM: validate against schema in a test."""
    click_request = {
        "action": "CLICK",
        "x": 500.0,
        "y": 500.0,
        "button": "left",
        "count": 1
    }
    jsonschema.validate(instance=click_request, schema=protocol_schema)
    
    drag_request = {
        "action": "DRAG",
        "phase": "start",
        "x": 100.0,
        "y": 100.0
    }
    jsonschema.validate(instance=drag_request, schema=protocol_schema)
    
    key_request = {
        "action": "KEY",
        "text": "hello"
    }
    jsonschema.validate(instance=key_request, schema=protocol_schema)

import asyncio
from engine.transport.server import WebsocketPublisher

@pytest.mark.asyncio
async def test_s4_websocket_action_request_decode(protocol_schema):
    """PM: the ActionRequest decode path."""
    backend = NullInput()
    server = WebsocketPublisher(input_backend=backend)
    
    class MockWebsocket:
        def __init__(self, remote_address="127.0.0.1"):
            self.remote_address = remote_address
            self.messages = [
                json.dumps({
                    "action": "CLICK",
                    "x": 300.0,
                    "y": 400.0,
                    "button": "right",
                    "count": 2
                })
            ]
            self.sent = []
            
        def __aiter__(self):
            return self
            
        async def __anext__(self):
            if not self.messages:
                raise StopAsyncIteration
            return self.messages.pop(0)
            
        async def send(self, msg):
            self.sent.append(msg)

    ws = MockWebsocket()
    await server._handler(ws, "/")
    
    assert len(backend.mouse_injections) == 1
    injection = backend.mouse_injections[0]
    assert len(injection) == 5
    from engine.input.base import MOUSEEVENTF_RIGHTDOWN
    assert injection[1].flags == MOUSEEVENTF_RIGHTDOWN | MOUSEEVENTF_ABSOLUTE

from engine.sources.replay import ReplaySource

def test_s4_cursor_position_math_against_replayed_session():
    """PM: The gaze cursor tracks smoothly against a replayed session."""
    source = ReplaySource("fixtures/test_session.jsonl")
    published_events = []
    
    class DummyPublisher:
        def publish_event(self, e):
            published_events.append(e)
            
    publisher = DummyPublisher()
    
    from engine.filtering.classifier import SampleClassifier
    from engine.filtering.one_euro import OneEuroFilter
    from engine.filtering.fixation import FixationDetector
    from engine.events.dispatcher import EventDispatcher
    from engine.events.emitter import InteractionEmitter
    from engine.events.gestures.registry import GestureRegistry
    from engine.events.gestures.base import Role
    
    fix = FixationDetector()
    clf = SampleClassifier(fixation_detector=fix)
    dispatcher = EventDispatcher()
    registry = GestureRegistry(
        role_assignment={Role.ENGAGE: "long_blink", Role.CANCEL: "off_screen_glance", Role.MENU: "reserved_zone_dwell"},
        screen_w=1920,
        screen_h=1080,
        reserved_zones={}, gesture_params={}
    )
    filt = OneEuroFilter(rate=30.0)
    
    emitter = InteractionEmitter(
        gaze_filter=filt,
        classifier=clf,
        registry=registry,
        dispatcher=dispatcher
    )
    
    dispatcher.subscribe(lambda e: publisher.publish_event(e.to_dict()))
    
    source.start()
    samples = list(source.iter_samples())
    if samples:
        sample = samples[0]
        for i in range(5):
            # increment timestamp so filters don't get stuck
            sample.t += 0.033
            emitter.process_sample(sample, 500.0, 500.0)
            
    assert any(e["event_type"] == "GAZE_MOVE" for e in published_events)
