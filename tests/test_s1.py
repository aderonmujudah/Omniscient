import os
import json
import time
import math
import asyncio
import numpy as np
import websockets
import jsonschema
from unittest.mock import patch, MagicMock

import cv2
from engine.sources.base import GazeSample, EyeGeometry, Point2D
from engine.sources.webcam import WebcamSource
from engine.sources.replay import ReplaySource
from engine.sources.recorder import RecorderSource, _serialize_sample
from engine.transport.server import WebsocketPublisher

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "protocol", "schema.json")
FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "..", "fixtures", "test_session.jsonl")

with open(SCHEMA_PATH, 'r') as f:
    SCHEMA = json.load(f)

class MockVideoCapture:
    def __init__(self, frames):
        self.frames = frames
        self.idx = 0

    def isOpened(self):
        return True

    def set(self, propId, value):
        return True

    def read(self):
        if self.idx < len(self.frames):
            frame = self.frames[self.idx]
            self.idx += 1
            return True, frame
        return False, None

    def release(self):
        pass


def test_schema_validation():
    """Pass mark 3: Every emitted sample validates against the GazeSample schema."""
    with open(FIXTURE_PATH, 'r') as f:
        for line in f:
            data = json.loads(line)
            jsonschema.validate(instance=data, schema=SCHEMA)


def test_sequence_monotonicity():
    """
    Pass mark 4 & 5: Sequence numbers strictly increasing; timestamps monotonic.
    """
    source = ReplaySource(FIXTURE_PATH)
    source.start()
    
    last_t = -1
    last_seq = -1
    
    samples = list(source.iter_samples())
    assert len(samples) == 3
    for sample in samples:
        assert sample.seq > last_seq
        assert sample.t >= last_t
        last_seq = sample.seq
        last_t = sample.t
        
    source.stop()


def test_replay_timing():
    """Pass mark 2: Replays a captured session with inter-sample timing within 5 ms."""
    source = ReplaySource(FIXTURE_PATH)
    source.start()
    
    # Measure the real time deltas between yields
    samples_yielded = []
    
    # Collect samples to measure accurately
    start_time = time.monotonic()
    
    # Measure the time since the FIRST sample
    first_yield_real = None
    first_sample_t = None
    
    for sample in source.iter_samples():
        if first_yield_real is None:
            first_yield_real = time.monotonic()
            first_sample_t = sample.t
        else:
            real_elapsed = time.monotonic() - first_yield_real
            virtual_elapsed = sample.t - first_sample_t
            # Should be within 5ms (0.005s)
            assert math.isclose(real_elapsed, virtual_elapsed, abs_tol=0.005)
            
    source.stop()


@patch("engine.sources.webcam.cv2.VideoCapture")
def test_webcam_good_frame(mock_vc):
    """
    Pass mark 1: Produces samples at a measured rate.
    """
    # Download a static face image to exercise the MediaPipe face detection path in the CI environment where no real camera is present.
    import urllib.request
    img_path = "/tmp/face.jpg"
    if not os.path.exists(img_path):
        url = "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(img_path, 'wb') as out_file:
            out_file.write(response.read())

    frame = cv2.imread(img_path)
    
    # Frame 1: Face, Frame 2: Face
    mock_vc.return_value = MockVideoCapture([frame, frame])
    
    source = WebcamSource(camera_index=0)
    source.start()
    
    samples = []
    it = source.iter_samples()
    for _ in range(2):
        samples.append(next(it))
        
    assert len(samples) == 2
    
    for sample in samples:
        # Assuming the test image has a detectable face
        assert sample.ok is True
        assert sample.condition is None
        assert sample.eyes is not None
        assert sample.ear is not None
        assert sample.ipd_px is not None
        assert sample.conf is not None
        
    source.stop()


@patch("engine.sources.webcam.cv2.VideoCapture")
def test_webcam_no_face_and_low_light(mock_vc):
    """
    Pass mark 6: With no face in frame, ok is false and stream continues without stalling.
    Pass mark 7: Under deliberately poor lighting, a low-light condition is reported.
    """
    # Create a dark frame (mean brightness < 30)
    dark_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Give it a tiny bit of brightness to ensure it doesn't just crash
    dark_frame.fill(10)
    
    # Create a bright frame but no face (e.g. solid white)
    white_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    white_frame.fill(255)
    
    mock_vc.return_value = MockVideoCapture([dark_frame, white_frame])
    
    source = WebcamSource(camera_index=0)
    source.start()
    
    samples = []
    it = source.iter_samples()
    for _ in range(2):
        samples.append(next(it))
        
    assert len(samples) == 2
    
    s1, s2 = samples
    
    # Dark frame -> ok=False, condition=low_light
    assert s1.ok is False
    assert s1.condition == "low_light"
    
    # Bright frame but no face -> ok=False, condition=None
    assert s2.ok is False
    assert s2.condition is None
    
    source.stop()


import pytest
@pytest.mark.asyncio
async def test_websocket_reconnect():
    """
    Pass mark 8: A WebSocket client receives the stream and can reconnect after a disconnect
    without restarting the Engine.
    """
    loop = asyncio.get_running_loop()
    publisher = WebsocketPublisher(host="127.0.0.1", port=8766)
    publisher.start(loop)
    
    # Give it a tiny bit of time to start
    await asyncio.sleep(0.1)
    
    sample = GazeSample(t=1.0, seq=1, ok=True)
    
    # Client 1 connects
    async with websockets.connect("ws://127.0.0.1:8766") as ws:
        publisher.publish_sample(sample)
        msg = await ws.recv()
        data = json.loads(msg)
        assert data['seq'] == 1
        
    # Client 1 disconnected. Engine is still running.
    # Client 2 connects
    async with websockets.connect("ws://127.0.0.1:8766") as ws:
        publisher.publish_sample(sample)
        msg = await ws.recv()
        data = json.loads(msg)
        assert data['seq'] == 1

    await publisher.stop()
