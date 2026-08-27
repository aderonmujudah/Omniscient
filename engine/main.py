import argparse
import asyncio
import logging
import threading
import time
from collections import deque
from engine.sources.webcam import WebcamSource
from engine.sources.replay import ReplaySource
from engine.sources.recorder import RecorderSource
from engine.transport.server import WebsocketPublisher
from engine.calibration.model import CalibrationModel
from engine.calibration.store import load_profile
from engine.events.dispatcher import EventDispatcher
from engine.events.dwell import DwellTimer
from engine.events.emitter import InteractionEmitter
from engine.events.gestures.registry import GestureRegistry
from engine.features.eye_features import extract_features
from engine.filtering.classifier import SampleClassifier
from engine.filtering.fixation import FixationDetector
from engine.filtering.one_euro import OneEuroFilter
from engine.calibration.online import OnlineRecalibrator
from engine.capture.null import NullCapture
from engine.input.null import NullInput
from engine.machine import StateMachine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Reserved zones are a property of the screen rather than of the user, so they are defined
# here rather than persisted per profile. Fractions of the screen, not pixels, so a profile
# calibrated on one display resolves correctly on another.
# Viewing distance is a measured per-user quantity. Until it is measured it is defaulted here,
# under a name that does not claim otherwise, because every on-screen geometry is sized off it.
DEFAULT_VIEWING_DIST_MM = 600.0

RESERVED_ZONES = {
    "engage": {"x": 0.0, "y": 0.92, "w": 0.08, "h": 0.08},
    "cancel": {"x": 0.92, "y": 0.0, "w": 0.08, "h": 0.08},
    "menu": {"x": 0.0, "y": 0.0, "w": 0.08, "h": 0.08},
}


def build_emitter(profile, dispatcher, rate, capture_backend, input_backend):
    """Compose the signal and event layers from a calibration profile.

    Returns the emitter, the calibration model, and the state machine, or (None, None, None) when the profile carries
    no mapping. Without a mapping there is no screen position, so no gesture that depends on
    one can run and no cursor can be drawn.
    """
    model_data = profile.get("model") or {}
    if not model_data.get("coeffs_x"):
        return None, None, None

    model = CalibrationModel()
    model.load_dict(model_data)

    screen = profile.get("screen") or {}
    screen_w = int(screen.get("w", 1920))
    screen_h = int(screen.get("h", 1080))
    screen_diag_mm = float(screen.get("diag_mm", 597))

    validation = profile.get("validation") or {}
    acc_deg = validation.get("mean_error_deg", 2.0)
    if acc_deg <= 0.0:
        acc_deg = 2.0

    viewing_dist_mm = float(screen.get("viewing_dist_mm") or 0.0)
    if viewing_dist_mm <= 0.0:
        viewing_dist_mm = DEFAULT_VIEWING_DIST_MM
        logger.warning(
            "Profile carries no viewing distance; falling back to %.0f mm. The grid dimension and "
            "the radial deadzone are both derived from it, so they are provisional until measured.",
            DEFAULT_VIEWING_DIST_MM)

    gestures = profile.get("gestures") or {}
    roles = gestures.get("roles") or {}
    gesture_params = {
        entry["id"]: entry.get("params") or {}
        for entry in gestures.get("assessed") or []
        if entry.get("id")
    }

    registry = GestureRegistry(
        role_assignment=roles,
        screen_w=screen_w,
        screen_h=screen_h,
        reserved_zones=RESERVED_ZONES,
        gesture_params=gesture_params,
        closure_threshold_ms=(profile.get("blink") or {}).get("long_threshold_ms"),
    )

    recalibrator = OnlineRecalibrator(model)

    machine = StateMachine(
        screen_w=screen_w,
        screen_h=screen_h,
        screen_diag_mm=screen_diag_mm,
        acc_deg=acc_deg,
        viewing_dist_mm=viewing_dist_mm,
        dispatcher=dispatcher,
        capture_backend=capture_backend,
        input_backend=input_backend,
        recalibrator=recalibrator,
        reserved_zones=RESERVED_ZONES
    )

    dispatcher.subscribe(machine.process_event)

    emitter = InteractionEmitter(
        gaze_filter=OneEuroFilter(rate=rate),
        classifier=SampleClassifier(fixation_detector=FixationDetector()),
        registry=registry,
        dispatcher=dispatcher,
        dwell=DwellTimer(),
        zone_resolver=machine.resolve_zone
    )
    return emitter, model, machine

def calibrated_position(model, sample):
    """Map a sample to screen coordinates, or None when its geometry is unusable.
    Returns (x, y, fx, fy).
    """
    if model is None or not sample.ok or not sample.eyes:
        return None
    left = sample.eyes.get("left")
    right = sample.eyes.get("right")
    if left is None or right is None:
        return None
    fx, fy = extract_features(left, right)
    x, y = model.predict(fx, fy)
    return x, y, fx, fy


def run_engine(source, publisher, loop, emitter=None, model=None, machine=None):
    source.start()
    logger.info("Engine started.")
    
    last_stat_time = time.monotonic()
    frames_since_stat = 0
    recent_confs = deque(maxlen=60)
    
    try:
        for sample in source.iter_samples():
            publisher.publish_sample(sample)

            if emitter is not None:
                position = calibrated_position(model, sample)
                if position is None:
                    emitter.process_sample(sample, None, None)
                else:
                    emitter.process_sample(sample, position[0], position[1])
                    if machine is not None:
                        machine.latest_features = (position[2], position[3])

            if machine is not None:
                machine.check_timeout(sample.t)
                if machine.shutdown_requested:
                    logger.info("Shutdown requested from the system menu.")
                    break

            frames_since_stat += 1
            if sample.conf is not None:
                recent_confs.append(sample.conf)

            now = time.monotonic()
            if now - last_stat_time >= 1.0:
                fps = frames_since_stat / (now - last_stat_time)
                
                dropped = 0
                if isinstance(source, WebcamSource):
                    dropped = source.dropped_frames
                elif isinstance(source, RecorderSource) and isinstance(source.inner, WebcamSource):
                    dropped = source.inner.dropped_frames
                
                avg_conf = sum(recent_confs) / len(recent_confs) if recent_confs else 0.0
                
                logger.info(f"Stats - FPS: {fps:.1f} | Dropped: {dropped} | Avg Conf: {avg_conf:.2f}")
                
                frames_since_stat = 0
                last_stat_time = now

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        source.stop()
        if machine is not None:
            machine.capture.close()
        asyncio.run_coroutine_threadsafe(publisher.stop(), loop)

def main():
    parser = argparse.ArgumentParser(description="Omniscient Engine")
    parser.add_argument("--source", choices=["webcam", "replay"], default="webcam", help="Gaze source to use")
    parser.add_argument("--replay-file", type=str, help="File to replay from (if source is replay)")
    parser.add_argument("--record-file", type=str, help="File to record to (if source is webcam)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="WebSocket host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index for webcam source")
    parser.add_argument("--model-path", type=str, default="face_landmarker.task", help="Path to face_landmarker.task")
    parser.add_argument("--profile", type=str, help="Calibration profile to load (defaults to the stored profile)")
    parser.add_argument("--rate", type=float, default=30.0, help="Expected sample rate in Hz, for filter tuning")
    args = parser.parse_args()

    if args.source == "replay":
        if not args.replay_file:
            parser.error("--replay-file is required when source is replay")
        source = ReplaySource(filepath=args.replay_file)
        capture_backend = NullCapture()
        input_backend = NullInput()
    else:
        source = WebcamSource(camera_index=args.camera_index, model_path=args.model_path)
        from engine.capture.windows import WindowsCapture
        from engine.input.windows import WindowsInput
        capture_backend = WindowsCapture()
        input_backend = WindowsInput()
        if args.record_file:
            source = RecorderSource(inner=source, filepath=args.record_file)

    # Setup WebSocket publisher
    loop = asyncio.new_event_loop()
    publisher = WebsocketPublisher(host=args.host, port=args.port)
    
    # Run the event loop in a background thread
    def loop_thread():
        asyncio.set_event_loop(loop)
        loop.run_forever()
        
    t = threading.Thread(target=loop_thread, daemon=True)
    t.start()
    
    publisher.start(loop)

    dispatcher = EventDispatcher()
    dispatcher.set_ws_broadcast(publisher.publish_event)

    profile = load_profile(args.profile) if args.profile else load_profile()
    emitter, model, machine = build_emitter(profile, dispatcher, args.rate, capture_backend, input_backend)
    if emitter is None:
        logger.warning(
            "No calibration mapping found. Publishing raw samples only; no interaction "
            "events will be emitted until a profile is calibrated."
        )

    run_engine(source, publisher, loop, emitter=emitter, model=model, machine=machine)
    
    # Cleanup
    loop.call_soon_threadsafe(loop.stop)
    t.join()

if __name__ == "__main__":
    main()
