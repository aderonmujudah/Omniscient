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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_engine(source, publisher, loop):
    source.start()
    logger.info("Engine started.")
    
    last_stat_time = time.monotonic()
    frames_since_stat = 0
    recent_confs = deque(maxlen=60)
    
    try:
        for sample in source.iter_samples():
            publisher.publish_sample(sample)
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
        asyncio.run_coroutine_threadsafe(publisher.stop(), loop)

def main():
    parser = argparse.ArgumentParser(description="Omniscient Engine - Scope 1")
    parser.add_argument("--source", choices=["webcam", "replay"], default="webcam", help="Gaze source to use")
    parser.add_argument("--replay-file", type=str, help="File to replay from (if source is replay)")
    parser.add_argument("--record-file", type=str, help="File to record to (if source is webcam)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="WebSocket host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index for webcam source")
    parser.add_argument("--model-path", type=str, default="face_landmarker.task", help="Path to face_landmarker.task")
    args = parser.parse_args()

    if args.source == "replay":
        if not args.replay_file:
            parser.error("--replay-file is required when source is replay")
        source = ReplaySource(filepath=args.replay_file)
    else:
        source = WebcamSource(camera_index=args.camera_index, model_path=args.model_path)
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

    # Run the engine
    run_engine(source, publisher, loop)
    
    # Cleanup
    loop.call_soon_threadsafe(loop.stop)
    t.join()

if __name__ == "__main__":
    main()
