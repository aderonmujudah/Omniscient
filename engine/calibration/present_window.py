import argparse
import ctypes
import logging
import queue
import sys
from typing import Optional
import threading
import tkinter as tk

from engine.calibration.labels import make_label_provider
from engine.calibration.presenter import PresenterLogic
from engine.calibration.session import CalibrationSession
from engine.sources.recorder import RecorderSource
from engine.sources.webcam import WebcamSource

logger = logging.getLogger(__name__)

DOT_RADIUS_PX = 12
BACKGROUND_COLOUR = "#101010"
DOT_COLOUR = "#f0f0f0"
POLL_INTERVAL_MS = 16
CLOSE_DELAY_MS = 500
MESSAGE_FONT_PX = 28

# Seconds between the window opening and the first target. The sequence allows SETTLE_TIME_S of
# 0.3 s per target, which is enough to settle a gaze already on the screen and not enough to
# find one, so the opening target needs its own allowance rather than a longer settle for every
# target. It also lets the landmarker finish loading before anything is measured.
LEAD_IN_S = 6


def declare_dpi_awareness() -> None:
    """
    Opts the process out of DPI virtualisation before any window is created.

    Windows reports a scaled-down logical resolution to a process that has not declared
    awareness, so on a display running a scale factor the screen appears smaller than the panel
    is. The targets would then be placed in logical pixels while the physical diagonal that
    converts an error into degrees describes the panel, and the two disagree by the scale
    factor. Declared here rather than left to a manifest so the presenter is correct however it
    is launched.
    """
    if sys.platform != "win32":
        return
    try:
        # PROCESS_PER_MONITOR_DPI_AWARE.
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            logger.warning("Could not declare DPI awareness. Screen dimensions may be scaled.")


class PresenterWindow:
    """
    Full screen opaque window that draws the calibration target.

    Deliberately not the transparent click-through overlay: this window owns the display and
    nothing behind it matters, so the two share no requirement. The state machine places
    targets at margins of the screen dimensions it is given, so those dimensions must be the
    real ones. A window inset by a title bar puts every target at the wrong physical position
    and biases the measurement this exists to produce.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.logic = PresenterLogic()

        self.width = root.winfo_screenwidth()
        self.height = root.winfo_screenheight()

        root.attributes("-fullscreen", True)
        root.configure(background=BACKGROUND_COLOUR, cursor="none")
        self.canvas = tk.Canvas(root, width=self.width, height=self.height,
                                background=BACKGROUND_COLOUR, highlightthickness=0,
                                cursor="none")
        self.canvas.pack(fill="both", expand=True)
        self.dot_id = None
        self.message_id = None

    def render(self) -> None:
        if self.dot_id is not None:
            self.canvas.delete(self.dot_id)
            self.dot_id = None
        dot = self.logic.dot
        if dot.visible:
            self.dot_id = self.canvas.create_oval(
                dot.x - DOT_RADIUS_PX, dot.y - DOT_RADIUS_PX,
                dot.x + DOT_RADIUS_PX, dot.y + DOT_RADIUS_PX,
                fill=DOT_COLOUR, outline="")

    def show_message(self, text: str) -> None:
        """Centred text for the lead-in. Cleared before the first target so that nothing but
        the target is on screen while a window is being collected."""
        if self.message_id is not None:
            self.canvas.delete(self.message_id)
            self.message_id = None
        if text:
            self.message_id = self.canvas.create_text(
                self.width / 2, self.height / 2, text=text, fill=DOT_COLOUR,
                font=("TkDefaultFont", MESSAGE_FONT_PX), justify="center")

    def apply(self, event) -> None:
        if self.logic.apply(event):
            self.render()


def run_calibration(record_path: str, camera_index: int = 0,
                    dispersion_threshold: Optional[float] = None) -> None:
    """
    Presents the calibration sequence and records the session with its target labels.

    Capture runs on a worker thread and the window is updated from the main thread, because a
    blocking capture loop on the main thread would stop the display from repainting between
    targets.
    """
    declare_dpi_awareness()
    root = tk.Tk()
    window = PresenterWindow(root)

    logger.info("Presenting on %dx%d", window.width, window.height)

    session = CalibrationSession(window.width, window.height, dispersion_threshold)
    logger.info("Accepting windows at dispersion threshold %.4f", session.dispersion_threshold)
    source = RecorderSource(
        inner=WebcamSource(camera_index=camera_index),
        filepath=record_path,
        label_provider=make_label_provider(session),
    )

    events: queue.Queue = queue.Queue()
    stop = threading.Event()
    ready = threading.Event()

    def capture() -> None:
        try:
            source.start()
            # Opening the source warms the camera and the landmarker without reading a sample,
            # so the lead-in costs the subject nothing and the first target is presented to
            # someone already looking at the screen.
            ready.wait()
            # The session must be presenting before the first sample is written, or the
            # opening samples would be recorded with no label.
            events.put(session.start())
            for sample in source.iter_samples():
                if stop.is_set():
                    break
                event = session.process_sample(sample)
                if event:
                    events.put(event)
                    if event["type"] in ("CALIBRATION_DONE", "CALIBRATION_FAILED"):
                        break
        except Exception:
            logger.exception("Capture stopped before the sequence completed")
            events.put({"type": "CALIBRATION_FAILED"})
        finally:
            source.stop()

    thread = threading.Thread(target=capture, daemon=True)
    thread.start()

    def poll() -> None:
        while True:
            try:
                window.apply(events.get_nowait())
            except queue.Empty:
                break
        if window.logic.is_complete():
            root.after(CLOSE_DELAY_MS, root.destroy)
            return
        root.after(POLL_INTERVAL_MS, poll)

    def abort(_event=None) -> None:
        stop.set()
        # Releases the capture thread if the sequence is abandoned during the lead-in, which
        # would otherwise leave it parked on the event with the camera open.
        ready.set()
        root.destroy()

    def lead_in(remaining: int) -> None:
        if stop.is_set():
            return
        if remaining > 0:
            window.show_message("Look at the centre of the screen and hold still.\n\n"
                                "Follow each dot as it appears.\n\n"
                                "Starting in %d" % remaining)
            root.after(1000, lead_in, remaining - 1)
            return
        window.show_message("")
        ready.set()
        root.after(POLL_INTERVAL_MS, poll)

    root.bind("<Escape>", abort)
    root.after(POLL_INTERVAL_MS, lead_in, LEAD_IN_S)
    root.mainloop()

    stop.set()
    thread.join(timeout=2.0)

    if window.logic.failed:
        logger.error("Sequence did not complete. The recording is at %s", record_path)
    else:
        logger.info("Recording written to %s", record_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run a calibration sequence and record it.")
    parser.add_argument("--record-file", required=True, help="Path to write the recording to")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index")
    parser.add_argument("--dispersion-threshold", type=float, default=None,
                        help="Override the live acceptance threshold. The value used is recorded "
                             "with every sample.")
    args = parser.parse_args()
    run_calibration(args.record_file, args.camera_index, args.dispersion_threshold)
