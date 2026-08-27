import logging
from engine.events.gestures.long_blink import LongBlinkDetector
from engine.events.gestures.extended_closure import ExtendedClosureDetector
from engine.events.gestures.off_screen_glance import OffScreenGlanceDetector
from engine.events.gestures.smooth_pursuit import SmoothPursuitDetector
from engine.events.gestures.gaze_stroke import GazeStrokeDetector

logger = logging.getLogger(__name__)

# The declared candidate set. The set actually presented to a user is derived from this
# in GestureAssessment, excluding any detector that cannot fire or whose input the
# caller cannot supply. A gesture that cannot fire must never be presented, because
# every attempt would be recorded as a failure the user caused.
CANDIDATE_GESTURES = [
    "long_blink",
    "extended_closure",
    "off_screen_glance",
    "smooth_pursuit",
    "gaze_stroke",
]

class GestureAssessment:
    def __init__(
        self,
        long_blink_threshold_ms: float = 450.0,
        screen_w: int = 1920,
        screen_h: int = 1080,
        *,
        gaze_position_available: bool,
    ):
        self.long_blink_threshold_ms = long_blink_threshold_ms
        ear_threshold = 0.2
        self.detectors = {
            "long_blink": LongBlinkDetector(
                ear_threshold=ear_threshold,
                closure_min_s=long_blink_threshold_ms / 1000.0,
            ),
            "extended_closure": ExtendedClosureDetector(ear_threshold=ear_threshold),
            "off_screen_glance": OffScreenGlanceDetector(screen_w=screen_w, screen_h=screen_h),
            "smooth_pursuit": SmoothPursuitDetector(),
            "gaze_stroke": GazeStrokeDetector(),
        }
        
        self.gaze_position_available = gaze_position_available
        self.candidates = [
            name for name in CANDIDATE_GESTURES
            if self.detectors[name].can_fire
            and (gaze_position_available or not self.detectors[name].requires_gaze_position)
        ]
        for name in CANDIDATE_GESTURES:
            if name not in self.candidates:
                logger.info("Gesture %s is not presented as a candidate in this session.", name)

        self.results = []
        self.current_idx = 0
        
        # State can be EXPLAIN, TEST_ACTIVE, TEST_CONTROL, DONE
        self.state = "IDLE"
        self.attempts = 0
        self.max_attempts = 3
        self.successes = 0
        self.false_positives = 0
        
        self.state_start_t = 0.0
        # Control duration lengthened to 10 seconds so a single spurious event (0.1/sec) 
        # does not permanently disable a gesture under a 0.15/sec ceiling.
        self.control_duration_s = 10.0
        
    def start(self):
        self.current_idx = 0
        self.results = []
        if self.current_idx < len(self.candidates):
            self._transition_to("EXPLAIN", 0.0)
        else:
            self._transition_to("DONE", 0.0)
        return self._get_event()
        
    def _transition_to(self, state: str, current_t: float):
        self.state = state
        self.state_start_t = current_t
        if state in ("TEST_ACTIVE", "TEST_CONTROL"):
            self.detectors[self.candidates[self.current_idx]].reset()

    def _get_event(self):
        if self.state == "DONE":
            return {"type": "ASSESSMENT_DONE"}
        
        current_gesture = self.candidates[self.current_idx]
        return {
            "type": "ASSESSMENT_STATE",
            "gesture": current_gesture,
            "state": self.state,
            "attempt": self.attempts + 1 if self.state == "TEST_ACTIVE" else None
        }

    def user_declines(self, current_t: float = 0.0):
        """User explicitly declines the current candidate."""
        if self.state in ("EXPLAIN", "TEST_ACTIVE", "TEST_CONTROL"):
            current_gesture = self.candidates[self.current_idx]
            
            success_rate = self.successes / self.max_attempts if self.max_attempts > 0 else 0.0
            elapsed = current_t - self.state_start_t
            control_elapsed = elapsed if self.state == "TEST_CONTROL" and elapsed > 0 else 0.0
            fp_rate = self.false_positives / self.control_duration_s
            
            self.results.append({
                "id": current_gesture,
                "success": success_rate,
                "attempts": self.attempts,
                "false_positive": fp_rate,
                "control_elapsed_s": control_elapsed,
                "enabled": False,
                "declined_by_user": True,
                "params": {}
            })
            self._next_gesture(current_t)
            return self._get_event()

    def _detector_params(self, gesture_name):
        """Parameters read back from the detector that will run at dispatch time."""
        detector = self.detectors[gesture_name]
        if gesture_name == "long_blink":
            return {
                "threshold_ms": detector.closure_min_s * 1000.0,
                "closure_max_ms": detector.closure_max_s * 1000.0,
            }
        return {}

    def _next_gesture(self, current_t: float):
        self.current_idx += 1
        self.attempts = 0
        self.successes = 0
        self.false_positives = 0
        if self.current_idx >= len(self.candidates):
            self._transition_to("DONE", current_t)
        else:
            self._transition_to("EXPLAIN", current_t)

    def user_ready(self, current_t: float = 0.0):
        if self.state == "EXPLAIN":
            self._transition_to("TEST_ACTIVE", current_t)
            return self._get_event()

    def process_sample(self, sample, *, gaze_x, gaze_y):
        """Advance the assessment by one sample.

        gaze_x and gaze_y are the calibrated screen coordinates for this sample, or None
        when tracking is lost. They are required rather than optional because a gaze
        position gesture whose detector never receives a position cannot fire at all.
        """
        now = sample.t
        if self.state_start_t == 0.0:
            self.state_start_t = now
            
        if self.state == "DONE":
            return None
            
        current_gesture = self.candidates[self.current_idx]
        detector = self.detectors[current_gesture]
        
        if gaze_x is not None and gaze_y is not None:
            detector.update_gaze_position(gaze_x, gaze_y)

        detected = detector.process_sample(sample)
        
        if self.state == "TEST_ACTIVE":
            if detected == current_gesture:
                self.successes += 1
                self.attempts += 1
                if self.attempts >= self.max_attempts:
                    self._transition_to("TEST_CONTROL", now)
                else:
                    self._transition_to("TEST_ACTIVE", now) # Reset timer for next attempt
                return self._get_event()
            elif now - self.state_start_t > 3.0: # 3 second timeout for attempt
                self.attempts += 1
                if self.attempts >= self.max_attempts:
                    self._transition_to("TEST_CONTROL", now)
                else:
                    self._transition_to("TEST_ACTIVE", now)
                return self._get_event()
                
        elif self.state == "TEST_CONTROL":
            if detected == current_gesture:
                self.false_positives += 1
                
            if now - self.state_start_t > self.control_duration_s:
                success_rate = self.successes / self.max_attempts if self.max_attempts > 0 else 0.0
                fp_rate = self.false_positives / self.control_duration_s # FP per second
                
                # Thresholds are tuned parameters pending large-scale data.
                # 0.66 success floor allows 2/3 successes to pass.
                # 0.15 fp ceiling allows 1 false positive per 10s control window.
                enabled = (success_rate >= 0.66) and (fp_rate <= 0.15)
                
                self.results.append({
                    "id": current_gesture,
                    "success": success_rate,
                    "attempts": self.attempts,
                    "false_positive": fp_rate,
                    "control_elapsed_s": self.control_duration_s,
                    "enabled": enabled,
                    "declined_by_user": False,
                    "params": self._detector_params(current_gesture)
                })
                
                self._next_gesture(now)
                return self._get_event()
                
        return None
        
    def assign_roles(self):
        """
        Assigns roles: ENGAGE, CANCEL, MENU.
        MENU is always corner dwell.
        Others are assigned from available gestures, preferring fastest.
        Falls back to reserved_zone_dwell.
        """
        roles = {
            "engage": "reserved_zone_dwell",
            "cancel": "reserved_zone_dwell",
            "menu": "corner_dwell"
        }
        
        enabled = [r for r in self.results if r["enabled"] and not r["declined_by_user"]]
        
        preference = [
            name for name in
            ["long_blink", "off_screen_glance", "gaze_stroke", "extended_closure", "smooth_pursuit"]
            if name in self.candidates
        ]
        
        available = []
        for pref in preference:
            if any(r["id"] == pref for r in enabled):
                available.append(pref)
                
        if len(available) > 0:
            roles["engage"] = available[0]
        if len(available) > 1:
            roles["cancel"] = available[1]
            
        return roles
