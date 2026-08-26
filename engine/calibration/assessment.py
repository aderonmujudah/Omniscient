import logging
from engine.events.gestures.long_blink import LongBlinkDetector
from engine.events.gestures.extended_closure import ExtendedClosureDetector
from engine.events.gestures.off_screen_glance import OffScreenGlanceDetector

logger = logging.getLogger(__name__)

# Candidate gestures.
# smooth_pursuit, gaze_stroke, and off_screen_glance are excluded from this 
# candidate set until their detectors are implemented.
CANDIDATE_GESTURES = [
    "long_blink",
    "extended_closure"
]

class GestureAssessment:
    def __init__(self, long_blink_threshold_ms: float = 450.0):
        self.long_blink_threshold_ms = long_blink_threshold_ms
        self.detectors = {
            "long_blink": LongBlinkDetector(threshold_ms=self.long_blink_threshold_ms),
            "extended_closure": ExtendedClosureDetector(threshold_ms=800.0)
        }
        
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
        if self.current_idx < len(CANDIDATE_GESTURES):
            self._transition_to("EXPLAIN", 0.0)
        else:
            self._transition_to("DONE", 0.0)
        return self._get_event()
        
    def _transition_to(self, state: str, current_t: float):
        self.state = state
        self.state_start_t = current_t
        if state == "TEST_ACTIVE":
            self.detectors[CANDIDATE_GESTURES[self.current_idx]].reset()
        elif state == "TEST_CONTROL":
            self.detectors[CANDIDATE_GESTURES[self.current_idx]].reset()

    def _get_event(self):
        if self.state == "DONE":
            return {"type": "ASSESSMENT_DONE"}
        
        current_gesture = CANDIDATE_GESTURES[self.current_idx]
        return {
            "type": "ASSESSMENT_STATE",
            "gesture": current_gesture,
            "state": self.state,
            "attempt": self.attempts + 1 if self.state == "TEST_ACTIVE" else None
        }

    def user_declines(self, current_t: float = 0.0):
        """User explicitly declines the current candidate."""
        if self.state in ("EXPLAIN", "TEST_ACTIVE", "TEST_CONTROL"):
            current_gesture = CANDIDATE_GESTURES[self.current_idx]
            
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

    def _next_gesture(self, current_t: float):
        self.current_idx += 1
        self.attempts = 0
        self.successes = 0
        self.false_positives = 0
        if self.current_idx >= len(CANDIDATE_GESTURES):
            self._transition_to("DONE", current_t)
        else:
            self._transition_to("EXPLAIN", current_t)

    def user_ready(self, current_t: float = 0.0):
        if self.state == "EXPLAIN":
            self._transition_to("TEST_ACTIVE", current_t)
            return self._get_event()

    def process_sample(self, sample):
        now = sample.t
        if self.state_start_t == 0.0:
            self.state_start_t = now
            
        if self.state == "DONE":
            return None
            
        current_gesture = CANDIDATE_GESTURES[self.current_idx]
        detector = self.detectors[current_gesture]
        
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
                    "params": {"threshold_ms": self.long_blink_threshold_ms} if current_gesture == "long_blink" else {}
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
        
        preference = ["long_blink", "off_screen_glance", "gaze_stroke", "extended_closure", "smooth_pursuit"]
        
        available = []
        for pref in preference:
            if any(r["id"] == pref for r in enabled):
                available.append(pref)
                
        if len(available) > 0:
            roles["engage"] = available[0]
        if len(available) > 1:
            roles["cancel"] = available[1]
            
        return roles
