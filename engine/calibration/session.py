import math
import logging
from typing import Optional
from engine.sources.base import GazeSample
from engine.features.eye_features import extract_features

logger = logging.getLogger(__name__)

# Dispersion threshold: If the dispersion (max distance between any two points in the window) 
# exceeds this, the fixation is unstable.
# Value is 0.02 normalized feature units, typical for this feature space but untuned pending real data.
DISPERSION_THRESHOLD = 0.02
SETTLE_TIME_S = 0.3
COLLECT_TIME_S = 0.68
MAX_RETRIES = 3

# Below this count the dispersion of a window is not meaningful, so the window is unstable
# regardless of what it measures.
MIN_WINDOW_SAMPLES = 10


def compute_dispersion(features: list[tuple[float, float]]) -> float:
    """Diagonal of the bounding box enclosing a feature window."""
    if not features:
        return float('inf')
    min_x = min(f[0] for f in features)
    max_x = max(f[0] for f in features)
    min_y = min(f[1] for f in features)
    max_y = max(f[1] for f in features)
    return math.hypot(max_x - min_x, max_y - min_y)


def window_accepted(features: list[tuple[float, float]], threshold: Optional[float] = None) -> bool:
    """
    The acceptance rule for one collection window.

    Offline analysis re-applies this rule to recorded windows at candidate thresholds, so it
    is defined once here rather than restated by each caller. A second copy would let the
    tuned value and the live value diverge without any test noticing.
    """
    if threshold is None:
        threshold = DISPERSION_THRESHOLD
    return len(features) > MIN_WINDOW_SAMPLES and compute_dispersion(features) <= threshold


def sample_feature(sample: GazeSample) -> Optional[tuple[float, float]]:
    """The feature a sample contributes to a collection window, or None if it contributes none."""
    if sample.ok and sample.eyes:
        return extract_features(sample.eyes["left"], sample.eyes["right"])
    return None


class CalibrationSession:
    def __init__(self, screen_w: int, screen_h: int,
                 dispersion_threshold: Optional[float] = None,
                 ear_threshold: Optional[float] = None):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.ear_threshold = 0.2 if ear_threshold is None else ear_threshold

        # Resolved to a concrete number rather than kept as None, because the value is recorded
        # with every sample and a recording that stated "the default" would not say what the
        # default was when it was made.
        self.dispersion_threshold = (DISPERSION_THRESHOLD if dispersion_threshold is None
                                     else dispersion_threshold)
        
        # 9 point grid
        margin_x = screen_w * 0.1
        margin_y = screen_h * 0.1
        self.fit_targets = [
            (margin_x, margin_y), (screen_w/2, margin_y), (screen_w - margin_x, margin_y),
            (margin_x, screen_h/2), (screen_w/2, screen_h/2), (screen_w - margin_x, screen_h/2),
            (margin_x, screen_h - margin_y), (screen_w/2, screen_h - margin_y), (screen_w - margin_x, screen_h - margin_y)
        ]
        
        # 4 unseen points for validation
        self.val_targets = [
            (screen_w * 0.25, screen_h * 0.25),
            (screen_w * 0.75, screen_h * 0.25),
            (screen_w * 0.25, screen_h * 0.75),
            (screen_w * 0.75, screen_h * 0.75)
        ]
        
        self.all_targets = self.fit_targets + self.val_targets
        
        self.current_point_idx = 0
        self.retries = 0
        
        self.state = "IDLE" # IDLE, SETTLING, COLLECTING, DONE, FAILED
        self.state_start_t = 0.0
        
        self.window_samples = []
        
        self.collected_features = [] # List of (fx, fy)
        self.collected_targets = [] # List of (tx, ty)
        self.collected_ipds = []
        self.frame_width = None
        
        # Blink tracking
        self.natural_blinks = []
        self.current_blink_start = None

    def _transition_to(self, state: str, current_t: float):
        self.state = state
        self.state_start_t = current_t
        if state == "COLLECTING":
            self.window_samples = []

    def start(self):
        self.current_point_idx = 0
        self.retries = 0
        self.collected_features = []
        self.collected_targets = []
        self.collected_ipds = []
        self._transition_to("SETTLING", 0.0) # Will be updated on first sample
        return self._get_event()

    def _get_event(self):
        if self.state in ("SETTLING", "COLLECTING"):
            tx, ty = self.all_targets[self.current_point_idx]
            return {"type": "CALIBRATION_POINT", "x": tx, "y": ty, "is_validation": self.current_point_idx >= len(self.fit_targets)}
        elif self.state == "DONE":
            return {"type": "CALIBRATION_DONE"}
        elif self.state == "FAILED":
            return {"type": "CALIBRATION_FAILED"}
        return None
        
    def _compute_dispersion(self, features: list[tuple[float, float]]) -> float:
        return compute_dispersion(features)

    def process_sample(self, sample: GazeSample) -> Optional[dict]:
        now = sample.t
        if self.state_start_t == 0.0:
            self.state_start_t = now
            
        if sample.frame_width is not None and self.frame_width is None:
            self.frame_width = sample.frame_width
        
        # Process blink
        is_closed = False
        if sample.ok and sample.ear:
            if sample.ear["left"] < self.ear_threshold and sample.ear["right"] < self.ear_threshold:
                is_closed = True
                
        if is_closed:
            if self.current_blink_start is None:
                self.current_blink_start = now
        else:
            if self.current_blink_start is not None:
                duration_ms = (now - self.current_blink_start) * 1000
                # Bounds for natural blink: 50 ms (minimum physiologically plausible) to 1500 ms.
                # Thresholds are assumed constants pending large-scale user data collection.
                if duration_ms > 50 and duration_ms < 1500:
                    self.natural_blinks.append(duration_ms)
                self.current_blink_start = None

        if self.state == "SETTLING":
            if now - self.state_start_t >= SETTLE_TIME_S:
                self._transition_to("COLLECTING", now)
                
        elif self.state == "COLLECTING":
            if now - self.state_start_t >= COLLECT_TIME_S:
                features = [(f_x, f_y) for f_x, f_y, _ in self.window_samples]
                dispersion = self._compute_dispersion(features)
                
                if window_accepted(features, self.dispersion_threshold):
                    # Accept
                    avg_fx = sum(f[0] for f in features) / len(features)
                    avg_fy = sum(f[1] for f in features) / len(features)
                    valid_ipds = [s[2] for s in self.window_samples if s[2]]
                    avg_ipd = sum(valid_ipds) / len(valid_ipds) if valid_ipds else 0.0
                    
                    self.collected_features.append((avg_fx, avg_fy))
                    self.collected_targets.append(self.all_targets[self.current_point_idx])
                    self.collected_ipds.append(avg_ipd)
                    
                    self.current_point_idx += 1
                    self.retries = 0
                    
                    if self.current_point_idx >= len(self.all_targets):
                        self._transition_to("DONE", now)
                    else:
                        self._transition_to("SETTLING", now)
                else:
                    # Reject and re-present
                    self.retries += 1
                    logger.info(f"Point {self.current_point_idx} rejected. Dispersion: {dispersion:.4f}, Samples: {len(features)}. Retry {self.retries}")
                    if self.retries > MAX_RETRIES:
                        self._transition_to("FAILED", now)
                    else:
                        self._transition_to("SETTLING", now)
                        
                return self._get_event()
                
            feature = sample_feature(sample)
            if feature is not None:
                self.window_samples.append((feature[0], feature[1], sample.ipd_px))

        return None
        
    def get_fit_data(self):
        n_fit = len(self.fit_targets)
        return self.collected_features[:n_fit], self.collected_targets[:n_fit]
        
    def get_val_data(self):
        n_fit = len(self.fit_targets)
        return self.collected_features[n_fit:], self.collected_targets[n_fit:]
        
    def get_avg_ipd(self):
        valid_ipds = [ipd for ipd in self.collected_ipds if ipd > 0.0]
        if not valid_ipds:
            return 0.0
        return sum(valid_ipds) / len(valid_ipds)
        
    def get_frame_width(self):
        """Returns the capture width, or None if no sample carried one."""
        return self.frame_width
        
    def get_long_blink_threshold_ms(self):
        if not self.natural_blinks:
            return 450.0
        p99 = sorted(self.natural_blinks)[int(len(self.natural_blinks) * 0.99)]
        return max(400.0, p99)
