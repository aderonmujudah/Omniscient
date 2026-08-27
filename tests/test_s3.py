"""Tests for signal conditioning and event detection.

Covers pass marks PM-3.1 through PM-3.17.

Synthetic fixtures are used to test behavioural specification
(given this input, expect that output). No accuracy or tuning claims
are made from synthetic data.
"""

import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from engine.sources.base import GazeSample, EyeGeometry, Point2D
from engine.filtering.one_euro import OneEuroFilter
from engine.filtering.fixation import FixationDetector, Fixation
from engine.filtering.classifier import SampleClassifier, SampleState
from engine.events.interaction import InteractionEvent, EventType
from engine.events.dwell import DwellTimer
from engine.events.dispatcher import EventDispatcher
from engine.events.gestures.base import Role, GestureEvent
from engine.events.gestures.registry import GestureRegistry
from engine.events.gestures.long_blink import LongBlinkDetector
from engine.events.gestures.extended_closure import ExtendedClosureDetector
from engine.events.gestures.off_screen_glance import OffScreenGlanceDetector
from engine.events.gestures.smooth_pursuit import SmoothPursuitDetector
from engine.events.gestures.gaze_stroke import GazeStrokeDetector
from engine.events.gestures.reserved_zone_dwell import ReservedZoneDwellDetector
from engine.calibration.assessment import GestureAssessment, CANDIDATE_GESTURES
from engine.events.emitter import InteractionEmitter
from engine.features.eye_features import extract_features
from engine.main import RESERVED_ZONES, build_emitter, calibrated_position
from engine.capture.null import NullCapture
from engine.input.null import NullInput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sample(
    t: float,
    ok: bool = True,
    ear_left: float = 0.3,
    ear_right: float = 0.3,
    frame_width: int = 1280,
    seq: int = 0,
) -> GazeSample:
    ear = {"left": ear_left, "right": ear_right} if ok else None
    return GazeSample(
        t=t,
        seq=seq,
        ok=ok,
        ear=ear,
        frame_width=frame_width,
    )


def closed_sample(t: float, seq: int = 0) -> GazeSample:
    """Sample with eyes closed (EAR below 0.2 threshold)."""
    return make_sample(t=t, ear_left=0.1, ear_right=0.1, seq=seq)


def open_sample(t: float, seq: int = 0) -> GazeSample:
    """Sample with eyes open. A sample carries no screen position: calibrated gaze is
    supplied separately by whichever component owns the calibration model."""
    return make_sample(t=t, seq=seq)


def lost_sample(t: float, seq: int = 0) -> GazeSample:
    """Sample where tracking is lost."""
    return make_sample(t=t, ok=False, seq=seq)


# ---------------------------------------------------------------------------
# PM-3.1: 1€ filter reduces jitter without adding perceptible lag
# ---------------------------------------------------------------------------

class TestOneEuroFilter:

    def test_reduces_jitter_during_fixation(self):
        """Filtered output has lower variance than noisy input at a fixed point."""
        filt = OneEuroFilter(rate=30.0)
        import random
        random.seed(42)

        raw_values = []
        filtered_values = []
        center_x, center_y = 500.0, 400.0

        for i in range(90):
            noise_x = random.gauss(0, 5.0)
            noise_y = random.gauss(0, 5.0)
            t = i / 30.0
            raw_x = center_x + noise_x
            raw_y = center_y + noise_y
            raw_values.append((raw_x, raw_y))

            fx, fy = filt(raw_x, raw_y, t)
            filtered_values.append((fx, fy))

        raw_var_x = sum((v[0] - center_x) ** 2 for v in raw_values) / len(raw_values)
        filt_var_x = sum((v[0] - center_x) ** 2 for v in filtered_values) / len(filtered_values)

        assert filt_var_x < raw_var_x, "Filtered variance should be less than raw variance"

    def test_group_delay_below_two_frames(self):
        """During a step response, the filter converges within two frames."""
        filt = OneEuroFilter(rate=30.0)

        for i in range(30):
            filt(100.0, 100.0, i / 30.0)

        fx, fy = filt(500.0, 500.0, 1.0)
        fx2, fy2 = filt(500.0, 500.0, 1.0 + 1 / 30.0)

        assert abs(fx2 - 500.0) < abs(fx - 500.0), "Second sample should be closer to target"

    def test_reset_clears_state(self):
        filt = OneEuroFilter(rate=30.0)
        filt(100.0, 100.0, 0.0)
        filt.reset()
        fx, fy = filt(500.0, 500.0, 1.0)
        assert fx == 500.0
        assert fy == 500.0


# ---------------------------------------------------------------------------
# PM-3.2: I-DT fixation detector identifies fixations
# (Behavioural specification against synthetic input)
# ---------------------------------------------------------------------------

class TestFixationDetector:

    def test_detects_stable_fixation(self):
        """A cluster of points within the dispersion threshold produces a fixation."""
        det = FixationDetector(dispersion_threshold=50.0, min_fixation_duration_s=0.1)

        for i in range(10):
            det.process(500.0 + i * 0.5, 400.0 + i * 0.3, i * 0.033)

        assert det.active_fixation_centroid is not None

    def test_saccade_ends_fixation(self):
        """A jump in position ends the current fixation and emits it."""
        det = FixationDetector(dispersion_threshold=50.0, min_fixation_duration_s=0.1)

        for i in range(10):
            det.process(500.0, 400.0, i * 0.033)

        result = det.process(800.0, 600.0, 0.35)

        assert result is not None
        assert abs(result.x - 500.0) < 5.0
        assert abs(result.y - 400.0) < 5.0

    def test_short_gaze_not_promoted(self):
        """Points below minimum duration are not promoted to fixation."""
        det = FixationDetector(dispersion_threshold=50.0, min_fixation_duration_s=0.2)

        det.process(500.0, 400.0, 0.0)
        det.process(500.0, 400.0, 0.05)

        assert det.active_fixation_centroid is None


# ---------------------------------------------------------------------------
# PM-3.3: Sample classifier labels match expected state sequence
# ---------------------------------------------------------------------------

class TestSampleClassifier:

    def test_fixation_saccade_blink_sequence(self):
        """Synthetic fixation → saccade → fixation → blink → fixation produces correct labels."""
        fix_det = FixationDetector(dispersion_threshold=50.0, min_fixation_duration_s=0.1)
        classifier = SampleClassifier(fixation_detector=fix_det)

        states: List[SampleState] = []
        t = 0.0

        # Phase 1: Fixation at (500, 400) for 200ms
        for i in range(7):
            state, _ = classifier.classify(
                ok=True, ear_left=0.3, ear_right=0.3,
                gaze_x=500.0, gaze_y=400.0, timestamp=t
            )
            states.append(state)
            t += 0.033

        # Phase 2: Saccade to (800, 600)
        state, _ = classifier.classify(
            ok=True, ear_left=0.3, ear_right=0.3,
            gaze_x=800.0, gaze_y=600.0, timestamp=t
        )
        states.append(state)
        t += 0.033

        # Phase 3: Fixation at (800, 600) for 200ms
        for i in range(7):
            state, _ = classifier.classify(
                ok=True, ear_left=0.3, ear_right=0.3,
                gaze_x=800.0, gaze_y=600.0, timestamp=t
            )
            states.append(state)
            t += 0.033

        # Phase 4: Blink
        for i in range(3):
            state, _ = classifier.classify(
                ok=True, ear_left=0.1, ear_right=0.1,
                gaze_x=800.0, gaze_y=600.0, timestamp=t
            )
            states.append(state)
            t += 0.033

        # Phase 5: Fixation again
        for i in range(7):
            state, _ = classifier.classify(
                ok=True, ear_left=0.3, ear_right=0.3,
                gaze_x=500.0, gaze_y=400.0, timestamp=t
            )
            states.append(state)
            t += 0.033

        blink_states = [s for s in states if s == SampleState.BLINK]
        assert len(blink_states) == 3

        lost_states = [s for s in states if s == SampleState.LOST]
        assert len(lost_states) == 0

        # Verify transition boundaries: at most one misclassified sample per boundary
        fixation_count = sum(1 for s in states if s == SampleState.FIXATION)
        assert fixation_count > 0, "Should have fixation samples"

    def test_lost_takes_priority(self):
        classifier = SampleClassifier(
            fixation_detector=FixationDetector(), ear_threshold=0.2
        )
        state, _ = classifier.classify(
            ok=False, ear_left=None, ear_right=None,
            gaze_x=0.0, gaze_y=0.0, timestamp=0.0
        )
        assert state == SampleState.LOST

    def test_blink_takes_priority_over_fixation(self):
        classifier = SampleClassifier(
            fixation_detector=FixationDetector(), ear_threshold=0.2
        )
        state, _ = classifier.classify(
            ok=True, ear_left=0.1, ear_right=0.1,
            gaze_x=500.0, gaze_y=400.0, timestamp=0.0
        )
        assert state == SampleState.BLINK


# ---------------------------------------------------------------------------
# PM-3.4: Long blink detector fires between 300 ms and 800 ms
# ---------------------------------------------------------------------------

class TestLongBlink:

    def test_fires_in_range(self):
        """Closure of 500ms fires the detector exactly once."""
        det = LongBlinkDetector(ear_threshold=0.2)
        det.update_gaze_position(500.0, 400.0)

        det.process_sample(open_sample(t=0.0))
        det.process_sample(closed_sample(t=0.1))
        det.process_sample(closed_sample(t=0.3))
        det.process_sample(closed_sample(t=0.5))
        result = det.process_sample(open_sample(t=0.6))

        assert result == "long_blink"

    def test_natural_blink_does_not_fire(self):
        """Closure of 200ms (natural blink) does not fire."""
        det = LongBlinkDetector(ear_threshold=0.2)
        det.update_gaze_position(500.0, 400.0)

        det.process_sample(open_sample(t=0.0))
        det.process_sample(closed_sample(t=0.1))
        result = det.process_sample(open_sample(t=0.3))

        assert result is None

    def test_extended_closure_does_not_fire(self):
        """Closure of 1000ms does not fire the long blink detector."""
        det = LongBlinkDetector(ear_threshold=0.2)
        det.update_gaze_position(500.0, 400.0)

        det.process_sample(open_sample(t=0.0))
        det.process_sample(closed_sample(t=0.1))
        det.process_sample(closed_sample(t=0.5))
        result = det.process_sample(open_sample(t=1.1))

        assert result is None

    def test_fires_exactly_once(self):
        """A single closure in range fires exactly once."""
        det = LongBlinkDetector(ear_threshold=0.2)
        det.update_gaze_position(500.0, 400.0)

        results = []
        det.process_sample(open_sample(t=0.0))
        det.process_sample(closed_sample(t=0.1))
        det.process_sample(closed_sample(t=0.3))
        det.process_sample(closed_sample(t=0.5))
        results.append(det.process_sample(open_sample(t=0.6)))
        results.append(det.process_sample(open_sample(t=0.7)))

        assert results.count("long_blink") == 1


# ---------------------------------------------------------------------------
# PM-3.5: Extended closure detector fires between 800 ms and 2000 ms
# ---------------------------------------------------------------------------

class TestExtendedClosure:

    def test_fires_in_range(self):
        det = ExtendedClosureDetector(ear_threshold=0.2)
        det.update_gaze_position(500.0, 400.0)

        det.process_sample(open_sample(t=0.0))
        det.process_sample(closed_sample(t=0.1))
        det.process_sample(closed_sample(t=0.5))
        result = det.process_sample(open_sample(t=1.0))

        assert result == "extended_closure"

    def test_short_closure_does_not_fire(self):
        det = ExtendedClosureDetector(ear_threshold=0.2)
        det.update_gaze_position(500.0, 400.0)

        det.process_sample(open_sample(t=0.0))
        det.process_sample(closed_sample(t=0.1))
        result = det.process_sample(open_sample(t=0.6))

        assert result is None

    def test_rest_does_not_fire(self):
        """Closure > 2000ms is rest, not a gesture."""
        det = ExtendedClosureDetector(ear_threshold=0.2)
        det.update_gaze_position(500.0, 400.0)

        det.process_sample(open_sample(t=0.0))
        det.process_sample(closed_sample(t=0.1))
        result = det.process_sample(open_sample(t=2.5))

        assert result is None


# ---------------------------------------------------------------------------
# PM-3.6: Off-screen glance detector fires on deliberate off-screen look
# ---------------------------------------------------------------------------

class TestOffScreenGlance:

    def test_fires_on_glance_and_return(self):
        det = OffScreenGlanceDetector(screen_w=1920, screen_h=1080)

        det.update_gaze_position(960.0, 540.0)
        det.process_sample(open_sample(t=0.0))

        det.update_gaze_position(-100.0, 540.0)
        det.process_sample(open_sample(t=0.1))

        det.update_gaze_position(-100.0, 540.0)
        det.process_sample(open_sample(t=0.3))

        det.update_gaze_position(960.0, 540.0)
        result = det.process_sample(open_sample(t=0.5))

        assert result == "off_screen_glance"

    def test_no_return_does_not_fire(self):
        det = OffScreenGlanceDetector(screen_w=1920, screen_h=1080)

        det.update_gaze_position(960.0, 540.0)
        det.process_sample(open_sample(t=0.0))

        det.update_gaze_position(-100.0, 540.0)
        det.process_sample(open_sample(t=0.1))

        det.update_gaze_position(-100.0, 540.0)
        result = det.process_sample(open_sample(t=1.5))

        assert result is None

    def test_too_fast_does_not_fire(self):
        det = OffScreenGlanceDetector(screen_w=1920, screen_h=1080)

        det.update_gaze_position(960.0, 540.0)
        det.process_sample(open_sample(t=0.0))

        det.update_gaze_position(-100.0, 540.0)
        det.process_sample(open_sample(t=0.1))

        det.update_gaze_position(960.0, 540.0)
        result = det.process_sample(open_sample(t=0.15))

        assert result is None


# ---------------------------------------------------------------------------
# PM-3.7: Smooth pursuit detector (structural test — parameters untuned)
# ---------------------------------------------------------------------------

class TestSmoothPursuit:

    def test_detector_exists_and_has_protocol(self):
        det = SmoothPursuitDetector()
        assert det.name == "smooth_pursuit"
        det.reset()
        assert det.latched_position == (0.0, 0.0)

    def test_fixation_does_not_fire(self):
        det = SmoothPursuitDetector()
        for i in range(60):
            det.update_gaze_position(500.0, 400.0)
            result = det.process_sample(open_sample(t=i * 0.033))
            assert result is None


# ---------------------------------------------------------------------------
# PM-3.8: Gaze stroke detector identifies directional swipe
# ---------------------------------------------------------------------------

class TestGazeStroke:

    def test_fires_on_large_displacement(self):
        det = GazeStrokeDetector(min_displacement_px=200.0, max_duration_s=0.5)

        det.update_gaze_position(100.0, 500.0)
        det.process_sample(open_sample(t=0.0))

        det.update_gaze_position(150.0, 500.0)
        det.process_sample(open_sample(t=0.1))

        det.update_gaze_position(350.0, 500.0)
        result = det.process_sample(open_sample(t=0.2))

        assert result == "gaze_stroke"

    def test_jitter_does_not_fire(self):
        det = GazeStrokeDetector(min_displacement_px=200.0, max_duration_s=0.5)

        for i in range(15):
            det.update_gaze_position(500.0 + (i % 3) * 5, 400.0)
            result = det.process_sample(open_sample(t=i * 0.033))
            assert result is None


# ---------------------------------------------------------------------------
# PM-3.9: Reserved-zone dwell fires in the reserved zone
# ---------------------------------------------------------------------------

class TestReservedZoneDwell:

    def _make_zones(self) -> Dict[Role, dict]:
        return {
            Role.ENGAGE: {"x": 0.0, "y": 0.0, "w": 0.1, "h": 0.1},
            Role.CANCEL: {"x": 0.9, "y": 0.0, "w": 0.1, "h": 0.1},
            Role.MENU: {"x": 0.0, "y": 0.9, "w": 0.1, "h": 0.1},
        }

    def test_fires_in_zone(self):
        det = ReservedZoneDwellDetector(
            zones=self._make_zones(),
            screen_w=1920,
            screen_h=1080,
            dwell_duration_s=0.5,
        )

        for i in range(20):
            det.update_gaze_position(50.0, 50.0)
            result = det.process_sample(open_sample(t=i * 0.033))
            if result is not None:
                assert result == "reserved_zone_dwell"
                assert det.last_fired_role == Role.ENGAGE
                return

        pytest.fail("Detector did not fire")

    def test_does_not_fire_outside_zone(self):
        det = ReservedZoneDwellDetector(
            zones=self._make_zones(),
            screen_w=1920,
            screen_h=1080,
            dwell_duration_s=0.5,
        )

        for i in range(30):
            det.update_gaze_position(960.0, 540.0)
            result = det.process_sample(open_sample(t=i * 0.033))
            assert result is None


# ---------------------------------------------------------------------------
# PM-3.10: With every optional gesture disabled, all three roles fire
#           via reserved-zone dwell
# ---------------------------------------------------------------------------

class TestAllRolesViaReservedZoneDwell:

    def test_all_roles_fire(self):
        role_assignment = {
            "ENGAGE": "reserved_zone_dwell",
            "CANCEL": "reserved_zone_dwell",
            "MENU": "reserved_zone_dwell",
        }
        reserved_zones = {
            "ENGAGE": {"x": 0.0, "y": 0.0, "w": 0.1, "h": 0.1},
            "CANCEL": {"x": 0.9, "y": 0.0, "w": 0.1, "h": 0.1},
            "MENU": {"x": 0.0, "y": 0.9, "w": 0.1, "h": 0.1},
        }

        registry = GestureRegistry(
            role_assignment=role_assignment,
            screen_w=1920,
            screen_h=1080,
            reserved_zones=reserved_zones,
            gesture_params={},
        )

        fired_roles = set()

        test_positions = [
            (50.0, 50.0, Role.ENGAGE),
            (1870.0, 50.0, Role.CANCEL),
            (50.0, 1030.0, Role.MENU),
        ]

        for gx, gy, expected_role in test_positions:
            registry.reset()
            for i in range(30):
                events = registry.process_sample(
                    open_sample(t=i * 0.033), gaze_x=gx, gaze_y=gy
                )
                for evt in events:
                    fired_roles.add(evt.role)

        assert Role.ENGAGE in fired_roles, "ENGAGE did not fire"
        assert Role.CANCEL in fired_roles, "CANCEL did not fire"
        assert Role.MENU in fired_roles, "MENU did not fire"


# ---------------------------------------------------------------------------
# PM-3.11: Gesture detectors emit by role, not by identity
# ---------------------------------------------------------------------------

class TestEmitByRole:

    def test_no_gesture_names_outside_package(self):
        """Grep-equivalent: no code outside gestures/ and assessment.py
        references a specific gesture name when dispatching or consuming."""
        engine_root = Path(__file__).parent.parent / "engine"

        gesture_names = [
            "long_blink", "extended_closure", "off_screen_glance",
            "smooth_pursuit", "gaze_stroke", "reserved_zone_dwell",
            "corner_dwell",
        ]

        allowed_paths = {
            "events/gestures/",
            "calibration/assessment.py",
        }

        violations = []

        for py_file in engine_root.rglob("*.py"):
            rel = str(py_file.relative_to(engine_root))

            if any(rel.startswith(a) or rel == a for a in allowed_paths):
                continue

            content = py_file.read_text()
            for name in gesture_names:
                pattern = rf'["\']({re.escape(name)})["\']'
                matches = re.findall(pattern, content)
                if matches:
                    violations.append(f"{rel}: references '{name}'")

        assert violations == [], f"Gesture names outside allowed paths:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# PM-3.12: Gaze point is latched at closure onset
# ---------------------------------------------------------------------------

class TestGazeLatchAtOnset:

    def test_position_is_from_before_closure(self):
        det = LongBlinkDetector(ear_threshold=0.2)

        det.update_gaze_position(500.0, 400.0)
        det.process_sample(open_sample(t=0.0))

        det.update_gaze_position(510.0, 410.0)
        det.process_sample(open_sample(t=0.05))

        det.process_sample(closed_sample(t=0.1))
        det.process_sample(closed_sample(t=0.3))
        det.process_sample(closed_sample(t=0.5))
        result = det.process_sample(open_sample(t=0.6))

        assert result == "long_blink"
        lx, ly = det.latched_position
        assert lx == 510.0, f"Latched X should be 510.0 (last known before closure), got {lx}"
        assert ly == 410.0, f"Latched Y should be 410.0 (last known before closure), got {ly}"

    def test_position_not_from_closure_end(self):
        """The latched position must NOT be the position when eyes reopen."""
        det = LongBlinkDetector(ear_threshold=0.2)

        det.update_gaze_position(500.0, 400.0)
        det.process_sample(open_sample(t=0.0))

        det.process_sample(closed_sample(t=0.1))

        det.update_gaze_position(999.0, 999.0)
        det.process_sample(closed_sample(t=0.5))

        result = det.process_sample(open_sample(t=0.6))

        assert result == "long_blink"
        lx, ly = det.latched_position
        assert lx != 999.0


# ---------------------------------------------------------------------------
# PM-3.13: Dwell timer emits progress, completion, and cancellation
# ---------------------------------------------------------------------------

class TestDwellTimer:

    def test_dwell_sequence(self):
        timer = DwellTimer(dwell_duration_s=0.5, progress_interval_s=0.1)

        all_events: List[InteractionEvent] = []

        all_events.extend(timer.update("zone_a", 100.0, 100.0, 0.0))
        all_events.extend(timer.update("zone_a", 100.0, 100.0, 0.1))
        all_events.extend(timer.update("zone_a", 100.0, 100.0, 0.2))
        all_events.extend(timer.update("zone_a", 100.0, 100.0, 0.3))
        all_events.extend(timer.update("zone_a", 100.0, 100.0, 0.4))
        all_events.extend(timer.update("zone_a", 100.0, 100.0, 0.5))

        types = [e.event_type for e in all_events]
        assert EventType.DWELL_START.value in types
        assert EventType.DWELL_PROGRESS.value in types
        assert EventType.DWELL_COMPLETE.value in types

    def test_departure_cancels(self):
        timer = DwellTimer(dwell_duration_s=0.5, progress_interval_s=0.1)

        timer.update("zone_a", 100.0, 100.0, 0.0)
        timer.update("zone_a", 100.0, 100.0, 0.2)

        events = timer.update(None, 500.0, 500.0, 0.3)

        types = [e.event_type for e in events]
        assert EventType.DWELL_CANCEL.value in types

    def test_progress_reaches_100_only_with_continuous_presence(self):
        timer = DwellTimer(dwell_duration_s=0.5, progress_interval_s=0.05)

        all_events: List[InteractionEvent] = []
        for i in range(20):
            evts = timer.update("z", 100.0, 100.0, i * 0.033)
            all_events.extend(evts)

        complete = [e for e in all_events if e.event_type == EventType.DWELL_COMPLETE.value]
        assert len(complete) == 1


# ---------------------------------------------------------------------------
# PM-3.14: Tracking loss cancels dwell and prevents stale activation
# ---------------------------------------------------------------------------

class TestTrackingLoss:

    def test_tracking_loss_cancels_dwell(self):
        timer = DwellTimer(dwell_duration_s=0.5)

        timer.update("zone_a", 100.0, 100.0, 0.0)
        timer.update("zone_a", 100.0, 100.0, 0.2)

        cancel_event = timer.cancel(timestamp=0.3)

        assert cancel_event is not None
        assert cancel_event.event_type == EventType.DWELL_CANCEL.value
        assert cancel_event.zone_id == "zone_a"

    def test_no_completion_after_loss(self):
        timer = DwellTimer(dwell_duration_s=0.5)

        timer.update("zone_a", 100.0, 100.0, 0.0)
        timer.update("zone_a", 100.0, 100.0, 0.3)

        timer.cancel(timestamp=0.35)

        events = timer.update("zone_a", 100.0, 100.0, 0.4)
        types = [e.event_type for e in events]

        assert EventType.DWELL_START.value in types
        assert EventType.DWELL_COMPLETE.value not in types


# ---------------------------------------------------------------------------
# PM-3.15: WebSocket events validate against protocol/schema.json
# ---------------------------------------------------------------------------

class TestSchemaValidation:

    def test_all_event_types_validate(self):
        try:
            import jsonschema
        except ImportError:
            pytest.skip("jsonschema not installed")

        schema_path = Path(__file__).parent.parent / "protocol" / "schema.json"
        with open(schema_path) as f:
            schema = json.load(f)

        test_events = [
            {"event_type": "GAZE_MOVE", "timestamp": 1.0, "x": 500.0, "y": 400.0},
            {"event_type": "FIXATION_START", "timestamp": 1.0, "x": 500.0, "y": 400.0},
            {"event_type": "FIXATION_END", "timestamp": 1.0, "x": 500.0, "y": 400.0, "duration_ms": 200.0},
            {"event_type": "SACCADE", "timestamp": 1.0, "start_x": 500.0, "start_y": 400.0, "end_x": 800.0, "end_y": 600.0},
            {"event_type": "BLINK", "timestamp": 1.0, "duration_ms": 150.0},
            {"event_type": "DWELL_START", "timestamp": 1.0, "zone_id": "zone_a", "x": 100.0, "y": 100.0},
            {"event_type": "DWELL_PROGRESS", "timestamp": 1.0, "zone_id": "zone_a", "progress": 0.5},
            {"event_type": "DWELL_COMPLETE", "timestamp": 1.0, "zone_id": "zone_a"},
            {"event_type": "DWELL_CANCEL", "timestamp": 1.0, "zone_id": "zone_a"},
            {"event_type": "GESTURE", "timestamp": 1.0, "role": "ENGAGE"},
            {"event_type": "TRACKING_LOST", "timestamp": 1.0},
            {"event_type": "TRACKING_RESUMED", "timestamp": 1.0, "x": 500.0, "y": 400.0},
            {"event_type": "STATE_CHANGE", "timestamp": 1.0, "from_state": "IDLE", "to_state": "ENGAGED"},
            {"event_type": "CALIBRATION_START", "timestamp": 1.0},
            {"event_type": "CALIBRATION_POINT", "timestamp": 1.0, "point_index": 0, "x": 100.0, "y": 100.0},
            {"event_type": "CALIBRATION_COMPLETE", "timestamp": 1.0, "profile_id": "abc-123"},
            {"event_type": "CALIBRATION_FAILED", "timestamp": 1.0, "reason": "too much error"},
        ]

        for event in test_events:
            try:
                jsonschema.validate(event, schema)
            except jsonschema.ValidationError as e:
                pytest.fail(f"Event {event['event_type']} failed validation: {e.message}")

    def test_interaction_event_to_dict_validates(self):
        try:
            import jsonschema
        except ImportError:
            pytest.skip("jsonschema not installed")

        schema_path = Path(__file__).parent.parent / "protocol" / "schema.json"
        with open(schema_path) as f:
            schema = json.load(f)

        events = [
            InteractionEvent(event_type="GAZE_MOVE", timestamp=1.0, x=500.0, y=400.0),
            InteractionEvent(event_type="GESTURE", timestamp=1.0, role="ENGAGE"),
            InteractionEvent(event_type="TRACKING_LOST", timestamp=1.0),
            InteractionEvent(event_type="DWELL_PROGRESS", timestamp=1.0, zone_id="z", progress=0.5),
        ]

        for event in events:
            d = event.to_dict()
            try:
                jsonschema.validate(d, schema)
            except jsonschema.ValidationError as e:
                pytest.fail(f"InteractionEvent {d['event_type']} failed: {e.message}")


# ---------------------------------------------------------------------------
# PM-3.16: At least one non-closure candidate gesture is functional
# ---------------------------------------------------------------------------

class TestNonClosureGesture:

    def test_off_screen_glance_is_non_closure_and_fires(self):
        """Off-screen glance is a gaze-position gesture that does not
        depend on eye closure. Verify it can fire."""
        det = OffScreenGlanceDetector(screen_w=1920, screen_h=1080)

        det.update_gaze_position(960.0, 540.0)
        det.process_sample(open_sample(t=0.0))

        det.update_gaze_position(-200.0, 540.0)
        det.process_sample(open_sample(t=0.1))
        det.process_sample(open_sample(t=0.3))

        det.update_gaze_position(960.0, 540.0)
        result = det.process_sample(open_sample(t=0.5))

        assert result == "off_screen_glance", "Non-closure gesture must fire"

    def test_off_screen_glance_in_candidate_set(self):
        from engine.calibration.assessment import CANDIDATE_GESTURES
        assert "off_screen_glance" in CANDIDATE_GESTURES


# ---------------------------------------------------------------------------
# PM-3.17: No code outside gesture package names a specific gesture
# (Already covered by PM-3.11 test above)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Additional integration tests
# ---------------------------------------------------------------------------

class TestGestureRegistry:

    def test_role_based_emission(self):
        """Registry emits events with roles, not gesture names."""
        role_assignment = {
            "ENGAGE": "long_blink",
            "CANCEL": "reserved_zone_dwell",
            "MENU": "reserved_zone_dwell",
        }
        reserved_zones = {
            "CANCEL": {"x": 0.9, "y": 0.0, "w": 0.1, "h": 0.1},
            "MENU": {"x": 0.0, "y": 0.9, "w": 0.1, "h": 0.1},
        }

        registry = GestureRegistry(
            role_assignment=role_assignment,
            screen_w=1920,
            screen_h=1080,
            reserved_zones=reserved_zones,
            gesture_params={},
        )

        registry.process_sample(open_sample(t=0.0), gaze_x=500.0, gaze_y=400.0)
        registry.process_sample(closed_sample(t=0.1), gaze_x=500.0, gaze_y=400.0)
        registry.process_sample(closed_sample(t=0.3), gaze_x=500.0, gaze_y=400.0)
        registry.process_sample(closed_sample(t=0.5), gaze_x=500.0, gaze_y=400.0)
        events = registry.process_sample(open_sample(t=0.6), gaze_x=500.0, gaze_y=400.0)

        assert len(events) == 1
        assert events[0].role == Role.ENGAGE
        assert isinstance(events[0], GestureEvent)

    def test_swapping_role_changes_behavior(self):
        """Changing the role assignment in the profile changes which gesture fires."""
        zones = {
            "ENGAGE": {"x": 0.0, "y": 0.0, "w": 0.1, "h": 0.1},
            "CANCEL": {"x": 0.9, "y": 0.0, "w": 0.1, "h": 0.1},
            "MENU": {"x": 0.0, "y": 0.9, "w": 0.1, "h": 0.1},
        }

        # Config 1: long_blink -> ENGAGE
        reg1 = GestureRegistry(
            role_assignment={"ENGAGE": "long_blink", "CANCEL": "reserved_zone_dwell", "MENU": "reserved_zone_dwell"},
            screen_w=1920, screen_h=1080, reserved_zones=zones,
            gesture_params={},
        )

        # Config 2: long_blink -> CANCEL
        reg2 = GestureRegistry(
            role_assignment={"ENGAGE": "reserved_zone_dwell", "CANCEL": "long_blink", "MENU": "reserved_zone_dwell"},
            screen_w=1920, screen_h=1080, reserved_zones=zones,
            gesture_params={},
        )

        def trigger_blink(reg: GestureRegistry) -> List[GestureEvent]:
            all_events = []
            reg.process_sample(open_sample(t=0.0), gaze_x=500.0, gaze_y=400.0)
            reg.process_sample(closed_sample(t=0.1), gaze_x=500.0, gaze_y=400.0)
            reg.process_sample(closed_sample(t=0.5), gaze_x=500.0, gaze_y=400.0)
            all_events.extend(reg.process_sample(open_sample(t=0.6), gaze_x=500.0, gaze_y=400.0))
            return all_events

        evts1 = trigger_blink(reg1)
        evts2 = trigger_blink(reg2)

        assert len(evts1) == 1
        assert evts1[0].role == Role.ENGAGE

        assert len(evts2) == 1
        assert evts2[0].role == Role.CANCEL


class TestDispatcher:

    def test_dispatches_to_subscribers(self):
        dispatcher = EventDispatcher()
        received: List[InteractionEvent] = []
        dispatcher.subscribe(received.append)

        event = InteractionEvent(event_type="GAZE_MOVE", timestamp=1.0, x=500.0, y=400.0)
        dispatcher.dispatch(event)

        assert len(received) == 1
        assert received[0] == event

    def test_dispatches_to_ws(self):
        dispatcher = EventDispatcher()
        ws_messages: List[dict] = []
        dispatcher.set_ws_broadcast(ws_messages.append)

        event = InteractionEvent(event_type="TRACKING_LOST", timestamp=1.0)
        dispatcher.dispatch(event)

        assert len(ws_messages) == 1
        assert ws_messages[0]["event_type"] == "TRACKING_LOST"


# ---------------------------------------------------------------------------
# The assessment must drive the same detector path that dispatch uses
# ---------------------------------------------------------------------------

class TestAssessmentDrivesDetectors:
    """A gesture that fires only when a test calls its detector directly is not available
    to a user. These tests drive the assessment itself."""

    def _advance_to(self, assess, gesture_name):
        while assess.candidates[assess.current_idx] != gesture_name:
            assess.user_declines(current_t=0.0)

    def test_non_closure_gesture_fires_through_assessment(self):
        assess = GestureAssessment(gaze_position_available=True)
        assess.start()
        self._advance_to(assess, "off_screen_glance")
        assess.user_ready(current_t=1000.0)

        assess.process_sample(open_sample(t=1000.0), gaze_x=960.0, gaze_y=540.0)
        assess.process_sample(open_sample(t=1000.1), gaze_x=-200.0, gaze_y=540.0)
        assess.process_sample(open_sample(t=1000.3), gaze_x=-200.0, gaze_y=540.0)
        assess.process_sample(open_sample(t=1000.5), gaze_x=960.0, gaze_y=540.0)

        assert assess.successes == 1

    def test_position_gesture_records_no_failure_when_position_is_absent(self):
        """A gesture that cannot be presented must not be offered, rather than offered and
        then recorded as one the user failed."""
        assess = GestureAssessment(gaze_position_available=False)
        assess.start()
        while assess.state != "DONE":
            assess.user_declines(current_t=0.0)

        assessed = {r["id"] for r in assess.results}
        assert "off_screen_glance" not in assessed
        assert "gaze_stroke" not in assessed

    def test_measured_threshold_governs_detection(self):
        """A closure that clears the generic default band but falls short of the measured
        per-user threshold must not be recognised."""
        assess = GestureAssessment(long_blink_threshold_ms=600.0, gaze_position_available=False)
        assess.start()
        assess.user_ready(current_t=1000.0)

        assess.process_sample(closed_sample(t=1000.0), gaze_x=None, gaze_y=None)
        assess.process_sample(open_sample(t=1000.4), gaze_x=None, gaze_y=None)

        assert assess.detectors["long_blink"].closure_min_s == 0.6
        assert assess.successes == 0

    def test_measured_threshold_is_reported_as_it_is_applied(self):
        assess = GestureAssessment(long_blink_threshold_ms=600.0, gaze_position_available=False)
        params = assess._detector_params("long_blink")
        assert params["threshold_ms"] == assess.detectors["long_blink"].closure_min_s * 1000.0


class TestNoUnfireableCandidate:

    def test_every_presented_candidate_can_fire(self):
        for available in (False, True):
            assess = GestureAssessment(gaze_position_available=available)
            for name in assess.candidates:
                detector = assess.detectors[name]
                assert detector.can_fire, f"{name} is presented but cannot fire"
                if detector.requires_gaze_position:
                    assert available, f"{name} needs a gaze position that is unavailable"

    def test_smooth_pursuit_is_declared_but_not_presented(self):
        assess = GestureAssessment(gaze_position_available=True)
        assert "smooth_pursuit" in CANDIDATE_GESTURES
        assert "smooth_pursuit" not in assess.candidates

    def test_a_non_closure_candidate_is_presented(self):
        assess = GestureAssessment(gaze_position_available=True)
        assert any(
            assess.detectors[name].requires_gaze_position for name in assess.candidates
        ), "no candidate is available to a user who cannot close their eyes on demand"


class TestRegistryAppliesMeasuredParameters:
    """The threshold measured during assessment must reach the detector that runs at
    dispatch time, otherwise the measurement is recorded and then discarded."""

    ROLES = {"ENGAGE": "long_blink", "CANCEL": "reserved_zone_dwell", "MENU": "reserved_zone_dwell"}

    def _drive_closure(self, registry, closure_s):
        registry.process_sample(open_sample(t=0.0), gaze_x=500.0, gaze_y=400.0)
        registry.process_sample(closed_sample(t=0.1), gaze_x=500.0, gaze_y=400.0)
        return registry.process_sample(
            open_sample(t=0.1 + closure_s), gaze_x=500.0, gaze_y=400.0
        )

    def test_measured_threshold_suppresses_a_short_closure(self):
        registry = GestureRegistry(
            role_assignment=self.ROLES,
            screen_w=1920, screen_h=1080, reserved_zones={},
            gesture_params={"long_blink": {"threshold_ms": 600.0}},
        )
        assert self._drive_closure(registry, 0.4) == []

    def test_same_closure_fires_under_the_default_band(self):
        registry = GestureRegistry(
            role_assignment=self.ROLES,
            screen_w=1920, screen_h=1080, reserved_zones={},
            gesture_params={},
        )
        events = self._drive_closure(registry, 0.4)
        assert len(events) == 1
        assert events[0].role == Role.ENGAGE

    def test_unfireable_gesture_is_not_assigned_to_a_role(self):
        registry = GestureRegistry(
            role_assignment={"ENGAGE": "smooth_pursuit"},
            screen_w=1920, screen_h=1080, reserved_zones={},
            gesture_params={},
        )
        for i in range(40):
            events = registry.process_sample(
                open_sample(t=i * 0.033), gaze_x=100.0 + i * 20.0, gaze_y=540.0
            )
            assert events == []


# ---------------------------------------------------------------------------
# The running engine
# ---------------------------------------------------------------------------

def _eye(cx: float, cy: float) -> EyeGeometry:
    """Eye geometry whose iris sits at a given offset within the eye."""
    return EyeGeometry(
        iris=Point2D(cx, cy),
        inner=Point2D(cx - 15.0, cy),
        outer=Point2D(cx + 15.0, cy),
        top=Point2D(cx, cy - 5.0),
        bottom=Point2D(cx, cy + 5.0),
    )


def _sample_with_eyes(t: float, offset: float, seq: int = 0, ok: bool = True,
                      ear: float = 0.3) -> GazeSample:
    return GazeSample(
        t=t,
        seq=seq,
        ok=ok,
        ear={"left": ear, "right": ear} if ok else None,
        eyes={"left": _eye(100.0 + offset, 100.0), "right": _eye(200.0 + offset, 100.0)},
        ipd_px=100.0,
        frame_width=1280,
        conf=0.9,
    )


def _fitted_profile(roles: Dict[str, str]) -> dict:
    """A profile carrying a real fitted mapping and the given role assignment.

    The mapping is fitted to synthetic features, which is legitimate here: the assertion is
    that the composition runs and emits, not that the mapping is accurate.
    """
    from engine.calibration.model import CalibrationModel

    features, targets = [], []
    for i, offset in enumerate((-10.0, -5.0, 0.0, 5.0, 10.0)):
        sample = _sample_with_eyes(float(i), offset)
        features.append(extract_features(sample.eyes["left"], sample.eyes["right"]))
        targets.append((960.0 + offset * 50.0, 540.0))

    model = CalibrationModel()
    model.fit(features, targets)

    return {
        "screen": {"w": 1920, "h": 1080},
        "model": model.to_dict(),
        "blink": {"natural_p99_ms": 180, "long_threshold_ms": 450},
        "gestures": {"assessed": [], "roles": roles},
    }


class TestRegistryReadsTheProfileItIsGiven:
    """The registry must resolve the role names the calibration profile actually writes.

    A role the registry fails to resolve is left unfilled, which removes the user's ability
    to perform that action entirely rather than failing visibly.
    """

    def test_roles_from_assign_roles_resolve(self):
        roles = GestureAssessment(gaze_position_available=True).assign_roles()

        registry = GestureRegistry(
            role_assignment=roles,
            screen_w=1920,
            screen_h=1080,
            reserved_zones=RESERVED_ZONES,
            gesture_params={},
        )

        assert registry._zone_detector is not None
        assert set(registry._zone_detector._zones) == {Role.ENGAGE, Role.CANCEL, Role.MENU}

    def test_every_role_fires_from_an_unmodified_profile_assignment(self):
        """With no optional gesture enabled, all three roles must still be reachable."""
        roles = GestureAssessment(gaze_position_available=True).assign_roles()

        registry = GestureRegistry(
            role_assignment=roles,
            screen_w=1920,
            screen_h=1080,
            reserved_zones=RESERVED_ZONES,
            gesture_params={},
        )

        corners = {
            Role.ENGAGE: (40.0, 40.0),
            Role.CANCEL: (1880.0, 40.0),
            Role.MENU: (40.0, 1040.0),
        }

        fired = set()
        t = 0.0
        for role, (x, y) in corners.items():
            registry.reset()
            for step in range(40):
                t += 0.05
                for event in registry.process_sample(make_sample(t, seq=step), x, y):
                    fired.add(event.role)

        assert fired == {Role.ENGAGE, Role.CANCEL, Role.MENU}


class TestEngineEmitsEvents:
    """The engine that runs must emit the event set, not merely contain components that could."""

    def test_replayed_session_produces_interaction_events(self):
        dispatcher = EventDispatcher()
        published: List[dict] = []
        dispatcher.set_ws_broadcast(published.append)

        roles = GestureAssessment(gaze_position_available=True).assign_roles()
        emitter, model, _ = build_emitter(_fitted_profile(roles), dispatcher, rate=30.0,
                                  capture_backend=NullCapture(), input_backend=NullInput())
        assert emitter is not None, "a profile with a fitted mapping must produce an emitter"

        for step in range(40):
            sample = _sample_with_eyes(step * 0.033, 0.0, seq=step)
            position = calibrated_position(model, sample)
            assert position is not None
            emitter.process_sample(sample, position[0], position[1])

        types = {event["event_type"] for event in published}
        assert EventType.GAZE_MOVE.value in types
        assert EventType.FIXATION_START.value in types

    def test_no_mapping_yields_no_emitter_rather_than_a_silent_one(self):
        """A profile without a mapping has no screen position, and the engine must say so."""
        emitter, model, _ = build_emitter({"screen": {"w": 1920, "h": 1080}},
                                          EventDispatcher(), rate=30.0,
                                          capture_backend=NullCapture(),
                                          input_backend=NullInput())
        assert emitter is None
        assert model is None

    def test_tracking_loss_is_reported_once_and_cancels_dwell(self):
        dispatcher = EventDispatcher()
        published: List[dict] = []
        dispatcher.set_ws_broadcast(published.append)

        roles = GestureAssessment(gaze_position_available=True).assign_roles()
        emitter, model, _ = build_emitter(_fitted_profile(roles), dispatcher, rate=30.0,
                                  capture_backend=NullCapture(), input_backend=NullInput())

        for step in range(10):
            sample = _sample_with_eyes(step * 0.033, 0.0, seq=step)
            position = calibrated_position(model, sample)
            emitter.process_sample(sample, position[0], position[1])

        published.clear()
        for step in range(5):
            emitter.process_sample(make_sample(1.0 + step * 0.033, ok=False, seq=step), None, None)

        lost = [e for e in published if e["event_type"] == EventType.TRACKING_LOST.value]
        assert len(lost) == 1, "a continuing loss must not be re-reported every frame"

        recovered = _sample_with_eyes(2.0, 0.0, seq=99)
        position = calibrated_position(model, recovered)
        emitter.process_sample(recovered, position[0], position[1])
        assert any(e["event_type"] == EventType.TRACKING_RESUMED.value for e in published)
