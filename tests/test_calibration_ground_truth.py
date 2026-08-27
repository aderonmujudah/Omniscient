"""
Ground truth for a calibration recording is what was on screen while it was captured.

These tests exist because that is not recoverable by re-running the state machine: the machine
pairs a collected window with whatever target its own index points at, and the index only lands
where it did originally while the acceptance constants are unchanged. Tuning one of them is the
entire purpose of analysing these recordings.
"""
import json
import time

import pytest

from engine.calibration import session as session_module
from engine.calibration.harness import run_labeled_harness
from engine.calibration.labels import (
    CollectionWindow,
    make_label_provider,
    read_labeled_session,
    select_accepted,
    split_fit_val,
)
from engine.calibration.presenter import PresenterLogic
from engine.calibration.session import CalibrationSession
from engine.calibration.sweep import sweep_dispersion_threshold
from engine.sources.base import LABEL_KEY, EyeGeometry, GazeSample, Point2D
from engine.sources.recorder import RecorderSource
from engine.sources.replay import ReplaySource

SCREEN_W, SCREEN_H = 1920, 1080

# The same panel as above as reported to a process that has not declared DPI awareness on a
# display running a 125 per cent scale factor.
SCALED_SCREEN_W, SCALED_SCREEN_H = 1536, 864
DIAG_MM = 597.0
FRAME_WIDTH = 1000
SAMPLE_RATE_HZ = 60.0
T0 = 1000.0
UNLABELED_FIXTURE = "/app/fixtures/s2_calibration.jsonl"

# Enough samples for thirteen targets plus the retries these tests provoke. Reaching it means the
# session never terminated, which the tests catch by asserting on the final state.
MAX_SAMPLES = 4000

STABLE_SPREAD = 0.0005
MARGINAL_SPREAD = 0.01
RECORDED_THRESHOLD = 0.02
TUNED_THRESHOLD = 0.005

# A gate wider than the default, and a spread that falls between the two, so a window the default
# would reject is accepted under it. Recording the value alone would not show it took effect.
LOOSE_THRESHOLD = 0.1
UNSTABLE_SPREAD = 0.05


def _feature_for(tx: float, ty: float) -> tuple[float, float]:
    return ((tx - SCREEN_W / 2) / SCREEN_W * 0.5, (ty - SCREEN_H / 2) / SCREEN_H * 0.5)


def _sample(t: float, seq: int, fx: float, fy: float) -> GazeSample:
    # Inter-corner distance is 100 px and extract_features scales displacement by it, so an
    # iris offset of fx * 100 px yields a feature of exactly fx.
    scale, cx, cy = 100.0, 500.0, 500.0
    geom = EyeGeometry(
        iris=Point2D(cx + fx * scale, cy + fy * scale),
        inner=Point2D(cx - 50.0, cy),
        outer=Point2D(cx + 50.0, cy),
        top=Point2D(cx, cy - 20.0),
        bottom=Point2D(cx, cy + 20.0),
    )
    return GazeSample(t=t, seq=seq, ok=True, eyes={"left": geom, "right": geom},
                      ear={"left": 0.3, "right": 0.3}, ipd_px=120.0,
                      frame_width=FRAME_WIDTH, conf=0.95)


class _ListSource:
    def __init__(self, samples):
        self.samples = samples

    def start(self):
        pass

    def stop(self):
        pass

    def iter_samples(self):
        return iter(self.samples)


class _SubjectSource:
    """
    A subject who looks wherever the driver is currently pointing.

    The stream is produced one sample at a time from the live session rather than laid out in
    fixed-length blocks per target. Blocks would require the test to reproduce the settle and
    collect arithmetic of the state machine exactly, and a boundary off by a single sample puts
    two targets inside one collection window. Reading the current target instead keeps the two
    aligned by construction, whatever the timing constants are set to.

    The gaze moves instantaneously, which SETTLE_TIME_S exists to absorb. Every target is held
    steady apart from a small tremor, except the optional marginal one, whose spread sits between
    the two thresholds these tests compare so that lowering the threshold turns its accepted
    window into a rejected one.
    """

    def __init__(self, session, marginal_idx=None, marginal_spread=MARGINAL_SPREAD):
        self.session = session
        self.marginal_idx = marginal_idx
        self.marginal_spread = marginal_spread

    def start(self):
        pass

    def stop(self):
        pass

    def iter_samples(self):
        for i in range(MAX_SAMPLES):
            idx = self.session.current_point_idx
            base_fx, base_fy = _feature_for(*self.session.all_targets[idx])
            spread = self.marginal_spread if idx == self.marginal_idx else STABLE_SPREAD
            yield _sample(T0 + i / SAMPLE_RATE_HZ, i + 1,
                          base_fx + (spread if i % 2 else 0.0), base_fy)
            if self.session.state in ("DONE", "FAILED"):
                return


def _record(path, marginal_idx=None, dispersion_threshold=None,
            marginal_spread=MARGINAL_SPREAD):
    """
    Writes a labeled recording the way the live driver does: the recorder wraps the source, the
    label provider reads the same session the samples are fed to, and the label is stored before
    the sample reaches the machine. Building the file by hand instead would prove only that the
    test agrees with itself.
    """
    session = CalibrationSession(SCREEN_W, SCREEN_H, dispersion_threshold)
    source = RecorderSource(inner=_SubjectSource(session, marginal_idx, marginal_spread),
                            filepath=str(path),
                            label_provider=make_label_provider(session))
    source.start()
    session.start()
    for sample in source.iter_samples():
        event = session.process_sample(sample)
        if event and event["type"] in ("CALIBRATION_DONE", "CALIBRATION_FAILED"):
            break
    source.stop()
    return session


def _read(path):
    source = ReplaySource(str(path), realtime=False)
    source.start()
    labeled = read_labeled_session(source.iter_labeled_samples())
    source.stop()
    return labeled


def _index_mode_pairs(path):
    """
    Re-runs the state machine over a recording and reports, per accepted window, the target the
    machine paired it with alongside the target the recording says was actually on screen while
    those samples were captured.
    """
    source = ReplaySource(str(path), realtime=False)
    session = CalibrationSession(SCREEN_W, SCREEN_H)
    source.start()
    session.start()

    pairs = []
    accepted = 0
    key = None
    window_labels = []

    for sample, label in source.iter_labeled_samples():
        if label is not None and label.get("state") == "COLLECTING":
            this_key = (label["point_idx"], label["attempt"])
            if this_key != key:
                key = this_key
                window_labels = []
            window_labels.append((label["x"], label["y"]))

        session.process_sample(sample)

        if len(session.collected_targets) > accepted:
            accepted = len(session.collected_targets)
            on_screen = max(set(window_labels), key=window_labels.count) if window_labels else None
            pairs.append((session.collected_targets[-1], on_screen))

    source.stop()
    return pairs


def test_labels_round_trip_and_match_what_was_presented(tmp_path):
    path = tmp_path / "labeled.jsonl"
    session = _record(path)

    assert session.state == "DONE"

    labeled = _read(path)
    assert labeled.labeled_count > 0
    assert labeled.frame_width == FRAME_WIDTH

    layout = CalibrationSession(SCREEN_W, SCREEN_H)
    assert len(labeled.windows) == len(layout.all_targets)

    for window, target in zip(labeled.windows, layout.all_targets):
        assert window.target == target
        assert window.is_validation == (window.point_idx >= len(layout.fit_targets))
        assert window.attempt == 0
        assert len(window.features) > session_module.MIN_WINDOW_SAMPLES


def test_pairings_agree_at_the_threshold_the_recording_was_made_at(tmp_path):
    path = tmp_path / "labeled.jsonl"
    _record(path, marginal_idx=4)

    assert session_module.DISPERSION_THRESHOLD == RECORDED_THRESHOLD

    pairs = _index_mode_pairs(path)
    assert len(pairs) == len(CalibrationSession(SCREEN_W, SCREEN_H).all_targets)
    for machine_target, on_screen_target in pairs:
        assert machine_target == on_screen_target


def test_pairings_diverge_when_the_threshold_is_tuned(tmp_path, monkeypatch):
    """
    The recording is unchanged and only the acceptance constant moves, which is what tuning does.

    The marginal window is now rejected, so the machine re-presents a target the recorded person
    has already looked away from and pairs the following window with a target that was never on
    screen while it was captured. Nothing in the state-machine path notices: it reports a fitted
    model and a plausible error over the mispairing.
    """
    path = tmp_path / "labeled.jsonl"
    _record(path, marginal_idx=4)

    monkeypatch.setattr(session_module, "DISPERSION_THRESHOLD", TUNED_THRESHOLD)

    pairs = _index_mode_pairs(path)
    mispaired = [(m, s) for m, s in pairs if m != s]
    assert mispaired, "expected the state machine to pair a window with an unpresented target"

    # The label-paired path drops the window the tuned threshold rejects and keeps every other
    # window matched to the target it was recorded against.
    labeled = _read(path)
    accepted, diverged = select_accepted(labeled.windows, TUNED_THRESHOLD)
    layout = CalibrationSession(SCREEN_W, SCREEN_H)
    assert len(accepted) == len(layout.all_targets) - 1
    assert diverged == 0
    for window in accepted:
        assert window.target == layout.all_targets[window.point_idx]

    result = run_labeled_harness(str(path), SCREEN_W, SCREEN_H, DIAG_MM,
                                 dispersion_threshold=TUNED_THRESHOLD)
    assert result is not None
    assert result["paired_by"] == "recorded_label"
    assert result["val_points"] == len(layout.val_targets)


def test_recording_without_labels_is_refused():
    """
    A recording made before the label track existed carries no ground truth, and must be refused
    rather than analysed as though the state machine could recover it.
    """
    assert run_labeled_harness(UNLABELED_FIXTURE, SCREEN_W, SCREEN_H, DIAG_MM) is None


def test_realtime_and_fast_modes_yield_identical_samples(tmp_path):
    """
    Timestamps come from the file in both modes, so skipping the sleeps must not change what a
    consumer sees. Only the wall clock differs.
    """
    path = tmp_path / "short.jsonl"
    samples = [_sample(1000.0 + i / SAMPLE_RATE_HZ, i + 1, 0.01 * i, 0.0) for i in range(5)]
    source = RecorderSource(inner=_ListSource(samples), filepath=str(path))
    source.start()
    list(source.iter_samples())
    source.stop()

    def read(realtime):
        replay = ReplaySource(str(path), realtime=realtime)
        replay.start()
        out = list(replay.iter_samples())
        replay.stop()
        return out

    started = time.monotonic()
    slow = read(True)
    slow_elapsed = time.monotonic() - started

    fast = read(False)

    assert [(s.t, s.seq, s.ok, s.ipd_px) for s in slow] == [(s.t, s.seq, s.ok, s.ipd_px) for s in fast]
    assert slow[0].eyes["left"].iris == fast[0].eyes["left"].iris
    assert slow_elapsed > 0.0


def test_sweep_of_fifty_values_completes_within_a_bounded_time(tmp_path):
    """
    Replayed in real time this sweep would take fifty times the duration of the recording. The
    bound is generous, and the point is the order of magnitude rather than the exact figure.
    """
    path = tmp_path / "labeled.jsonl"
    _record(path, marginal_idx=4)

    values = [0.002 + i * 0.001 for i in range(50)]
    started = time.monotonic()
    rows = sweep_dispersion_threshold(str(path), SCREEN_W, SCREEN_H, DIAG_MM, values)
    elapsed = time.monotonic() - started

    assert len(rows) == 50
    assert elapsed < 60.0


def test_sweep_reports_every_value_and_selects_none(tmp_path):
    """
    A sweep that returned a chosen value would let a figure fitted to one session from one person
    enter the codebase with the authority of a measurement.
    """
    path = tmp_path / "labeled.jsonl"
    _record(path, marginal_idx=4)

    values = [0.001, 0.005, 0.02, 0.05]
    rows = sweep_dispersion_threshold(str(path), SCREEN_W, SCREEN_H, DIAG_MM, values)

    assert [row["threshold"] for row in rows] == values
    assert all("usable" in row for row in rows)
    for row in rows:
        assert "selected" not in row
        assert "best" not in row
        assert "recommended" not in row


def test_a_dropped_window_cannot_move_a_validation_target_into_the_fit_set():
    """
    Slicing the accepted windows by position would pull a validation target into the fit set as
    soon as tuning dropped one fit window, and the accuracy figure would then be computed on a
    point the model had already seen.
    """
    windows = [
        CollectionWindow(point_idx=i, attempt=0, target=(float(i), 0.0), is_validation=i >= 3)
        for i in range(5)
    ]
    del windows[1]

    fit, val = split_fit_val(windows)

    assert [w.point_idx for w in fit] == [0, 2]
    assert [w.point_idx for w in val] == [3, 4]


def test_validation_targets_are_drawn_exactly_like_fit_targets():
    """
    The accuracy figure is computed from the validation points alone, so marking them on screen
    would change how they are looked at.
    """
    fit_logic = PresenterLogic()
    val_logic = PresenterLogic()

    fit_logic.apply({"type": "CALIBRATION_POINT", "x": 480.0, "y": 270.0, "is_validation": False})
    val_logic.apply({"type": "CALIBRATION_POINT", "x": 480.0, "y": 270.0, "is_validation": True})

    assert fit_logic.dot == val_logic.dot


def test_a_re_presented_target_does_not_redraw():
    """
    A rejected point is re-presented unchanged. Nothing on screen may mark when a collection
    window opens or restarts, or the person has something to time their gaze against.
    """
    logic = PresenterLogic()
    event = {"type": "CALIBRATION_POINT", "x": 192.0, "y": 108.0, "is_validation": False}

    assert logic.apply(event) is True
    assert logic.apply(event) is False


def test_completion_hides_the_target():
    logic = PresenterLogic()
    logic.apply({"type": "CALIBRATION_POINT", "x": 192.0, "y": 108.0, "is_validation": False})

    assert logic.apply({"type": "CALIBRATION_DONE"}) is True
    assert logic.dot.visible is False
    assert logic.is_complete() is True


def test_the_recorded_screen_size_travels_with_the_recording(tmp_path):
    path = tmp_path / "labeled.jsonl"
    _record(path)

    labeled = _read(path)

    assert (labeled.screen_w, labeled.screen_h) == (SCREEN_W, SCREEN_H)
    assert labeled.screen_conflicts == 0


def test_a_screen_size_that_disagrees_with_the_recording_is_refused(tmp_path):
    """
    A display running a scale factor reports fewer pixels than the panel has to a process that
    has not declared DPI awareness, so one machine offers two plausible screen sizes. The fit
    succeeds under either and the error is reported in degrees scaled by the ratio between them,
    so the mismatch has to be refused outright rather than left to be noticed downstream.
    """
    path = tmp_path / "labeled.jsonl"
    _record(path)

    assert run_labeled_harness(str(path), SCREEN_W, SCREEN_H, DIAG_MM) is not None
    assert run_labeled_harness(str(path), SCALED_SCREEN_W, SCALED_SCREEN_H, DIAG_MM) is None


def test_a_recording_reporting_two_screen_sizes_is_refused(tmp_path):
    """
    No production path produces this, since one run reads one session, so the file is edited
    directly. Concatenated or spliced recordings are what it guards.
    """
    path = tmp_path / "labeled.jsonl"
    _record(path)

    lines = path.read_text().splitlines()
    edited = []
    for index, line in enumerate(lines):
        data = json.loads(line)
        if data.get(LABEL_KEY) and index > len(lines) // 2:
            data[LABEL_KEY]["screen_w"] = SCALED_SCREEN_W
            data[LABEL_KEY]["screen_h"] = SCALED_SCREEN_H
        edited.append(json.dumps(data))
    path.write_text("\n".join(edited) + "\n")

    labeled = _read(path)

    assert labeled.screen_conflicts > 0
    assert run_labeled_harness(str(path), SCREEN_W, SCREEN_H, DIAG_MM) is None


def test_the_acceptance_threshold_travels_with_the_recording(tmp_path):
    path = tmp_path / "labeled.jsonl"
    _record(path)

    assert _read(path).dispersion_threshold == session_module.DISPERSION_THRESHOLD


def test_a_loosened_threshold_is_recorded_as_the_value_it_was_run_at(tmp_path):
    """
    The threshold shipped with the product is not reachable on every camera, so a recording may
    have to be collected under a wider gate than the one being evaluated. What the recording must
    not do is stay silent about which gate it was collected under: how many targets the person was
    asked to look at twice, and therefore which windows exist in the file at all, follows from it.

    The unstable target is held wider than the default gate and narrower than the loosened one, so
    the recorded value is shown to have been applied rather than merely written down.
    """
    default_path = tmp_path / "default.jsonl"
    loose_path = tmp_path / "loose.jsonl"

    default_session = _record(default_path, marginal_idx=0, marginal_spread=UNSTABLE_SPREAD)
    loose_session = _record(loose_path, marginal_idx=0, dispersion_threshold=LOOSE_THRESHOLD,
                            marginal_spread=UNSTABLE_SPREAD)

    assert default_session.state == "FAILED"
    assert loose_session.state == "DONE"

    assert _read(default_path).dispersion_threshold == session_module.DISPERSION_THRESHOLD
    assert _read(loose_path).dispersion_threshold == LOOSE_THRESHOLD
