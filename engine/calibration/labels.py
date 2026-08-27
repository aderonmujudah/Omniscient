import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

from engine.calibration.session import (
    compute_dispersion,
    sample_feature,
    window_accepted,
)

logger = logging.getLogger(__name__)

# States in which a target is on screen. Outside them the display shows no dot, so there is
# nothing to label.
PRESENTING_STATES = ("SETTLING", "COLLECTING")


def make_label_provider(session):
    """
    Returns a zero-argument callable reporting what the calibration display is showing, for a
    recorder to store alongside each sample.

    Ground truth is recorded rather than re-derived because re-running the state machine over a
    recording reproduces the original pairing only while the acceptance constants are
    unchanged. Tuning one of them turns an accepted window into a rejected one; the machine
    then re-presents a target the recorded person has already looked away from, and every
    later window is averaged against a target that was never on screen while it was captured.
    """
    def provider() -> Optional[dict]:
        if session.state not in PRESENTING_STATES:
            return None
        idx = session.current_point_idx
        tx, ty = session.all_targets[idx]
        return {
            "point_idx": idx,
            "x": tx,
            "y": ty,
            "state": session.state,
            "attempt": session.retries,
            "is_validation": idx >= len(session.fit_targets),
            # Target coordinates are meaningless without the dimensions they are expressed in.
            # A display running a scale factor reports fewer pixels to a process that has not
            # declared DPI awareness, so the same panel yields two different coordinate spaces
            # and neither is recoverable from the coordinates alone.
            "screen_w": session.screen_w,
            "screen_h": session.screen_h,
            # Which windows were accepted, and therefore which targets the person was asked to
            # look at a second time, depends on this value. Offline analysis at a candidate
            # threshold has to know the one the recording was made under to say how far the
            # recording supports the candidate.
            "dispersion_threshold": session.dispersion_threshold,
        }
    return provider


@dataclass
class CollectionWindow:
    """One uninterrupted run of samples captured while a single target was being collected."""
    point_idx: int
    attempt: int
    target: tuple[float, float]
    is_validation: bool
    features: list[tuple[float, float]] = field(default_factory=list)
    ipds: list[Optional[float]] = field(default_factory=list)

    def dispersion(self) -> float:
        return compute_dispersion(self.features)

    def mean_feature(self) -> tuple[float, float]:
        n = len(self.features)
        return (sum(f[0] for f in self.features) / n,
                sum(f[1] for f in self.features) / n)

    def mean_ipd(self) -> float:
        valid = [i for i in self.ipds if i]
        return sum(valid) / len(valid) if valid else 0.0


@dataclass
class LabeledSession:
    windows: list[CollectionWindow] = field(default_factory=list)
    labeled_count: int = 0
    frame_width: Optional[int] = None
    screen_w: Optional[int] = None
    screen_h: Optional[int] = None
    screen_conflicts: int = 0
    dispersion_threshold: Optional[float] = None


def read_labeled_session(labeled_samples: Iterable) -> LabeledSession:
    """
    Groups labeled samples into collection windows.

    labeled_count is reported so a caller can tell a recording made before the label track
    existed from one whose targets are known. Zero means what was on screen is unrecoverable,
    and such a recording must not be analysed as though it carried ground truth.
    """
    result = LabeledSession()
    key = None

    for sample, label in labeled_samples:
        if result.frame_width is None and sample.frame_width is not None:
            result.frame_width = sample.frame_width

        if label is None:
            key = None
            continue

        result.labeled_count += 1

        recorded = (label.get("screen_w"), label.get("screen_h"))
        if recorded != (None, None):
            if result.screen_w is None:
                result.screen_w, result.screen_h = recorded
            elif recorded != (result.screen_w, result.screen_h):
                result.screen_conflicts += 1

        if result.dispersion_threshold is None:
            result.dispersion_threshold = label.get("dispersion_threshold")

        if label.get("state") != "COLLECTING":
            # Settling samples carry a label so the boundary survives tuning, but they never
            # reach a fit: they were captured while the gaze was still travelling.
            key = None
            continue

        this_key = (label["point_idx"], label["attempt"])
        if this_key != key:
            result.windows.append(CollectionWindow(
                point_idx=label["point_idx"],
                attempt=label["attempt"],
                target=(label["x"], label["y"]),
                is_validation=label["is_validation"],
            ))
            key = this_key

        feature = sample_feature(sample)
        if feature is not None:
            result.windows[-1].features.append(feature)
            result.windows[-1].ipds.append(sample.ipd_px)

    return result


def select_accepted(windows: list[CollectionWindow],
                    threshold: Optional[float] = None) -> tuple[list[CollectionWindow], int]:
    """
    Applies the live acceptance rule to recorded windows and returns one window per target,
    together with the number of targets whose selected window is not the one the recording
    itself accepted.

    A recording cannot show what the person would have done had a window been judged
    differently while they sat there, so that count is a limit on how far this recording
    supports the candidate value. It is returned rather than absorbed.
    """
    chosen: dict[int, CollectionWindow] = {}
    recorded_final: dict[int, CollectionWindow] = {}

    for w in windows:
        recorded_final[w.point_idx] = w
        if w.point_idx not in chosen and window_accepted(w.features, threshold):
            chosen[w.point_idx] = w

    diverged = sum(1 for idx, w in chosen.items() if recorded_final.get(idx) is not w)
    return [chosen[idx] for idx in sorted(chosen)], diverged


def split_fit_val(accepted: list[CollectionWindow]) -> tuple[list[CollectionWindow], list[CollectionWindow]]:
    """
    Splits by the recorded validation flag rather than by position in the sequence. A window
    dropped by tuning would otherwise shift a validation target into the set the model is
    fitted on, and the accuracy figure would then be computed on a point the model had seen.
    """
    return ([w for w in accepted if not w.is_validation],
            [w for w in accepted if w.is_validation])
