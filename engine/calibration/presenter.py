from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DotState:
    visible: bool
    x: float
    y: float


HIDDEN = DotState(False, 0.0, 0.0)


class PresenterLogic:
    """
    Maps calibration events to what must be drawn. No windowing toolkit is involved, so the
    mapping is testable without a display.
    """

    def __init__(self):
        self.dot = HIDDEN
        self.finished = False
        self.failed = False

    def apply(self, event: Optional[dict]) -> bool:
        """
        Folds one event in and reports whether the drawn output changed.

        is_validation is read but deliberately not used. The validation points are the ones the
        accuracy figure is computed from, and distinguishing them on screen would change how
        they are looked at.

        A re-presented target produces no change, so a rejected point is invisible to the
        person: nothing on screen marks when a collection window opens or restarts, and there
        is therefore nothing for them to time their gaze against.
        """
        if event is None:
            return False

        event_type = event.get("type")
        if event_type == "CALIBRATION_POINT":
            new_dot = DotState(True, event["x"], event["y"])
        elif event_type == "CALIBRATION_DONE":
            self.finished = True
            new_dot = HIDDEN
        elif event_type == "CALIBRATION_FAILED":
            self.failed = True
            new_dot = HIDDEN
        else:
            return False

        if new_dot == self.dot:
            return False
        self.dot = new_dot
        return True

    def is_complete(self) -> bool:
        return self.finished or self.failed
