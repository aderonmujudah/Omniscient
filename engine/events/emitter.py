from __future__ import annotations

import logging
from typing import Callable, List, Optional

from engine.events.dispatcher import EventDispatcher
from engine.events.dwell import DwellTimer
from engine.events.gestures.registry import GestureRegistry
from engine.events.interaction import EventType, InteractionEvent
from engine.filtering.classifier import SampleClassifier, SampleState
from engine.filtering.one_euro import OneEuroFilter
from engine.filtering.ema import EMAFilter
from engine.sources.base import GazeSample

logger = logging.getLogger(__name__)


class InteractionEmitter:
    """Assembles the InteractionEvent stream from calibrated gaze and publishes it.

    This is the only component that owns the order the signal layers run in: smooth, then
    classify, then detect. Each layer is independently testable, but a layer that no
    composition runs contributes nothing to the product, so the running engine builds this
    rather than calling the layers itself.
    """

    def __init__(
        self,
        *,
        gaze_filter: OneEuroFilter,
        classifier: SampleClassifier,
        registry: GestureRegistry,
        dispatcher: EventDispatcher,
        dwell: Optional[DwellTimer] = None,
        zone_resolver: Optional[Callable[[float, float], Optional[str]]] = None,
    ) -> None:
        """
        Args:
            dwell: Timer for dwell against an arbitrary zone. Reserved-zone dwell is resolved
                inside the registry as a gesture; this timer serves zones the interface
                defines, and stays idle until a zone resolver is supplied.
            zone_resolver: Maps a filtered screen position to the zone containing it, or None
                when the position is over no zone. The zones themselves belong to whatever
                draws them, so they are resolved through a caller-supplied function rather
                than known here.
        """
        self._filter = gaze_filter
        self._classifier = classifier
        self._registry = registry
        self._dispatcher = dispatcher
        self._dwell = dwell
        self._zone_resolver = zone_resolver

        self._blink_filter_left = EMAFilter(rate=30.0, cutoff=5.0)
        self._blink_filter_right = EMAFilter(rate=30.0, cutoff=5.0)
        self._blink_score_filter = EMAFilter(rate=30.0, cutoff=5.0)

        self._tracking_ok = True
        self._in_fixation = False

    def process_sample(
        self, sample: GazeSample, gaze_x: Optional[float], gaze_y: Optional[float]
    ) -> List[InteractionEvent]:
        """Turn one sample into the events it produces, dispatch them, and return them.

        Args:
            gaze_x: Calibrated horizontal screen coordinate, or None when no calibration
                model is loaded. Without a position the signal layers cannot run, and only
                tracking state is reported.
        """
        events: List[InteractionEvent] = []

        if not sample.ok or gaze_x is None or gaze_y is None:
            events.extend(self._handle_tracking_loss(sample))
            self._dispatcher.dispatch_many(events)
            return events

        if not self._tracking_ok:
            self._tracking_ok = True
            events.append(InteractionEvent(
                event_type=EventType.TRACKING_RESUMED.value, timestamp=sample.t,
            ))

        fx, fy = self._filter(gaze_x, gaze_y, sample.t)

        ear = sample.ear or {}
        filtered_ear = None
        if "left" in ear and "right" in ear:
            filtered_ear = {
                "left": self._blink_filter_left(ear["left"], sample.t),
                "right": self._blink_filter_right(ear["right"], sample.t)
            }
            sample.ear = filtered_ear
        
        if sample.blink_score is not None:
            sample.blink_score = self._blink_score_filter(sample.blink_score, sample.t)

        state, completed = self._classifier.classify(
            ok=sample.ok,
            ear_left=filtered_ear.get("left") if filtered_ear else None,
            ear_right=filtered_ear.get("right") if filtered_ear else None,
            gaze_x=fx,
            gaze_y=fy,
            timestamp=sample.t,
            blink_score=sample.blink_score,
        )

        if not hasattr(self, '_gaze_buffer'):
            self._gaze_buffer = []

        if state is SampleState.BLINK:
            self._gaze_buffer.clear()
        else:
            self._gaze_buffer.append((sample.t, fx, fy))
            # 4 frames of lag at 30fps = ~133ms. Hides the eyelid closure darting.
            if len(self._gaze_buffer) > 4:
                emit_t, emit_x, emit_y = self._gaze_buffer.pop(0)
                events.append(InteractionEvent(
                    event_type=EventType.GAZE_MOVE.value, timestamp=emit_t, x=emit_x, y=emit_y,
                ))

        events.extend(self._fixation_events(state, completed, sample.t))

        # Gestures run on the calibrated position rather than the smoothed one. The filter
        # trades lag for stability, which is correct for a cursor and wrong for a boundary
        # crossing or a displacement measured over a few frames.
        for gesture in self._registry.process_sample(sample, gaze_x, gaze_y, self._in_fixation):
            events.append(InteractionEvent(
                event_type=EventType.GESTURE.value,
                timestamp=gesture.timestamp,
                x=gesture.gaze_x,
                y=gesture.gaze_y,
                role=gesture.role.value,
            ))

        if self._dwell is not None and self._zone_resolver is not None:
            zone_id = self._zone_resolver(fx, fy)
            events.extend(self._dwell.update(zone_id, fx, fy, sample.t))

        self._dispatcher.dispatch_many(events)
        return events

    def _handle_tracking_loss(self, sample: GazeSample) -> List[InteractionEvent]:
        """Report the loss once and cancel anything in progress that depended on position."""
        events: List[InteractionEvent] = []
        if not self._tracking_ok:
            return events

        self._tracking_ok = False
        reason = "no_face" if not sample.ok else "no_calibration"
        events.append(InteractionEvent(
            event_type=EventType.TRACKING_LOST.value, timestamp=sample.t, reason=reason,
        ))

        if self._in_fixation:
            self._in_fixation = False
        self._classifier.reset()
        self._filter.reset()
        self._registry.reset()

        if self._dwell is not None:
            cancelled = self._dwell.cancel(sample.t)
            if cancelled is not None:
                events.append(cancelled)

        return events

    def _fixation_events(
        self, state: SampleState, completed, timestamp: float
    ) -> List[InteractionEvent]:
        """Report fixation boundaries. A completed fixation carries the centroid, which is
        the position the interface should act on rather than the latest raw point."""
        events: List[InteractionEvent] = []

        if completed is not None:
            events.append(InteractionEvent(
                event_type=EventType.FIXATION_END.value,
                timestamp=completed.end_t,
                x=completed.x,
                y=completed.y,
                duration_ms=completed.duration_s * 1000.0,
            ))
            self._in_fixation = False

        if state is SampleState.FIXATION and not self._in_fixation:
            centroid = self._classifier.active_fixation_centroid
            if centroid is not None:
                self._in_fixation = True
                events.append(InteractionEvent(
                    event_type=EventType.FIXATION_START.value,
                    timestamp=timestamp,
                    x=centroid[0],
                    y=centroid[1],
                ))
        elif state is not SampleState.FIXATION:
            self._in_fixation = False

        return events

    def reset(self) -> None:
        self._filter.reset()
        self._blink_filter_left.reset()
        self._blink_filter_right.reset()
        self._blink_score_filter.reset()
        self._classifier.reset()
        self._registry.reset()
        if self._dwell is not None:
            self._dwell.reset()
        self._tracking_ok = True
        self._in_fixation = False
