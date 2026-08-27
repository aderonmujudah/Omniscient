from typing import Optional, List
from engine.events.interaction import InteractionEvent, EventType

class DwellTimer:
    def __init__(self, dwell_duration_s: float = 0.8, progress_interval_s: float = 0.05) -> None:
        self._dwell_duration_s = dwell_duration_s
        self._progress_interval_s = progress_interval_s
        self._active_zone: Optional[str] = None
        self._dwell_start_t: float = 0.0
        self._last_progress_t: float = 0.0
        self._completed: bool = False
        self._start_x: float = 0.0
        self._start_y: float = 0.0

    def update(self, zone_id: Optional[str], x: float, y: float, timestamp: float) -> List[InteractionEvent]:
        """Update the dwell timer with current gaze position."""
        events: List[InteractionEvent] = []
        
        if zone_id != self._active_zone:
            if self._active_zone is not None and not self._completed:
                events.append(InteractionEvent(
                    event_type=EventType.DWELL_CANCEL.value,
                    timestamp=timestamp,
                    zone_id=self._active_zone,
                ))
            
            if zone_id is not None:
                self._active_zone = zone_id
                self._dwell_start_t = timestamp
                self._last_progress_t = timestamp
                self._completed = False
                self._start_x = x
                self._start_y = y
                events.append(InteractionEvent(
                    event_type=EventType.DWELL_START.value,
                    timestamp=timestamp,
                    zone_id=zone_id,
                    x=x,
                    y=y,
                ))
            else:
                self._active_zone = None
                self._completed = False
        
        elif zone_id is not None and not self._completed:
            elapsed = timestamp - self._dwell_start_t
            
            # Use shorter dwell inside radial menu
            current_duration = 0.4 if zone_id.startswith("radial_") else self._dwell_duration_s
            progress = min(elapsed / current_duration, 1.0)
            
            if progress >= 1.0:
                self._completed = True
                events.append(InteractionEvent(
                    event_type=EventType.DWELL_COMPLETE.value,
                    timestamp=timestamp,
                    zone_id=zone_id,
                ))
            elif timestamp - self._last_progress_t >= self._progress_interval_s:
                self._last_progress_t = timestamp
                events.append(InteractionEvent(
                    event_type=EventType.DWELL_PROGRESS.value,
                    timestamp=timestamp,
                    zone_id=zone_id,
                    progress=progress,
                ))
        
        return events

    def cancel(self, timestamp: float) -> Optional[InteractionEvent]:
        """Force-cancel any active dwell."""
        if self._active_zone is not None and not self._completed:
            event = InteractionEvent(
                event_type=EventType.DWELL_CANCEL.value,
                timestamp=timestamp,
                zone_id=self._active_zone,
            )
            self._active_zone = None
            self._completed = False
            return event
        self._active_zone = None
        self._completed = False
        return None

    def reset(self) -> None:
        self._active_zone = None
        self._dwell_start_t = 0.0
        self._last_progress_t = 0.0
        self._completed = False
