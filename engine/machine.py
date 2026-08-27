import math
import logging
import base64
try:
    import cv2
except ImportError:
    cv2 = None
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from engine.capture.base import ScreenCapture
from engine.events.interaction import InteractionEvent, EventType

logger = logging.getLogger(__name__)

@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

def derive_grid(
    acc_deg: float, 
    viewing_dist_mm: float, 
    screen_w_px: int, 
    screen_h_px: int, 
    screen_diag_mm: float,
    cell_size_radii: float = 1.0
) -> Optional[Tuple[int, int]]:
    """
    Derive grid dimensions from measured accuracy and viewing distance.
    
    Reasoning:
    1. Calculate pixel density from the screen's physical diagonal and pixel dimensions.
    2. Calculate the physical radius of the gaze uncertainty cone on the screen surface
       using the viewing distance and accuracy angle.
    3. Convert that physical radius to pixels.
    4. A cell's minimum dimension is proportional to the uncertainty radius (controlled by cell_size_radii).
       (Note: whether this should be 1 radius or 2 radii is an open question raised for §10).
    5. The grid dimension is the screen dimension divided by the cell size, floored.
    
    If the resulting grid is 1x1, it provides no targeting value and we return None (refusal).
    """
    diag_px = math.hypot(screen_w_px, screen_h_px)
    if screen_diag_mm <= 0:
        return None
    px_per_mm = diag_px / screen_diag_mm
    
    # Uncertainty radius in mm = viewing_distance * tan(accuracy in radians)
    radius_mm = viewing_dist_mm * math.tan(math.radians(acc_deg))
    radius_px = radius_mm * px_per_mm
    
    cell_size = radius_px * cell_size_radii
    
    if cell_size <= 0:
        return None
        
    cols = max(1, int(screen_w_px / cell_size))
    rows = max(1, int(screen_h_px / cell_size))
    
    if cols < 2 and rows < 2:
        return None
        
    return cols, rows

def compute_cells(screen_w: int, screen_h: int, cols: int, rows: int) -> List[List[Rect]]:
    """
    Compute exactly the bounding boxes of every cell in a grid.
    Distributes remainder pixels.
    """
    cells = []
    base_w = screen_w // cols
    rem_w = screen_w % cols
    
    base_h = screen_h // rows
    rem_h = screen_h % rows
    
    y = 0
    for r in range(rows):
        h = base_h + (1 if r < rem_h else 0)
        row_cells = []
        x = 0
        for c in range(cols):
            w = base_w + (1 if c < rem_w else 0)
            row_cells.append(Rect(x, y, w, h))
            x += w
        cells.append(row_cells)
        y += h
    return cells

def get_cell_for_point(cells: List[List[Rect]], px: float, py: float) -> Optional[Tuple[int, int]]:
    for r, row in enumerate(cells):
        for c, rect in enumerate(row):
            if rect.x <= px < rect.x + rect.w and rect.y <= py < rect.y + rect.h:
                return r, c
    if not cells:
        return None
    c_idx = -1
    r_idx = -1
    for c, rect in enumerate(cells[0]):
        if rect.x <= px <= rect.x + rect.w:
            c_idx = c
            break
    for r, row in enumerate(cells):
        rect = row[0]
        if rect.y <= py <= rect.y + rect.h:
            r_idx = r
            break
    if r_idx != -1 and c_idx != -1:
        return r_idx, c_idx
    return None

class StateMachine:
    def __init__(
        self, 
        screen_w: int, 
        screen_h: int, 
        screen_diag_mm: float, 
        acc_deg: float, 
        viewing_dist_mm: float, 
        dispatcher,
        capture_backend: ScreenCapture,
        input_backend=None,
        recalibrator=None
    ):
        self.state = "IDLE"
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.screen_diag_mm = screen_diag_mm
        self.acc_deg = acc_deg
        self.viewing_dist_mm = viewing_dist_mm
        
        self.grid_cols = 0
        self.grid_rows = 0
        self.cells = []
        self.zoom1_rect = None
        self.zoom2_rect = None
        self.resolved_point = None
        self.frozen_image_b64 = None
        
        self.last_input_time = 0.0
        self.dispatcher = dispatcher
        self.capture = capture_backend
        self.input_backend = input_backend
        self.recalibrator = recalibrator
        self.latest_features = None

    def process_event(self, event: InteractionEvent):
        # We only consider these events for timeout tracking
        if event.event_type in (EventType.GESTURE.value, EventType.DWELL_COMPLETE.value, EventType.GAZE_MOVE.value):
            if event.event_type != EventType.GAZE_MOVE.value:
                self.last_input_time = event.timestamp

        if event.event_type == EventType.GESTURE.value:
            if event.role == "ENGAGE":
                if self.state == "IDLE":
                    self.enter_grid(event.timestamp)
                elif self.state in ("ZOOM1", "ZOOM2"):
                    # The radial menu center is the coordinate where ENGAGE happened
                    self.enter_radial(event.x, event.y, event.timestamp)
            elif event.role == "CANCEL":
                self.step_back(event.timestamp)
                
        elif event.event_type == EventType.DWELL_COMPLETE.value:
            if self.state == "GRID":
                cell_idx = get_cell_for_point(self.cells, event.x, event.y)
                if cell_idx:
                    r, c = cell_idx
                    self.enter_zoom1(self.cells[r][c], event.timestamp)
            elif self.state == "ZOOM1":
                orig_x, orig_y = self._map_to_orig(event.x, event.y, self.zoom1_rect)
                zw = self.zoom1_rect.w / self.grid_cols
                zh = self.zoom1_rect.h / self.grid_rows
                zx = orig_x - zw / 2
                zy = orig_y - zh / 2
                self.enter_zoom2(Rect(int(zx), int(zy), int(zw), int(zh)), event.timestamp)
            elif self.state == "ZOOM2":
                orig_x, orig_y = self._map_to_orig(event.x, event.y, self.zoom2_rect)
                self.resolved_point = (orig_x, orig_y)
                self._execute_action("left_click")
                self.change_state("IDLE", event.timestamp)
            elif self.state == "RADIAL":
                if event.zone_id and event.zone_id.startswith("radial_"):
                    action = event.zone_id.replace("radial_", "")
                    if action == "cancel":
                        self.change_state("IDLE", event.timestamp)
                    else:
                        self._execute_action(action)
                        self.change_state("IDLE", event.timestamp)

    def _map_to_orig(self, px, py, rect):
        px_ratio = px / self.screen_w
        py_ratio = py / self.screen_h
        orig_x = rect.x + px_ratio * rect.w
        orig_y = rect.y + py_ratio * rect.h
        return orig_x, orig_y

    def _execute_action(self, action: str):
        if not self.resolved_point:
            return
        x, y = self.resolved_point
        
        # Implicit recalibration on successful activation
        if self.recalibrator and self.latest_features and action != "cancel":
            self.recalibrator.hook_successful_activation(self.latest_features, (x, y))

        if self.input_backend:
            if action == "left_click":
                self.input_backend.click(x, y, button="left")
            elif action == "right_click":
                self.input_backend.click(x, y, button="right")
            elif action == "double_click":
                self.input_backend.click(x, y, button="left", count=2)
            elif action == "middle_click":
                self.input_backend.click(x, y, button="middle")
            elif action == "drag_start":
                self.input_backend.drag("start", x, y)
            elif action == "drag_end":
                self.input_backend.drag("end", x, y)

    def enter_grid(self, timestamp: float):
        grid = derive_grid(self.acc_deg, self.viewing_dist_mm, self.screen_w, self.screen_h, self.screen_diag_mm)
        if not grid:
            logger.warning("Derived grid is unusable, refusing to enter GRID.")
            return
            
        self.grid_cols, self.grid_rows = grid
        self.cells = compute_cells(self.screen_w, self.screen_h, self.grid_cols, self.grid_rows)
        
        # Capture and freeze screen
        img = self.capture.capture()
        if cv2 is not None:
            _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self.frozen_image_b64 = base64.b64encode(buf).decode('utf-8')
        else:
            self.frozen_image_b64 = "dummy_base64_image"
        
        self.change_state("GRID", timestamp)

    def enter_zoom1(self, rect: Rect, timestamp: float):
        self.zoom1_rect = rect
        self.change_state("ZOOM1", timestamp)

    def enter_zoom2(self, rect: Rect, timestamp: float):
        self.zoom2_rect = rect
        self.change_state("ZOOM2", timestamp)

    def enter_radial(self, px: float, py: float, timestamp: float):
        if self.state == "ZOOM1" and self.zoom1_rect:
            self.resolved_point = self._map_to_orig(px, py, self.zoom1_rect)
        elif self.state == "ZOOM2" and self.zoom2_rect:
            self.resolved_point = self._map_to_orig(px, py, self.zoom2_rect)
        else:
            self.resolved_point = (px, py)
        self.change_state("RADIAL", timestamp)

    def enter_resolved(self, x: float, y: float, timestamp: float):
        self.resolved_point = (x, y)
        self.change_state("RESOLVED", timestamp)

    def step_back(self, timestamp: float):
        if self.state == "RADIAL":
            # If we cancel radial, we go back to IDLE
            self.change_state("IDLE", timestamp)
        elif self.state == "RESOLVED":
            self.change_state("ZOOM2", timestamp)
        elif self.state == "ZOOM2":
            self.change_state("ZOOM1", timestamp)
        elif self.state == "ZOOM1":
            self.change_state("GRID", timestamp)
        elif self.state == "GRID":
            self.change_state("IDLE", timestamp)

    def check_timeout(self, timestamp: float):
        if self.state != "IDLE" and timestamp - self.last_input_time >= 8.0:
            self.change_state("IDLE", timestamp)

    def change_state(self, to_state: str, timestamp: float):
        from_state = self.state
        self.state = to_state
        
        if to_state == "IDLE":
            self.frozen_image_b64 = None
            self.cells = []
            
        event = InteractionEvent(
            event_type=EventType.STATE_CHANGE.value,
            timestamp=timestamp,
            from_state=from_state,
            to_state=to_state,
            image_b64=self.frozen_image_b64 if to_state in ("GRID", "ZOOM1", "ZOOM2", "RADIAL") else None,
            grid_dim=[self.grid_cols, self.grid_rows] if self.grid_cols > 0 else None,
            cells=[[{"x": c.x, "y": c.y, "w": c.w, "h": c.h} for c in row] for row in self.cells] if self.cells else None,
            zoom_rect={"x": self.zoom1_rect.x, "y": self.zoom1_rect.y, "w": self.zoom1_rect.w, "h": self.zoom1_rect.h} if to_state == "ZOOM1" and self.zoom1_rect else ({"x": self.zoom2_rect.x, "y": self.zoom2_rect.y, "w": self.zoom2_rect.w, "h": self.zoom2_rect.h} if to_state == "ZOOM2" and self.zoom2_rect else None),
            x=self.resolved_point[0] if self.resolved_point and to_state in ("RESOLVED", "RADIAL") else None,
            y=self.resolved_point[1] if self.resolved_point and to_state in ("RESOLVED", "RADIAL") else None
        )
        self.dispatcher.dispatch(event)

    def resolve_zone(self, fx: float, fy: float) -> Optional[str]:
        if self.state == "GRID":
            cell = get_cell_for_point(self.cells, fx, fy)
            if cell:
                return f"cell_{cell[0]}_{cell[1]}"
        elif self.state in ("ZOOM1", "ZOOM2"):
            # The whole screen is a single zone for selecting the point
            return "zoom_point"
        elif self.state == "RADIAL" and self.resolved_point:
            # We assume fx, fy is now in original screen coords?
            # Wait, no. The overlay clears the zoom in RADIAL state.
            # So the user looks at the wedge.
            dx = fx - self.resolved_point[0]
            dy = fy - self.resolved_point[1]
            if dx == 0 and dy == 0:
                return None
            distance = math.hypot(dx, dy)
            if distance < 50: # deadzone
                return None
            angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
            wedges = ["left_click", "right_click", "double_click", "middle_click", "drag_start", "drag_end", "cancel"]
            wedge_idx = int((angle + 360 / 14) % 360 / (360 / 7))
            return f"radial_{wedges[wedge_idx]}"
        return None
