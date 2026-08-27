from engine.input.base import InputBackend

class ScrollController:
    def __init__(self, screen_h: int, input_backend: InputBackend):
        self.screen_h = screen_h
        self.input_backend = input_backend
        self.zone_h = int(screen_h * 0.08)
        
        self.active_zone = None
        self.zone_entry_time = None
        self.last_scroll_time = None
        
    def update(self, x: float, y: float, timestamp: float, paused: bool):
        if not self.input_backend or paused:
            self.active_zone = None
            self.zone_entry_time = None
            return
            
        current_zone = None
        proximity = 0.0
        
        if y < self.zone_h:
            current_zone = "top"
            proximity = 1.0 - (y / self.zone_h)
        elif y > self.screen_h - self.zone_h:
            current_zone = "bottom"
            proximity = 1.0 - ((self.screen_h - y) / self.zone_h)
            
        if current_zone != self.active_zone:
            self.active_zone = current_zone
            self.zone_entry_time = timestamp if current_zone else None
            self.last_scroll_time = None
            
        if self.active_zone and self.zone_entry_time is not None:
            if timestamp - self.zone_entry_time >= 0.4: # Armed after 400ms
                if self.last_scroll_time is None:
                    self.last_scroll_time = timestamp
                else:
                    dt = timestamp - self.last_scroll_time
                    if dt > 0:
                        # Ramp rate based on proximity (0.0 to 1.0)
                        # e.g., max 1000 units per second
                        rate = proximity * 1000.0
                        dy = rate * dt
                        if self.active_zone == "top":
                            self.input_backend.scroll(x, y, dy) # scroll up
                        else:
                            self.input_backend.scroll(x, y, -dy) # scroll down
                        self.last_scroll_time = timestamp

