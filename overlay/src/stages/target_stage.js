export class TargetStage {
    constructor() {
        this.layer = document.getElementById('targeting-layer');
        this.bg = document.getElementById('frozen-bg');
        this.grid = document.getElementById('grid-container');
        this.marker = document.getElementById('resolved-marker');
        this.cellsData = [];
        this.state = "IDLE";
        this.zoomRect = null;
        this.image_w = 1920;
        this.image_h = 1080;
    }

    onStateChange(event) {
        this.state = event.to_state;
        if (this.state === "IDLE") {
            this.layer.style.display = "none";
            this.bg.style.backgroundImage = "none";
            this.grid.innerHTML = "";
            this.cellsData = [];
            this.marker.style.display = "none";
            return;
        }

        this.layer.style.display = "block";
        this.marker.style.display = "none";

        if (event.image_b64) {
            this.bg.style.backgroundImage = `url(data:image/jpeg;base64,${event.image_b64})`;
            this.bg.style.backgroundSize = "100% 100%"; // stretch to fit screen
            this.bg.style.backgroundPosition = "0 0";
            // Wait, does zoom background need adjusting?
            // In ZOOM, the background should display the magnified portion of the frozen image.
            // If the whole screen maps to zoomRect, then we scale and translate the background.
        }

        if (this.state === "GRID") {
            this.bg.style.backgroundSize = "100% 100%";
            this.bg.style.backgroundPosition = "0 0";
            this.grid.innerHTML = "";
            this.cellsData = [];
            if (event.cells) {
                event.cells.forEach((row, r) => {
                    row.forEach((cData, c) => {
                        const cell = document.createElement('div');
                        cell.className = 'grid-cell';
                        cell.style.left = cData.x + 'px';
                        cell.style.top = cData.y + 'px';
                        cell.style.width = cData.w + 'px';
                        cell.style.height = cData.h + 'px';
                        cell.id = `cell_${r}_${c}`;
                        this.grid.appendChild(cell);
                        this.cellsData.push(cData);
                    });
                });
            }
        } else if (this.state === "ZOOM1" || this.state === "ZOOM2") {
            this.grid.innerHTML = "";
            if (event.zoom_rect) {
                this.zoomRect = event.zoom_rect;
                const rw = this.zoomRect.w;
                const rh = this.zoomRect.h;
                const rx = this.zoomRect.x;
                const ry = this.zoomRect.y;
                // Scale the background image so that zoom_rect fills the screen
                const scaleX = window.innerWidth / rw;
                const scaleY = window.innerHeight / rh;
                this.bg.style.backgroundSize = `${scaleX * 100}% ${scaleY * 100}%`;
                this.bg.style.backgroundPosition = `${-rx * scaleX}px ${-ry * scaleY}px`;

                // No cell lattice is drawn over the magnified view. Selection here is continuous:
                // the engine centres the next window on wherever the gaze rests, so drawn cell
                // boundaries would advertise discrete targets that do not exist. They would also be
                // a second implementation of the cell arithmetic, computed from the window's CSS
                // width rather than from the screen rectangles the engine resolves against, and the
                // two disagree wherever the display is scaled.
            }
        } else if (this.state === "RADIAL") {
            this.grid.innerHTML = "";
            this.bg.style.backgroundSize = "100% 100%";
            this.bg.style.backgroundPosition = "0 0";
            
            // Draw radial menu around event.x, event.y
            if (event.x !== undefined && event.y !== undefined) {
                const cx = event.x;
                const cy = event.y;
                const wedges = ["Left", "Right", "Dbl", "Mid", "Drag In", "Drag Out", "Cancel"];
                const radius = 150;
                for (let i = 0; i < wedges.length; i++) {
                    const angle = (i * 360 / 7) * Math.PI / 180;
                    const wx = cx + Math.cos(angle) * radius;
                    const wy = cy + Math.sin(angle) * radius;
                    
                    const label = document.createElement('div');
                    label.style.position = 'absolute';
                    label.style.left = wx + 'px';
                    label.style.top = wy + 'px';
                    label.style.transform = 'translate(-50%, -50%)';
                    label.style.background = 'rgba(0,0,0,0.7)';
                    label.style.padding = '10px';
                    label.style.borderRadius = '5px';
                    label.style.color = 'white';
                    label.style.fontSize = '18px';
                    label.style.pointerEvents = 'none';
                    label.innerText = wedges[i];
                    
                    // We need id to match zone_id for dwell progress highlighting
                    const zoneMap = ["radial_left_click", "radial_right_click", "radial_double_click", "radial_middle_click", "radial_drag_start", "radial_drag_end", "radial_cancel"];
                    label.id = zoneMap[i];
                    
                    this.grid.appendChild(label);
                }
                
                // Draw center point
                this.marker.style.left = cx + 'px';
                this.marker.style.top = cy + 'px';
                this.marker.style.display = "block";
            }
        } else if (this.state === "SYSTEM_MENU") {
            this.grid.innerHTML = "";
            this.bg.style.background = "rgba(0, 0, 0, 0.8)";
            this.bg.style.backgroundSize = "100% 100%";
            this.bg.style.backgroundPosition = "0 0";
            
            const quadrants = [
                {id: "sys_resume", label: "Resume", left: "0%", top: "0%", bg: "rgba(50, 150, 50, 0.6)"},
                {id: "sys_recalibrate", label: "Recalibrate", left: "50%", top: "0%", bg: "rgba(50, 50, 150, 0.6)"},
                {id: "sys_pause", label: "Pause", left: "0%", top: "50%", bg: "rgba(150, 150, 50, 0.6)"},
                {id: "sys_quit", label: "Quit", left: "50%", top: "50%", bg: "rgba(150, 50, 50, 0.6)"}
            ];
            
            quadrants.forEach(q => {
                const div = document.createElement('div');
                div.id = q.id;
                div.style.position = 'absolute';
                div.style.left = q.left;
                div.style.top = q.top;
                div.style.width = '50%';
                div.style.height = '50%';
                div.style.background = q.bg;
                div.style.display = 'flex';
                div.style.alignItems = 'center';
                div.style.justifyContent = 'center';
                div.style.color = 'white';
                div.style.fontSize = '48px';
                div.style.fontWeight = 'bold';
                div.style.boxSizing = 'border-box';
                div.style.border = '2px solid rgba(255,255,255,0.2)';
                div.innerText = q.label;
                this.grid.appendChild(div);
            });
            this.marker.style.display = "none";
        } else if (this.state === "RESOLVED") {
            this.grid.innerHTML = "";
            // Keep background zoomed? Yes, maybe remove dimming? 
            if (event.x !== undefined && event.y !== undefined) {
                // Actually the marker is placed at the resolved position... but wait!
                // The resolved position is in original screen coordinates.
                // But the screen is currently displaying the ZOOM2 view, or is it back to IDLE?
                // The prompt says "No action is performed on it yet." and "A point selected in the magnified view maps back to the intended screen coordinate... resolve a precise screen coordinate"
                // Let's just go back to IDLE in a moment or show the marker on the zoomed view?
                // The prompt says "Eight seconds without input from any state returns to IDLE". 
                // If we show the marker, we should show it in original coordinates? 
                // If the screen is still zoomed, we need to map the original coordinates to the zoomed view to show it, or just unzoom and show it.
                // Let's unzoom and show it.
                this.bg.style.backgroundSize = "100% 100%";
                this.bg.style.backgroundPosition = "0 0";
                
                this.marker.style.left = event.x + 'px';
                this.marker.style.top = event.y + 'px';
                this.marker.style.display = "block";
            }
        }
    }

    onDwellProgress(event) {
        if (this.state === "GRID" && event.zone_id && event.zone_id.startsWith("cell_")) {
            // highlight cell
            Array.from(this.grid.children).forEach(el => el.classList.remove('active'));
            const cell = document.getElementById(event.zone_id);
            if (cell) cell.classList.add('active');
        } else if (this.state === "RADIAL" && event.zone_id && event.zone_id.startsWith("radial_")) {
            Array.from(this.grid.children).forEach(el => el.style.background = 'rgba(0,0,0,0.7)');
            const wedge = document.getElementById(event.zone_id);
            if (wedge) wedge.style.background = 'rgba(0,255,0,0.5)';
        } else if (this.state === "SYSTEM_MENU" && event.zone_id && event.zone_id.startsWith("sys_")) {
            Array.from(this.grid.children).forEach(el => el.style.border = '2px solid rgba(255,255,255,0.2)');
            const quad = document.getElementById(event.zone_id);
            if (quad) quad.style.border = '10px solid rgba(0,255,0,0.8)';
        }
    }
}
