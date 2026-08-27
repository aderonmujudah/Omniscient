import { EngineClient } from './transport/client.js';
import { CursorStage } from './stages/cursor.js';
import { TargetStage } from './stages/target_stage.js';

const cursor = new CursorStage(document.getElementById('cursor'));
const targetStage = new TargetStage();

const dwellRing = document.getElementById('dwell-progress');
const dwellFill = dwellRing.querySelector('.dwell-fill');

const engine = new EngineClient('ws://127.0.0.1:8765');
engine.addEventListener((event) => {
    if (event.event_type === 'GAZE_MOVE') {
        cursor.update(event.x, event.y);
    } else if (event.event_type === 'STATE_CHANGE') {
        const ind = document.getElementById("status-indicator");
        if (ind) {
            if (event.is_paused) {
                ind.style.display = "block";
            } else {
                ind.style.display = "none";
            }
        }
        targetStage.onStateChange(event);
        dwellRing.style.display = "none";
    } else if (event.event_type === 'DWELL_START') {
        dwellRing.style.display = "block";
        dwellRing.style.left = event.x + 'px';
        dwellRing.style.top = event.y + 'px';
        dwellFill.style.clipPath = `polygon(50% 50%, 50% 0%, 50% 0%, 50% 0%, 50% 0%, 50% 0%, 50% 0%)`;
    } else if (event.event_type === 'DWELL_PROGRESS') {
        targetStage.onDwellProgress(event);
        if (dwellRing.style.display === "block") {
            const pct = event.progress * 100;
            // Simple approach for pie slice, clip-path is tricky for dynamic circles
            // Let's just use a conic-gradient for the fill instead of clip-path
            dwellFill.style.background = `conic-gradient(#00ff00 ${pct}%, transparent ${pct}%)`;
            dwellFill.style.border = "none"; // since we use background
        }
    } else if (event.event_type === 'DWELL_COMPLETE' || event.event_type === 'DWELL_CANCEL') {
        dwellRing.style.display = "none";
    }
});
engine.connect();
