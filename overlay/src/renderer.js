import { EngineClient } from './transport/client.js';
import { CursorStage } from './stages/cursor.js';

const cursor = new CursorStage(document.getElementById('cursor'));

const engine = new EngineClient('ws://127.0.0.1:8765');
engine.addEventListener((event) => {
    if (event.event_type === 'GAZE_MOVE') {
        cursor.update(event.x, event.y);
    }
});
engine.connect();
