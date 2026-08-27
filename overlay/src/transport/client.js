export class EngineClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.listeners = [];
    }
    
    connect() {
        this.ws = new WebSocket(this.url);
        
        this.ws.onmessage = (message) => {
            try {
                const data = JSON.parse(message.data);
                if (data.event_type) {
                    this.listeners.forEach(fn => fn(data));
                }
            } catch (e) {
                console.error("Error decoding message", e);
            }
        };
        
        this.ws.onclose = () => {
            setTimeout(() => this.connect(), 1000);
        };
    }
    
    addEventListener(fn) {
        this.listeners.push(fn);
    }
    
    sendAction(actionDict) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(actionDict));
        }
    }
}
