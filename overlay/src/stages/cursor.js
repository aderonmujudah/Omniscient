export class CursorStage {
    constructor(element) {
        this.element = element;
        this.isVisible = false;
    }
    
    update(x, y) {
        if (!this.isVisible) {
            this.element.style.display = 'block';
            this.isVisible = true;
        }
        this.element.style.left = `${x}px`;
        this.element.style.top = `${y}px`;
    }
    
    hide() {
        this.element.style.display = 'none';
        this.isVisible = false;
    }
}
