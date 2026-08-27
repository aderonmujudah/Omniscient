from typing import List
from engine.input.base import BaseInputBackend, MouseInput, KeyboardInput

class NullInput(BaseInputBackend):
    """
    Headless input backend for tests.
    Records intended hardware-level injections without making OS calls.
    """
    
    def __init__(self):
        self.mouse_injections: List[List[MouseInput]] = []
        self.keyboard_injections: List[List[KeyboardInput]] = []
        
    def _inject_mouse(self, inputs: List[MouseInput]) -> None:
        self.mouse_injections.append(inputs)
        
    def _inject_keyboard(self, inputs: List[KeyboardInput]) -> None:
        self.keyboard_injections.append(inputs)
