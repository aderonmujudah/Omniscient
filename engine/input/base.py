from typing import Protocol, Optional, List, Tuple
from dataclasses import dataclass

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

@dataclass
class MouseInput:
    x: float
    y: float
    flags: int
    data: int = 0

@dataclass
class KeyboardInput:
    wScan: int
    wVk: int
    flags: int

class InputBackend(Protocol):
    """
    Interface for synthesizing OS input events.
    """
    def click(self, x: float, y: float, button: str = "left", count: int = 1) -> None:
        pass
        
    def drag(self, phase: str, x: float, y: float) -> None:
        pass
        
    def scroll(self, x: float, y: float, dy: float) -> None:
        pass
        
    def key(self, text: Optional[str] = None, keycode: Optional[str] = None) -> None:
        pass

class BaseInputBackend(InputBackend):
    def _inject_mouse(self, inputs: List[MouseInput]) -> None:
        raise NotImplementedError
        
    def _inject_keyboard(self, inputs: List[KeyboardInput]) -> None:
        raise NotImplementedError

    def _get_button_flags(self, button: str) -> Tuple[int, int]:
        if button == "right":
            return MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP
        elif button == "middle":
            return MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP
        return MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP

    def click(self, x: float, y: float, button: str = "left", count: int = 1) -> None:
        down_flag, up_flag = self._get_button_flags(button)
        inputs = [MouseInput(x, y, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)]
        for _ in range(count):
            inputs.append(MouseInput(x, y, down_flag | MOUSEEVENTF_ABSOLUTE))
            inputs.append(MouseInput(x, y, up_flag | MOUSEEVENTF_ABSOLUTE))
        self._inject_mouse(inputs)
        
    def drag(self, phase: str, x: float, y: float) -> None:
        down_flag, up_flag = self._get_button_flags("left")
        if phase == "start":
            inputs = [
                MouseInput(x, y, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE),
                MouseInput(x, y, down_flag | MOUSEEVENTF_ABSOLUTE)
            ]
        elif phase == "end":
            inputs = [
                MouseInput(x, y, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE),
                MouseInput(x, y, up_flag | MOUSEEVENTF_ABSOLUTE)
            ]
        else:
            return
        self._inject_mouse(inputs)
        
    def scroll(self, x: float, y: float, dy: float) -> None:
        inputs = [
            MouseInput(x, y, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE),
            MouseInput(x, y, MOUSEEVENTF_WHEEL, data=int(dy))
        ]
        self._inject_mouse(inputs)
        
    def key(self, text: Optional[str] = None, keycode: Optional[str] = None) -> None:
        inputs = []
        if text:
            for char in text:
                wScan = ord(char)
                inputs.append(KeyboardInput(wScan, 0, KEYEVENTF_UNICODE))
                inputs.append(KeyboardInput(wScan, 0, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
        elif keycode:
            # Map known keycodes (like "backspace", "enter") to VK constants
            vk = self._map_keycode(keycode)
            if vk is not None:
                inputs.append(KeyboardInput(0, vk, 0))
                inputs.append(KeyboardInput(0, vk, KEYEVENTF_KEYUP))
        if inputs:
            self._inject_keyboard(inputs)
            
    def _map_keycode(self, keycode: str) -> Optional[int]:
        mapping = {
            "backspace": 0x08,
            "enter": 0x0D,
            "tab": 0x09,
            "escape": 0x1B,
            "space": 0x20
        }
        return mapping.get(keycode.lower())
