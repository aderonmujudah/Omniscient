import ctypes
from ctypes import wintypes
from typing import List
from engine.input.base import BaseInputBackend, MouseInput, KeyboardInput

user32 = ctypes.windll.user32

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))
    ]

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))
    ]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD)
    ]

class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT)
    ]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", INPUT_UNION)
    ]

class WindowsInput(BaseInputBackend):
    """
    Windows input backend using SendInput.
    """
    def __init__(self):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2) # PROCESS_PER_MONITOR_DPI_AWARE
        except AttributeError:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except AttributeError:
                pass
                
        self.screen_width = user32.GetSystemMetrics(0)
        self.screen_height = user32.GetSystemMetrics(1)

    def _inject_mouse(self, inputs: List[MouseInput]) -> None:
        c_inputs = (INPUT * len(inputs))()
        for i, inp in enumerate(inputs):
            # Normalize absolute coordinates for SendInput (0 to 65535)
            # SendInput expects coords to map to the whole virtual desktop or primary monitor.
            # Using absolute screen coordinates.
            nx = int((inp.x / self.screen_width) * 65535)
            ny = int((inp.y / self.screen_height) * 65535)
            
            c_inputs[i].type = INPUT_MOUSE
            c_inputs[i].union.mi.dx = nx
            c_inputs[i].union.mi.dy = ny
            c_inputs[i].union.mi.mouseData = ctypes.c_uint32(inp.data).value
            c_inputs[i].union.mi.dwFlags = inp.flags
            c_inputs[i].union.mi.time = 0
            c_inputs[i].union.mi.dwExtraInfo = None
            
        ret = user32.SendInput(len(c_inputs), ctypes.byref(c_inputs), ctypes.sizeof(INPUT))
        if ret != len(c_inputs):
            import ctypes
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"SendInput failed. Expected {len(c_inputs)}, got {ret}. Error: {ctypes.GetLastError()}")        
    def _inject_keyboard(self, inputs: List[KeyboardInput]) -> None:
        c_inputs = (INPUT * len(inputs))()
        for i, inp in enumerate(inputs):
            c_inputs[i].type = INPUT_KEYBOARD
            c_inputs[i].union.ki.wVk = inp.wVk
            c_inputs[i].union.ki.wScan = inp.wScan
            c_inputs[i].union.ki.dwFlags = inp.flags
            c_inputs[i].union.ki.time = 0
            c_inputs[i].union.ki.dwExtraInfo = None
            
        user32.SendInput(len(c_inputs), ctypes.byref(c_inputs), ctypes.sizeof(INPUT))
