import numpy as np
import ctypes
from ctypes import wintypes
from engine.capture.base import ScreenCapture

# Windows API structures and functions
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

SRCCOPY = 0x00CC0020
CAPTUREBLT = 0x40000000

class WindowsCapture(ScreenCapture):
    """
    Windows screen capture using GDI BitBlt.
    Captures the primary monitor. With WDA_EXCLUDEFROMCAPTURE set on the
    overlay window, it will be excluded from this capture automatically by DWM.
    """
    
    def __init__(self):
        self.screen_width = user32.GetSystemMetrics(0)
        self.screen_height = user32.GetSystemMetrics(1)
        
        self.hwnd = user32.GetDesktopWindow()
        self.hdc_screen = user32.GetWindowDC(self.hwnd)
        self.hdc_mem = gdi32.CreateCompatibleDC(self.hdc_screen)
        self.hbitmap = gdi32.CreateCompatibleBitmap(self.hdc_screen, self.screen_width, self.screen_height)
        gdi32.SelectObject(self.hdc_mem, self.hbitmap)

    def capture(self) -> np.ndarray:
        # BitBlt from screen to memory DC. CAPTUREBLT flag captures layered windows (if not excluded).
        gdi32.BitBlt(self.hdc_mem, 0, 0, self.screen_width, self.screen_height, 
                     self.hdc_screen, 0, 0, SRCCOPY | CAPTUREBLT)
                     
        # Extract bitmap data
        bmp_info = dict(
            bmiHeader=dict(
                biSize=40,
                biWidth=self.screen_width,
                biHeight=-self.screen_height, # top-down
                biPlanes=1,
                biBitCount=32,
                biCompression=0,
                biSizeImage=0,
                biXPelsPerMeter=0,
                biYPelsPerMeter=0,
                biClrUsed=0,
                biClrImportant=0
            )
        )
        
        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)
            ]
            
        class BITMAPINFO(ctypes.Structure):
            _fields_ = [
                ("bmiHeader", BITMAPINFOHEADER),
                ("bmiColors", wintypes.DWORD * 3)
            ]
            
        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = self.screen_width
        bmi.bmiHeader.biHeight = -self.screen_height
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = 0
        
        buffer = ctypes.create_string_buffer(self.screen_width * self.screen_height * 4)
        gdi32.GetDIBits(self.hdc_mem, self.hbitmap, 0, self.screen_height, buffer, ctypes.byref(bmi), 0)
        
        img = np.frombuffer(buffer, dtype=np.uint8).reshape((self.screen_height, self.screen_width, 4))
        # Drop alpha channel
        return img[:, :, :3]

    def close(self) -> None:
        if self.hdc_mem:
            gdi32.DeleteDC(self.hdc_mem)
            self.hdc_mem = None
        if self.hbitmap:
            gdi32.DeleteObject(self.hbitmap)
            self.hbitmap = None
        if self.hdc_screen:
            user32.ReleaseDC(self.hwnd, self.hdc_screen)
            self.hdc_screen = None
