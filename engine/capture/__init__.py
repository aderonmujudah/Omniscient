import sys
from engine.capture.base import ScreenCapture
from engine.capture.null import NullCapture

if sys.platform == 'win32':
    from engine.capture.windows import WindowsCapture
