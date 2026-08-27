import sys
from engine.input.base import InputBackend
from engine.input.null import NullInput

if sys.platform == 'win32':
    from engine.input.windows import WindowsInput
