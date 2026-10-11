"""Call-site capture: every frame of the stack, raw.

The SDK does not decide which frames are "user code"; the server filters.
Frames are innermost first.
"""
from __future__ import annotations

import sys
import traceback

from costkey.types import CallSite, StackFrame


def capture_call_site() -> CallSite | None:
    try:
        summary = traceback.extract_stack(sys._getframe(1))
    except Exception:
        return None
    frames = [
        StackFrame(function_name=frame.name, file_name=frame.filename, line_number=frame.lineno)
        for frame in reversed(summary)
    ]
    if not frames:
        return None
    return CallSite(raw="".join(traceback.format_list(summary)), frames=frames)
