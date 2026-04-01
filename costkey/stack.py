"""Stack trace capture — auto-attribute AI calls to code."""
from __future__ import annotations
import traceback
from costkey.types import CallSite, StackFrame

_INTERNAL = ("costkey/", "site-packages/costkey")


def capture_call_site() -> CallSite | None:
    raw = traceback.format_stack()
    frames: list[StackFrame] = []

    for line in reversed(raw):
        line = line.strip()
        if not line.startswith("File "):
            continue
        # Parse: File "path", line N, in func
        parts = line.split(", ")
        if len(parts) < 3:
            continue

        file_name = parts[0].replace('File "', "").rstrip('"')
        if any(p in file_name for p in _INTERNAL):
            continue

        line_num = None
        func_name = None
        for p in parts[1:]:
            if p.startswith("line "):
                try:
                    line_num = int(p.replace("line ", ""))
                except ValueError:
                    pass
            elif p.startswith("in "):
                func_name = p.replace("in ", "").strip()

        frames.append(StackFrame(
            function_name=func_name,
            file_name=file_name,
            line_number=line_num,
        ))

    if not frames:
        return None
    return CallSite(raw="".join(raw), frames=frames)
