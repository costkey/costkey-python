"""Calls a framework makes on its own task are attributed to the code that opened the scope."""
from __future__ import annotations

import asyncio
import os

import costkey
from costkey.stack import capture_call_site, has_own_code
from costkey.types import StackFrame


def _library_call():
    # Stands in for a model call made inside a framework: its own stack is
    # only "library" frames once the in-app filter is applied.
    return capture_call_site()


def test_entry_point_alone_is_not_own_code():
    assert not has_own_code([StackFrame(function_name="run", file_name="/app/svc/__main__.py", line_number=8)])
    assert has_own_code([StackFrame(function_name="answer", file_name="/app/svc/chat.py", line_number=3)])


def test_a_call_with_own_code_keeps_its_own_stack():
    with costkey.agent("dash"):
        site = capture_call_site()
    assert site is not None
    assert site.frames[0].function_name == "test_a_call_with_own_code_keeps_its_own_stack"


def test_a_call_from_a_framework_task_uses_where_the_agent_was_opened(monkeypatch):
    import costkey.stack as stack

    def run_dash():
        with costkey.agent("dash"):
            # The framework runs the model call on its own task: a copy of this context.
            async def framework_task():
                monkeypatch.setattr(stack, "has_own_code", lambda frames, real=stack.has_own_code: real(frames) and not any(f.function_name == "framework_task" for f in frames))
                return _library_call()
            return asyncio.run(framework_task())

    site = run_dash()
    assert site is not None
    names = [frame.function_name for frame in site.frames]
    assert "run_dash" in names, names
    opened = next(frame for frame in site.frames if frame.function_name == "run_dash")
    assert opened.context_line and "costkey.agent" in opened.context_line
    assert os.path.basename(opened.file_name or "") == "test_scope_site.py"


def test_the_scope_site_is_forgotten_when_the_scope_closes():
    import costkey.stack as stack
    with costkey.agent("dash"):
        assert stack._scope_site.get() is not None
    assert stack._scope_site.get() is None
