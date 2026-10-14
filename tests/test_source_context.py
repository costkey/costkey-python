"""Source context on in-app call-site frames."""
from __future__ import annotations

import os

import httpx

from costkey import stack
from costkey.stack import add_source_context, capture_call_site, is_in_app
from costkey.transport import _frame_dict
from costkey.types import StackFrame


def _make_the_call():
    marker = "before"
    site = capture_call_site()  # THE CALL LINE
    marker = "after"
    return site, marker


def test_in_app_frame_gets_the_call_line_and_three_lines_around_it():
    site, _ = _make_the_call()
    frame = next(f for f in site.frames if f.function_name == "_make_the_call")
    assert frame.context_line == "    site = capture_call_site()  # THE CALL LINE"
    assert frame.pre_context == ['', 'def _make_the_call():', '    marker = "before"']
    assert frame.post_context == ['    marker = "after"', '    return site, marker', '']


def test_library_stdlib_and_costkey_frames_get_no_context():
    assert not is_in_app(httpx.__file__)
    assert not is_in_app(os.__file__)
    assert not is_in_app(stack.__file__)
    assert not is_in_app("<frozen importlib._bootstrap>")
    assert not is_in_app("<string>")
    assert not is_in_app("/app/.venv/lib/python3.12/site-packages/openai/x.py")
    assert not is_in_app("/usr/lib/python3/dist-packages/foo.py")
    assert is_in_app(__file__)


def test_only_the_first_eight_in_app_frames_get_context():
    frames = [StackFrame(function_name=f"f{i}", file_name=__file__, line_number=1) for i in range(12)]
    add_source_context(frames)
    filled = [f for f in frames if f.context_line is not None]
    assert len(filled) == 8
    assert frames[8].context_line is None


def test_library_frames_do_not_count_toward_the_cap():
    lib = StackFrame(function_name="lib", file_name=httpx.__file__, line_number=1)
    frames = [lib] * 10 + [StackFrame(function_name="mine", file_name=__file__, line_number=1)]
    add_source_context(frames)
    assert frames[-1].context_line is not None
    assert lib.context_line is None


def test_long_lines_are_capped_and_newlines_stripped(tmp_path):
    path = tmp_path / "long.py"
    path.write_text("a = 1\r\n" + "x" * 2000 + "\n" + "b = 2\n")
    frame = StackFrame(file_name=str(path), line_number=2)
    add_source_context([frame])
    assert frame.context_line == "x" * 500
    assert frame.pre_context == ["a = 1"]
    assert frame.post_context == ["b = 2"]


def test_unreadable_files_and_bad_lines_are_skipped_silently(tmp_path):
    missing = StackFrame(file_name=str(tmp_path / "nope.py"), line_number=3)
    past_end = StackFrame(file_name=__file__, line_number=10_000_000)
    no_line = StackFrame(file_name=__file__, line_number=None)
    add_source_context([missing, past_end, no_line])
    assert missing.context_line is None
    assert past_end.context_line is None
    assert no_line.context_line is None


def test_wire_format_includes_context_only_when_present():
    bare = _frame_dict(StackFrame(function_name="f", file_name="/a.py", line_number=1))
    assert "contextLine" not in bare
    full = _frame_dict(StackFrame(function_name="f", file_name="/a.py", line_number=2,
                                  context_line="x()", pre_context=["a"], post_context=[]))
    assert full["contextLine"] == "x()"
    assert full["preContext"] == ["a"]
    assert full["postContext"] == []


def test_secret_shapes_in_source_lines_are_redacted():
    from costkey.stack import scrub_source_line as scrub
    assert scrub('client = OpenAI(api_key="sk-abcdefghijklmnopqrstuvwxyz")') == 'client = OpenAI(api_key="[REDACTED]")'
    assert scrub("PASSWORD = 'hunter2'") == "PASSWORD = '[REDACTED]'"
    assert scrub('{"token": "abc"}') == '{"token": "[REDACTED]"}'
    assert scrub('url = "https://u:p@host/path?key=s&x=1"') == 'url = "https://[REDACTED]@host/path?[REDACTED]"'
    assert scrub('h = {"Authorization": f"Bearer {tok}"}') == 'h = {"Authorization": f"Bearer {tok}"}'
    assert scrub("headers['x'] = 'Bearer abcdefghijklmnopqrstuvwxyz'") == "headers['x'] = 'Bearer [REDACTED]'"
    assert scrub('requests.post(f"{base}?key=abc&x=1")') == 'requests.post(f"{base}?key=[REDACTED]&x=1")'
    assert scrub("resp = client.chat.completions.create(model=m)") == "resp = client.chat.completions.create(model=m)"
    assert scrub('requests.get("https://api.example.com/v1/x")') == 'requests.get("https://api.example.com/v1/x")'
