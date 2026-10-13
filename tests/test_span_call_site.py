"""A span sends where it was opened, raw; the server attributes calls to it."""
from __future__ import annotations

import costkey
import costkey.patch as patch


def test_a_span_sends_the_stack_where_it_was_opened(monkeypatch):
    sent = []
    monkeypatch.setattr(patch, "_dispatch_event", sent.append)

    def run_dash():
        with costkey.agent("dash"):
            pass

    run_dash()
    spans = [event for event in sent if event.event_type == "span"]
    assert spans, "the span is sent"
    for event in spans:
        assert event.call_site is not None
        names = [frame.function_name for frame in event.call_site.frames]
        assert "run_dash" in names
        opened = next(frame for frame in event.call_site.frames if frame.function_name == "run_dash")
        assert opened.context_line and "costkey.agent" in opened.context_line
