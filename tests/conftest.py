"""Shared fixtures: capture events without a network, with every transport patched."""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import costkey.patch as patch_module


class Sink:
    def __init__(self) -> None:
        self.events = []

    def enqueue(self, event):
        self.events.append(event)

    def calls(self):
        return [event for event in self.events if event.event_type == "call"]

    def spans(self, status=None):
        return [
            event for event in self.events
            if event.event_type == "span" and (status is None or event.span_status == status)
        ]

    def ended_spans(self):
        return [event for event in self.spans() if event.span_status != "running"]

    def span_names(self):
        """span id → "kind:name" for every span seen."""
        return {event.id: f"{event.span_kind}:{event.span_name}" for event in self.spans()}


@contextmanager
def capturing(*, capture_body: bool = False):
    state = patch_module._state
    saved = {key: getattr(state, key) for key in (
        "transport", "project_id", "before_send", "before_send_event", "debug", "admission",
        "admission_mode", "capture_body", "local_capture_body", "config_known", "config_status",
        "default_context",
    )}
    sink = Sink()
    state.transport = sink
    state.project_id = "project-test"
    state.before_send = None
    state.before_send_event = None
    state.debug = False
    state.admission = None
    state.admission_mode = "monitor"
    state.capture_body = capture_body
    state.local_capture_body = capture_body
    state.config_known = True
    state.config_status = "available"
    state.default_context = {}
    patch_module._patch_httpx()
    patch_module._patch_requests()
    patch_module._patch_aiohttp()
    patch_module._patch_botocore()
    state.patched = True
    try:
        yield sink
    finally:
        patch_module._unpatch_httpx()
        patch_module._unpatch_requests()
        patch_module._unpatch_aiohttp()
        patch_module._unpatch_botocore()
        state.patched = False
        for key, value in saved.items():
            setattr(state, key, value)


@pytest.fixture()
def sink():
    with capturing() as captured:
        yield captured


@pytest.fixture()
def content_sink():
    with capturing(capture_body=True) as captured:
        yield captured
