"""Screen capture: images sent to the model are listed on the call and uploaded once each."""
from __future__ import annotations

import base64
import hashlib
from contextlib import contextmanager

import httpx

import costkey.patch as patch_module
import costkey.transport as transport_module
from conftest import capturing
from costkey.frames import MAX_FRAME_BYTES, extract_frames
from costkey.transport import Transport, frames_endpoint

PNG_HEADER = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


def png(fill: int, size: int = 200) -> tuple[str, str, int]:
    """A fake PNG: the real signature plus filler. Returns (base64, sha256, byte count)."""
    raw = PNG_HEADER + bytes([fill]) * size
    return base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest(), len(raw)


def anthropic_part(data: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


# ── extraction ──

def test_finds_images_in_each_provider_shape_in_order():
    a, b, c, d = png(1), png(2), png(3), png(4)
    frames = extract_frames({
        "messages": [
            {"role": "user", "content": [anthropic_part(a[0])]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b[0]}"}}]},
        ],
        "input": [{"type": "computer_call_output", "output": {"type": "input_image", "image_url": f"data:image/png;base64,{c[0]}"}}],
        "contents": [{"parts": [{"inline_data": {"mime_type": "image/png", "data": d[0]}}]}],
    })
    assert [ref["sha256"] for ref in frames.refs] == [a[1], b[1], c[1], d[1]]
    assert frames.refs[0] == {"sha256": a[1], "mediaType": "image/png", "bytes": a[2]}
    assert base64.b64encode(frames.uploads[a[1]][1]).decode() == a[0]


def test_sniffs_undeclared_image_parts_and_ignores_other_media():
    a = png(5)
    frames = extract_frames({"parts": [
        {"type": "image", "data": a[0]},
        {"type": "input_audio", "input_audio": {"data": base64.b64encode(bytes([7]) * 300).decode(), "format": "wav"}},
        {"data": base64.b64encode(bytes([8]) * 300).decode(), "mime_type": "audio/wav"},
    ]})
    assert frames.refs == [{"sha256": a[1], "mediaType": "image/png", "bytes": a[2]}]


def test_lists_a_history_image_each_time_but_uploads_it_once():
    a, b = png(9), png(10)
    frames = extract_frames({"messages": [
        {"content": [anthropic_part(a[0])]}, {"content": [anthropic_part(a[0])]}, {"content": [anthropic_part(b[0])]},
    ]})
    assert [ref["sha256"] for ref in frames.refs] == [a[1], a[1], b[1]]
    assert len(frames.uploads) == 2


def test_lists_an_image_over_the_cap_without_uploading_it():
    big = png(11, MAX_FRAME_BYTES + 1)
    frames = extract_frames({"image_url": f"data:image/png;base64,{big[0]}"})
    assert frames.refs == [{"sha256": big[1], "mediaType": "image/png", "bytes": big[2], "omitted": "too_large"}]
    assert frames.uploads == {}


# ── through the patch ──

@contextmanager
def screen_capture(*, screens: bool, capture_body: bool = False, before_send=None):
    with capturing(capture_body=capture_body) as sink:
        uploads: list[tuple[str, str, bytes]] = []
        sink.enqueue_frame = lambda sha256, media_type, data: uploads.append((sha256, media_type, data))
        state = patch_module._state
        saved = (state.capture_screens, state.before_send)
        state.capture_screens = screens
        state.before_send = before_send
        try:
            yield sink, uploads
        finally:
            state.capture_screens, state.before_send = saved


def call_anthropic(body: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "msg_1", "model": "claude-sonnet-4-5", "usage": {"input_tokens": 1, "output_tokens": 1}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        http.post("https://api.anthropic.com/v1/messages", json=body)


def screenshot_request(data: str) -> dict:
    return {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [anthropic_part(data)]}]}],
    }


def test_lists_request_images_on_the_call_and_queues_their_bytes():
    shot = png(20)
    with screen_capture(screens=True) as (sink, uploads):
        call_anthropic(screenshot_request(shot[0]))
    (call,) = sink.calls()
    assert call.frames == [{"sha256": shot[1], "mediaType": "image/png", "bytes": shot[2]}]
    assert call.request_body is None
    assert [(sha, media_type) for sha, media_type, _ in uploads] == [(shot[1], "image/png")]
    wire = Transport("https://x.test/api/v1/events", "k", 50, 5, False)._serialize(call)
    assert wire["frames"] == call.frames
    assert shot[0] not in str(wire["telemetry"])


def test_names_the_frame_in_a_captured_body():
    shot = png(21)
    with screen_capture(screens=True, capture_body=True) as (sink, _uploads):
        call_anthropic(screenshot_request(shot[0]))
    (call,) = sink.calls()
    body = str(call.request_body)
    assert f"[image data omitted: image/png; ~{shot[2]} bytes; frame {shot[1]}]" in body
    assert shot[0] not in body


def test_sends_no_frames_when_screen_capture_is_off():
    shot = png(22)
    with screen_capture(screens=False, capture_body=True) as (sink, uploads):
        call_anthropic(screenshot_request(shot[0]))
    (call,) = sink.calls()
    assert call.frames is None
    assert uploads == []
    assert "; frame " not in str(call.request_body)


def test_uploads_only_frames_the_hook_leaves():
    shot = png(23)

    def drop_frames(event):
        event.frames = []
        return event

    with screen_capture(screens=True, before_send=drop_frames) as (sink, uploads):
        call_anthropic(screenshot_request(shot[0]))
    (call,) = sink.calls()
    assert call.frames == []
    assert uploads == []


def test_strips_a_frame_list_a_hook_adds_while_off():
    def add_frames(event):
        event.frames = [{"sha256": "a" * 64, "mediaType": "image/png", "bytes": 1}]
        return event

    with screen_capture(screens=False, before_send=add_frames) as (sink, uploads):
        call_anthropic({"model": "claude-sonnet-4-5", "messages": []})
    (call,) = sink.calls()
    assert call.frames is None
    assert uploads == []


# ── settings ──

def test_runtime_config_needs_both_the_app_and_the_project():
    state = patch_module._state
    saved = (state.project_id, state.local_capture_screens, state.capture_screens, state.local_capture_body, state.capture_body)
    try:
        state.project_id = "p"
        config = {"projectId": "p", "capture": {"mode": "metadata", "bodyCaptureEnabled": False, "redactionEnabled": True}}
        for local, remote, expected in [(True, True, True), (True, None, False), (False, True, False), (True, False, False)]:
            state.local_capture_screens = local
            capture = dict(config["capture"])
            if remote is not None:
                capture["screenCaptureEnabled"] = remote
            assert patch_module.apply_runtime_config({**config, "capture": capture}, project_id="p")
            assert state.capture_screens is expected, (local, remote)
        state.local_capture_screens = True
        state.capture_screens = True
        patch_module._mark_runtime_config("unavailable", project_id="p")
        assert state.capture_screens is False
    finally:
        (state.project_id, state.local_capture_screens, state.capture_screens, state.local_capture_body, state.capture_body) = saved


# ── transport ──

class FakeHttp:
    def __init__(self, monkeypatch, status=201):
        self.requests: list[tuple[str, dict, bytes]] = []
        self.status = status
        self.error: Exception | None = None
        monkeypatch.setattr(transport_module.httpx, "put", self.put)

    def put(self, url, *, content, headers, timeout):
        self.requests.append((url, headers, content))
        if self.error:
            raise self.error
        return httpx.Response(self.status)


def new_transport() -> Transport:
    return Transport("https://costkey.dev/api/v1/events", "ck_test", 50, 60, False)


def test_frames_endpoint_sits_next_to_the_events_endpoint():
    assert frames_endpoint("https://costkey.dev/api/v1/events") == "https://costkey.dev/api/v1/frames"
    assert frames_endpoint("https://self.host/ck/api/v1/events/") == "https://self.host/ck/api/v1/frames"


def test_puts_each_image_once_keyed_by_hash(monkeypatch):
    http = FakeHttp(monkeypatch)
    t = new_transport()
    shot = png(30)
    raw = base64.b64decode(shot[0])
    t.enqueue_frame(shot[1], "image/png", raw)
    t.enqueue_frame(shot[1], "image/png", raw)
    t.flush()
    t.enqueue_frame(shot[1], "image/png", raw)
    t.flush()
    assert len(http.requests) == 1
    url, headers, content = http.requests[0]
    assert url == f"https://costkey.dev/api/v1/frames/{shot[1]}"
    assert headers["Content-Type"] == "image/png"
    assert headers["Authorization"] == "Bearer ck_test"
    assert content == raw


def test_stops_uploading_when_the_server_does_not_take_frames(monkeypatch):
    http = FakeHttp(monkeypatch, status=404)
    t = new_transport()
    for fill in (31, 32):
        shot = png(fill)
        t.enqueue_frame(shot[1], "image/png", base64.b64decode(shot[0]))
        t.flush()
    assert len(http.requests) == 1


def test_retries_a_rate_limited_frame_on_the_next_flush(monkeypatch):
    http = FakeHttp(monkeypatch, status=429)
    t = new_transport()
    shot = png(33)
    t.enqueue_frame(shot[1], "image/png", base64.b64decode(shot[0]))
    t.flush()
    http.status = 201
    t.flush()
    assert [url for url, _, _ in http.requests] == [f"https://costkey.dev/api/v1/frames/{shot[1]}"] * 2


def test_forgets_a_frame_that_failed_so_a_resend_uploads_it(monkeypatch):
    http = FakeHttp(monkeypatch)
    http.error = httpx.ConnectError("offline")
    t = new_transport()
    shot = png(34)
    raw = base64.b64decode(shot[0])
    t.enqueue_frame(shot[1], "image/png", raw)
    t.flush()
    http.error = None
    t.enqueue_frame(shot[1], "image/png", raw)
    t.flush()
    assert len(http.requests) == 2
