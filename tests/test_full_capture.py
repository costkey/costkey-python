"""With body capture on, content is sent whole; the transport splits by bytes."""
from __future__ import annotations

import json

import httpx
import pytest

import costkey
import costkey.patch as patch_module
import costkey.transport as transport_module
from conftest import capturing
from costkey import telemetry
from costkey.transport import MAX_BATCH_BYTES, Transport


def test_span_input_and_output_over_2000_chars_are_sent_whole():
    question = "Which funds hold INF1? " * 250  # 5,750 chars
    answer = {"rows": ["fund %d holds it" % i for i in range(1000)]}
    with capturing(capture_body=True) as sink:
        with costkey.tool("lookup", input=question) as span:
            span.set_output(answer)
    (started,) = sink.spans(status="running")
    (ended,) = sink.ended_spans()
    assert started.attributes["input"] == question
    assert ended.attributes["output"] == json.dumps(answer)
    assert len(ended.attributes["output"]) > 2000


def test_span_input_is_still_dropped_when_capture_is_off():
    with capturing(capture_body=False) as sink:
        with costkey.tool("lookup", input="q" * 5000) as span:
            span.set_output("a" * 5000)
    for event in sink.spans():
        assert "input" not in (event.attributes or {})
        assert "output" not in (event.attributes or {})


def test_framework_attributes_over_64kb_are_sent_whole_when_capture_is_on():
    raw = {
        "gen_ai.prompt": "Summarise this filing. " * 5000,
        "messages": [{"role": "user", "content": {"deep": {"deeper": "a long turn " * 100}}} for _ in range(200)],
    }
    assert telemetry.serialized_size(raw) > telemetry.MAX_SKELETON_BYTES
    with capturing(capture_body=True):
        attrs = patch_module.framework_attributes("crewai", raw)
    assert attrs == {"framework": "crewai", "crewai": raw}


def test_framework_attributes_stay_a_skeleton_when_capture_is_off():
    with capturing(capture_body=False):
        attrs = patch_module.framework_attributes("crewai", {"gen_ai.prompt": "secret"})
    assert attrs == {"framework": "crewai", "crewai": {"gen_ai.prompt": {"$s": 6}}}


def test_otel_attributes_over_64kb_are_sent_whole_when_capture_is_on():
    pytest.importorskip("opentelemetry.sdk")
    from costkey.integrations.opentelemetry import otel_attributes

    prompt = "Summarise this filing. " * 5000
    raw = {"gen_ai.prompt": prompt, "gen_ai.request.model": "gpt-4o"}
    assert telemetry.serialized_size(raw) > telemetry.MAX_SKELETON_BYTES
    assert otel_attributes(raw, capture=True) == raw
    assert otel_attributes(raw, capture=False)["gen_ai.prompt"] == {"$s": len(prompt)}


def test_request_and_response_bodies_over_1mb_are_sent_whole():
    prompt = "word " * (300 * 1024)  # 1.5 MB
    reply = "token " * (250 * 1024)  # 1.5 MB
    assert len(prompt) > 1024 * 1024 and len(reply) > 1024 * 1024

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "id": "chatcmpl_1", "model": "gpt-4o",
            "choices": [{"message": {"role": "assistant", "content": reply}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

    with capturing(capture_body=True) as sink:
        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            http.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}]},
            )
    (call,) = sink.calls()
    assert call.request_body["messages"][0]["content"] == prompt
    assert call.response_body["choices"][0]["message"]["content"] == reply
    wire = Transport("https://x.test", "k", 50, 5, False)._serialize(call)
    assert wire["requestBody"]["messages"][0]["content"] == prompt


def test_multipart_form_fields_over_4kb_are_captured_whole():
    long_prompt = "a cinematic shot of a lighthouse, " * 300
    body = (
        b"--b\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\n"
        + long_prompt.encode() + b"\r\n--b--\r\n"
    )
    assert patch_module._form_fields(body, "multipart/form-data; boundary=b") == {"prompt": long_prompt}


class _Recorder:
    def __init__(self, statuses=None):
        self.bodies: list[bytes] = []
        self.statuses = list(statuses or [])

    def __call__(self, url, *, content, headers, timeout):
        assert headers["Content-Type"] == "application/json"
        self.bodies.append(content)
        return httpx.Response(self.statuses.pop(0) if self.statuses else 202)

    def batches(self):
        return [[event["id"] for event in json.loads(body)["events"]] for body in self.bodies]


def _event(event_id: str, size: int) -> dict:
    return {"id": event_id, "eventType": "call", "requestBody": {"text": "x" * size}}


def test_transport_splits_a_large_queue_by_bytes(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(transport_module.httpx, "post", recorder)
    transport = Transport("https://x.test/api/v1/events", "k", 50, 5, False)
    three_mb = 3 * 1024 * 1024
    transport._queue = [_event(f"e{i}", three_mb) for i in range(5)] + [_event(f"s{i}", 10) for i in range(3)]

    transport.flush()

    assert recorder.batches() == [["e0", "e1"], ["e2", "e3"], ["e4", "s0", "s1", "s2"]]
    assert all(len(body) <= MAX_BATCH_BYTES for body in recorder.bodies)
    assert transport._queue == []


def test_an_event_over_the_byte_limit_goes_alone(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(transport_module.httpx, "post", recorder)
    transport = Transport("https://x.test/api/v1/events", "k", 50, 5, False)
    transport._queue = [_event("small", 10), _event("huge", MAX_BATCH_BYTES + 1024), _event("after", 10)]

    transport.flush()

    assert recorder.batches() == [["small"], ["huge"], ["after"]]
    huge = json.loads(recorder.bodies[1])["events"][0]
    assert len(huge["requestBody"]["text"]) == MAX_BATCH_BYTES + 1024


def test_byte_batches_still_respect_max_batch_size(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(transport_module.httpx, "post", recorder)
    transport = Transport("https://x.test/api/v1/events", "k", 2, 5, False)
    transport._queue = [_event(f"e{i}", 10) for i in range(5)]

    transport.flush()

    assert recorder.batches() == [["e0", "e1"], ["e2", "e3"], ["e4"]]


def test_a_failed_byte_batch_is_requeued_in_order(monkeypatch):
    recorder = _Recorder(statuses=[503])
    monkeypatch.setattr(transport_module.httpx, "post", recorder)
    transport = Transport("https://x.test/api/v1/events", "k", 50, 5, False)
    three_mb = 3 * 1024 * 1024
    transport._queue = [_event(f"e{i}", three_mb) for i in range(3)]

    transport.flush()  # the first POST fails; flushing stops and backs off

    assert recorder.batches() == [["e0", "e1"]]
    assert [event["id"] for event in transport._queue] == ["e0", "e1", "e2"]
    assert transport._consecutive_failures == 1

    transport.flush()  # an explicit flush ignores the backoff

    assert recorder.batches()[1:] == [["e0", "e1"], ["e2"]]
    assert transport._queue == []
    assert transport._consecutive_failures == 0
