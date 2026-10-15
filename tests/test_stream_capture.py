"""With body capture on, a streamed response sends every event whole.

``response_body`` is ``{"__costkey_stream__": [every event, in order]}`` so
the server can assemble the full answer. The caller's stream is unchanged.
"""
from __future__ import annotations

import asyncio
import binascii
import json
import struct
import zlib

import httpx
import pytest

import costkey.patch as patch_module
from conftest import capturing
from costkey.telemetry import STREAM_BODY_KEY
from costkey.transport import Transport

KEY = "__costkey_stream__"
assert STREAM_BODY_KEY == KEY


def chunked(module, body: bytes, size: int = 7):
    """A response body delivered in small pieces that split events and lines."""

    class Chunked(module.SyncByteStream, module.AsyncByteStream):
        def __init__(self) -> None:
            self.pieces = [body[i:i + size] for i in range(0, len(body), size)]

        def __iter__(self):
            yield from self.pieces

        async def __aiter__(self):
            for piece in self.pieces:
                yield piece

    return Chunked()


def sse(events: list[tuple[str | None, object]]) -> bytes:
    out = []
    for name, data in events:
        if name:
            out.append(f"event: {name}\n")
        out.append(f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n")
    return "".join(out).encode()


def only_call(sink):
    (call,) = sink.calls()
    return call


def wire(call) -> dict:
    return Transport("https://x.test", "k", 50, 5, False)._serialize(call)


RESPONSES_EVENTS = [
    ("response.created", {"type": "response.created", "sequence_number": 0,
                          "response": {"id": "resp_1", "object": "response", "status": "in_progress",
                                       "model": "gpt-5", "output": []}}),
    ("response.output_text.delta", {"type": "response.output_text.delta", "sequence_number": 1,
                                    "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": "Hel"}),
    ("response.output_text.delta", {"type": "response.output_text.delta", "sequence_number": 2,
                                    "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": "lo"}),
    ("response.completed", {"type": "response.completed", "sequence_number": 3,
                            "response": {"id": "resp_1", "object": "response", "status": "completed",
                                         "model": "gpt-5", "output": [{"type": "message", "id": "msg_1",
                                                                       "role": "assistant", "content": [
                                             {"type": "output_text", "text": "Hello", "annotations": []}]}],
                                         "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}}}),
]
RESPONSES_EXPECTED = [{"event": name, "data": data} for name, data in RESPONSES_EVENTS]


def _handler(module, body: bytes, content_type: str = "text/event-stream"):
    def handler(request):
        return module.Response(200, headers={"content-type": content_type}, stream=chunked(module, body))
    return handler


# ── OpenAI Responses API (the Agents SDK path) ──

@pytest.mark.parametrize("module_name", ["httpx", "httpx2"])
def test_responses_sse_with_event_names_sync(module_name):
    module = pytest.importorskip(module_name)
    body = sse(RESPONSES_EVENTS)
    seen = bytearray()
    with capturing(capture_body=True) as sink:
        with module.Client(transport=module.MockTransport(_handler(module, body))) as http:
            with http.stream("POST", "https://api.openai.com/v1/responses",
                             json={"model": "gpt-5", "input": "hi", "stream": True}) as response:
                for piece in response.iter_bytes():
                    seen += piece
    assert bytes(seen) == body
    call = only_call(sink)
    assert call.response_body == {KEY: RESPONSES_EXPECTED}
    assert wire(call)["responseBody"] == {KEY: RESPONSES_EXPECTED}
    # The telemetry skeleton is still sent alongside.
    assert call.telemetry["stream"]["chunkCount"] == 4


@pytest.mark.parametrize("module_name", ["httpx", "httpx2"])
def test_responses_sse_with_event_names_async(module_name):
    module = pytest.importorskip(module_name)
    body = sse(RESPONSES_EVENTS)
    seen: list[str] = []

    async def run():
        async with module.AsyncClient(transport=module.MockTransport(_handler(module, body))) as http:
            async with http.stream("POST", "https://api.openai.com/v1/responses",
                                   json={"model": "gpt-5", "input": "hi", "stream": True}) as response:
                async for line in response.aiter_lines():
                    seen.append(line)

    with capturing(capture_body=True) as sink:
        asyncio.run(run())
    assert "\n".join(seen) + "\n" == body.decode()
    assert only_call(sink).response_body == {KEY: RESPONSES_EXPECTED}


def test_openai_sdk_responses_stream_sends_every_event():
    openai = pytest.importorskip("openai")
    httpx2 = pytest.importorskip("httpx2")
    body = sse(RESPONSES_EVENTS)

    async def run_async():
        client = openai.AsyncOpenAI(api_key="sk-test", http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(_handler(httpx2, body))))
        stream = await client.responses.create(model="gpt-5", input="hi", stream=True)
        return [event.type async for event in stream]

    with capturing(capture_body=True) as sink:
        client = openai.OpenAI(api_key="sk-test", http_client=httpx2.Client(
            transport=httpx2.MockTransport(_handler(httpx2, body))))
        sync_types = [event.type for event in client.responses.create(model="gpt-5", input="hi", stream=True)]
        async_types = asyncio.run(run_async())

    expected_types = [name for name, _ in RESPONSES_EVENTS]
    assert sync_types == async_types == expected_types
    calls = sink.calls()
    assert len(calls) == 2
    for call in calls:
        assert call.streaming is True
        assert call.response_body == {KEY: RESPONSES_EXPECTED}


# ── Chat completions ──

def test_chat_completions_sse_drops_done():
    chunks = [
        {"id": "c1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Hi"}}]},
        {"id": "c1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"id": "c1", "object": "chat.completion.chunk", "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}},
    ]
    body = sse([(None, chunk) for chunk in chunks] + [(None, "[DONE]")])
    seen: list[str] = []
    with capturing(capture_body=True) as sink:
        with httpx.Client(transport=httpx.MockTransport(_handler(httpx, body))) as http:
            with http.stream("POST", "https://api.openai.com/v1/chat/completions",
                             json={"model": "gpt-4o", "messages": [], "stream": True}) as response:
                seen.extend(response.iter_lines())
    assert "\n".join(seen) + "\n" == body.decode()
    assert only_call(sink).response_body == {KEY: chunks}


def test_non_json_data_lines_are_not_events():
    body = sse([(None, {"a": 1}), ("ping", "keep-alive"), (None, {"b": 2}), (None, "[DONE]")])
    with capturing(capture_body=True) as sink:
        with httpx.Client(transport=httpx.MockTransport(_handler(httpx, body))) as http:
            with http.stream("POST", "https://api.openai.com/v1/chat/completions",
                             json={"stream": True}) as response:
                response.read()
    assert only_call(sink).response_body == {KEY: [{"a": 1}, {"b": 2}]}


# ── Anthropic ──

def test_anthropic_stream():
    events = [
        ("message_start", {"type": "message_start", "message": {"id": "msg_1", "model": "claude-sonnet-4-5",
                                                                 "usage": {"input_tokens": 12, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("ping", {"type": "ping"}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": "Hello"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                           "usage": {"output_tokens": 2}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    body = sse(events)
    seen = bytearray()

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(_handler(httpx, body))) as http:
            async with http.stream("POST", "https://api.anthropic.com/v1/messages",
                                   json={"model": "claude-sonnet-4-5", "messages": [], "stream": True}) as response:
                async for piece in response.aiter_bytes():
                    seen.extend(piece)

    with capturing(capture_body=True) as sink:
        asyncio.run(run())
    assert bytes(seen) == body
    assert only_call(sink).response_body == {KEY: [{"event": name, "data": data} for name, data in events]}


# ── Gemini ──

GEMINI_CHUNKS = [
    {"candidates": [{"content": {"parts": [{"text": "Hel"}], "role": "model"}}], "modelVersion": "gemini-2.5-flash"},
    {"candidates": [{"content": {"parts": [{"text": "lo"}], "role": "model"}, "finishReason": "STOP"}],
     "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2}, "modelVersion": "gemini-2.5-flash"},
]


def test_gemini_sse_stream():
    body = sse([(None, chunk) for chunk in GEMINI_CHUNKS])
    seen = bytearray()
    with capturing(capture_body=True) as sink:
        with httpx.Client(transport=httpx.MockTransport(_handler(httpx, body))) as http:
            with http.stream("POST", "https://generativelanguage.googleapis.com/v1beta/models/"
                             "gemini-2.5-flash:streamGenerateContent?alt=sse", json={"contents": []}) as response:
                for piece in response.iter_bytes():
                    seen += piece
    assert bytes(seen) == body
    assert only_call(sink).response_body == {KEY: GEMINI_CHUNKS}


def test_gemini_json_array_stream():
    body = json.dumps(GEMINI_CHUNKS).encode()
    with capturing(capture_body=True) as sink:
        with httpx.Client(transport=httpx.MockTransport(_handler(httpx, body, "application/json"))) as http:
            response = http.post("https://aiplatform.googleapis.com/v1/projects/p/locations/global/publishers/"
                                 "google/models/gemini-2.5-pro:streamGenerateContent", json={"contents": []})
    assert response.content == body
    assert only_call(sink).response_body == {KEY: GEMINI_CHUNKS}


# ── NDJSON ──

def test_ndjson_stream_one_event_per_line():
    lines = [{"model": "llama3", "message": {"content": "Hi"}, "done": False},
             {"model": "llama3", "done": True, "prompt_eval_count": 3, "eval_count": 1}]
    body = ("\n".join(json.dumps(line) for line in lines) + "\n").encode()
    from costkey.providers import apply_runtime_hosts, clear_runtime_hosts
    apply_runtime_hosts([{"id": "local", "hostname": "localhost"}])
    try:
        with capturing(capture_body=True) as sink:
            with httpx.Client(transport=httpx.MockTransport(_handler(httpx, body, "application/x-ndjson"))) as http:
                with http.stream("POST", "http://localhost:11434/api/chat", json={"model": "llama3"}) as response:
                    assert b"".join(response.iter_bytes()) == body
    finally:
        clear_runtime_hosts()
    assert only_call(sink).response_body == {KEY: lines}


# ── Bedrock ──

def _frame(headers: dict[str, str], payload: bytes) -> bytes:
    raw_headers = b""
    for name, value in headers.items():
        encoded = value.encode()
        raw_headers += bytes([len(name)]) + name.encode() + b"\x07" + struct.pack(">H", len(encoded)) + encoded
    total = 12 + len(raw_headers) + len(payload) + 4
    prelude = struct.pack(">II", total, len(raw_headers))
    prelude += struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF)
    message = prelude + raw_headers + payload
    return message + struct.pack(">I", binascii.crc32(message) & 0xFFFFFFFF)


def test_bedrock_eventstream_frames_over_httpx():
    import base64
    inner = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}
    body = (
        _frame({":event-type": "chunk", ":message-type": "event"},
               json.dumps({"bytes": base64.b64encode(json.dumps(inner).encode()).decode()}).encode())
        + _frame({":event-type": "metadata", ":message-type": "event"},
                 json.dumps({"usage": {"inputTokens": 3, "outputTokens": 1}}).encode())
    )
    with capturing(capture_body=True) as sink:
        with httpx.Client(transport=httpx.MockTransport(
                _handler(httpx, body, "application/vnd.amazon.eventstream"))) as http:
            with http.stream("POST", "https://bedrock-runtime.us-east-1.amazonaws.com/model/m/"
                             "invoke-with-response-stream", content=b"{}") as response:
                assert b"".join(response.iter_bytes()) == body
    assert only_call(sink).response_body == {KEY: [
        {"chunk": {"bytes": inner}},
        {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 1}}},
    ]}


def test_bedrock_botocore_stream_events():
    class _Meta:
        endpoint_url = "https://bedrock-runtime.us-east-1.amazonaws.com"

        class service_model:
            service_name = "bedrock-runtime"

    class _Client:
        meta = _Meta()

    events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Hi"}, "contentBlockIndex": 0}},
        {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 1, "totalTokens": 4}}},
    ]

    def original(client, operation, params):
        return {"stream": iter(events), "ResponseMetadata": {"HTTPStatusCode": 200, "HTTPHeaders": {}}}

    with capturing(capture_body=True) as sink:
        response = patch_module._bedrock_api_call(original, _Client(), "ConverseStream", "converse-stream",
                                                  {"modelId": "m", "messages": []})
        assert list(response["stream"]) == events
    assert only_call(sink).response_body == {KEY: events}


# ── Scrubbing and capture off ──

def test_stream_events_are_scrubbed():
    body = sse([(None, {"text": "ok", "api_key": "leaked", "echo": "sk-" + "a" * 30})])
    with capturing(capture_body=True) as sink:
        with httpx.Client(transport=httpx.MockTransport(_handler(httpx, body))) as http:
            with http.stream("POST", "https://api.openai.com/v1/chat/completions",
                             json={"stream": True}) as response:
                response.read()
    assert only_call(sink).response_body == {KEY: [{"text": "ok", "api_key": "[REDACTED]", "echo": "[REDACTED]"}]}


def test_large_streams_are_not_truncated():
    chunks = [{"choices": [{"delta": {"content": f"token {i} " * 50}}]} for i in range(2000)]
    body = sse([(None, chunk) for chunk in chunks])
    with capturing(capture_body=True) as sink:
        with httpx.Client(transport=httpx.MockTransport(_handler(httpx, body))) as http:
            with http.stream("POST", "https://api.openai.com/v1/chat/completions",
                             json={"stream": True}) as response:
                for _ in response.iter_lines():
                    pass
    call = only_call(sink)
    assert call.response_body == {KEY: chunks}
    assert call.telemetry["stream"]["chunkCount"] == 2000
    assert len(call.telemetry["stream"]["chunks"]) == 70  # the skeleton is still capped


def test_capture_off_sends_no_response_body():
    body = sse(RESPONSES_EVENTS)
    seen = bytearray()
    with capturing(capture_body=False) as sink:
        with httpx.Client(transport=httpx.MockTransport(_handler(httpx, body))) as http:
            with http.stream("POST", "https://api.openai.com/v1/responses",
                             json={"model": "gpt-5", "input": "hi", "stream": True}) as response:
                for piece in response.iter_bytes():
                    seen += piece
    assert bytes(seen) == body
    call = only_call(sink)
    assert call.response_body is None
    assert wire(call)["responseBody"] is None
    assert call.telemetry["stream"]["chunkCount"] == 4
    assert "Hel" not in json.dumps(call.telemetry)


# ── Streams the app stops reading ──

def _lines_then_stop(module, stop_after: str | None):
    """Read a Responses stream line by line, stopping after the line that contains ``stop_after``."""
    body = sse(RESPONSES_EVENTS)
    with capturing(capture_body=True) as sink:
        with module.Client(transport=module.MockTransport(_handler(module, body))) as http:
            with http.stream("POST", "https://api.openai.com/v1/responses",
                             json={"model": "gpt-5", "input": "hi", "stream": True}) as response:
                for line in response.iter_lines():
                    if stop_after and stop_after in line:
                        break
    return only_call(sink)


@pytest.mark.parametrize("module_name", ["httpx", "httpx2"])
def test_a_stream_closed_before_its_end_is_reported_as_closed_early(module_name):
    module = pytest.importorskip(module_name)
    call = _lines_then_stop(module, '"response.created"')
    sent = wire(call)
    # The raw fact, as Python sees it; the server maps it to a cancelled call.
    assert sent["telemetry"]["error"] == {"name": "GeneratorExit", "code": None}
    assert call.response_body == {KEY: RESPONSES_EXPECTED[:1]}


@pytest.mark.parametrize("module_name", ["httpx", "httpx2"])
def test_a_stream_closed_right_after_its_last_event_finished(module_name):
    module = pytest.importorskip(module_name)
    call = _lines_then_stop(module, '"response.completed"')
    assert wire(call)["telemetry"]["error"] is None
    assert call.response_body == {KEY: RESPONSES_EXPECTED}


@pytest.mark.parametrize("module_name", ["httpx", "httpx2"])
def test_a_stream_read_to_the_end_has_no_error(module_name):
    module = pytest.importorskip(module_name)
    call = _lines_then_stop(module, None)
    assert wire(call)["telemetry"]["error"] is None


def test_a_closed_early_stream_lasts_until_its_last_data_not_its_cleanup():
    httpx = pytest.importorskip("httpx")
    import time as _time

    body = sse(RESPONSES_EVENTS)
    with capturing(capture_body=True) as sink:
        http = httpx.Client(transport=httpx.MockTransport(_handler(httpx, body)))
        response = http.send(http.build_request("POST", "https://api.openai.com/v1/responses",
                                                json={"model": "gpt-5", "input": "hi", "stream": True}), stream=True)
        lines = response.iter_lines()
        next(lines)                       # the app reads one line and moves on
        _time.sleep(0.3)                  # the connection is cleaned up later
        lines.close()
        response.close()
        http.close()
    assert wire(only_call(sink))["durationMs"] < 250
