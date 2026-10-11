"""The raw telemetry contract (v1): skeleton rules, caps, headers, URLs, streams."""
from __future__ import annotations

import base64
import binascii
import json
import struct
import zlib

import httpx
import pytest

from costkey import telemetry as t


# ── Skeleton ──

def test_strings_are_lengths_unless_their_key_is_kept():
    body = {
        "model": "gpt-4o",
        "id": "chatcmpl-1",
        "speech_model": "best",
        "response_id": "resp_1",
        "role": "user",
        "content": "héllo wörld 🌍",
        "prompt": "x" * 10,
    }
    assert t.skeleton(body) == {
        "model": "gpt-4o",
        "id": "chatcmpl-1",
        "speech_model": "best",
        "response_id": "resp_1",
        "role": {"$s": 4},
        "content": {"$s": 13},  # Unicode code points, not bytes
        "prompt": {"$s": 10},
    }


def test_every_keep_key_from_the_contract():
    expected = {
        "model", "model_id", "modelId", "modelVersion", "id", "object", "type", "status",
        "finish_reason", "stop_reason", "finishReason", "service_tier", "system_fingerprint",
        "responseId", "request_id", "requestId", "search_depth", "topic", "output_format",
        "response_format", "encoding", "language", "language_code", "voice_id", "size",
        "quality", "style", "mode", "operation", "event", "provider", "modality", "tier",
        "currency", "unit", "job_id", "jobId", "trafficType", "created", "created_at",
        "speech_model", "response_id",
    }
    assert t.KEEP_STRING_KEYS == expected


def test_kept_strings_over_200_chars_become_lengths():
    assert t.skeleton({"model": "m" * 200}) == {"model": "m" * 200}
    assert t.skeleton({"model": "m" * 201}) == {"model": {"$s": 201}}


def test_scalars_are_verbatim_and_non_finite_numbers_are_null():
    assert t.skeleton({"a": 1, "b": 2.5, "c": True, "d": None, "e": float("nan"), "f": float("inf")}) == {
        "a": 1, "b": 2.5, "c": True, "d": None, "e": None, "f": None,
    }


def test_strings_inside_arrays_have_no_key_and_always_become_lengths():
    assert t.skeleton({"model": ["gpt-4o", "x"]}) == {"model": [{"$s": 6}, {"$s": 1}]}


def test_arrays_keep_32_items_then_the_first_8_and_a_count():
    assert t.skeleton(list(range(32))) == list(range(32))
    assert t.skeleton(list(range(33))) == list(range(8)) + [{"$more": 25}]
    embedding = {"data": [{"embedding": [0.1] * 1536, "index": 0, "object": "embedding"}]}
    assert t.skeleton(embedding) == {
        "data": [{"embedding": [0.1] * 8 + [{"$more": 1528}], "index": 0, "object": "embedding"}],
    }


def test_root_is_depth_0_and_a_container_at_depth_12_is_truncated():
    value: dict = {"leaf": 1}
    for _ in range(12):
        value = {"x": value}
    # Containers at depths 0..11 survive; the one at depth 12 is replaced.
    out = t.skeleton(value)
    node = out
    for _ in range(12):
        node = node["x"]
    assert node == {"$trunc": True}
    # Scalars at depth 12 are kept.
    shallow: dict = {"leaf": 1}
    for _ in range(11):
        shallow = {"x": shallow}
    node = t.skeleton(shallow)
    for _ in range(11):
        node = node["x"]
    assert node == {"leaf": 1}


def test_arrays_count_toward_depth():
    value: list = [1]
    for _ in range(12):
        value = [value]
    node = t.skeleton(value)
    for _ in range(12):
        node = node[0]
    assert node == {"$trunc": True}


def test_bytes_become_byte_counts():
    assert t.skeleton({"image": b"\x00" * 10}) == {"image": {"$bytes": 10}}


def test_size_cap_truncates_the_deepest_level_first_until_it_fits():
    wide = {f"k{i}": {"inner": {"a": 1234567, "b": 1234567, "c": 1234567}} for i in range(1800)}
    assert t.serialized_size(t._skel(wide, None, 0)) > t.MAX_SKELETON_BYTES
    out = t.skeleton(wide)
    assert t.serialized_size(out) <= t.MAX_SKELETON_BYTES
    # The deepest level went first, everywhere at once; one level was enough.
    assert out["k0"] == {"inner": {"$trunc": True}}
    assert out["k1799"] == {"inner": {"$trunc": True}}


def test_size_cap_can_truncate_the_root():
    flat = {f"key{i}": i for i in range(20000)}
    assert t.skeleton(flat) == {"$trunc": True}


def test_framework_objects_are_turned_into_data_without_interpretation():
    from dataclasses import dataclass

    @dataclass
    class Row:
        provider: str
        model: str
        input_tokens: int
        note: str

    assert t.skeleton(Row("deepgram", "nova-3", 5, "hello")) == {
        "provider": "deepgram", "model": "nova-3", "input_tokens": 5, "note": {"$s": 5},
    }


def test_attribute_skeleton_also_matches_the_last_dotted_segment():
    attrs = {
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.prompt": "secret",
        "gen_ai.operation.name": "chat",
        "gen_ai.tool.call.id": "call_1",
        "gen_ai.usage.input_tokens": 12,
    }
    assert t.attribute_skeleton(attrs) == {
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.prompt": {"$s": 6},
        "gen_ai.operation.name": {"$s": 4},
        "gen_ai.tool.call.id": "call_1",
        "gen_ai.usage.input_tokens": 12,
    }


# ── URLs ──

@pytest.mark.parametrize("url,expected", [
    (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini:streamGenerateContent?alt=sse&key=AIzaSECRET",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini:streamGenerateContent?alt=sse",
    ),
    (
        "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&token=t&Authorization=x",
        "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true",
    ),
    (
        "https://user:pass@x.openai.azure.com/openai/deployments/d/chat/completions?api-version=2024-10-21&api-key=k#frag",
        "https://x.openai.azure.com/openai/deployments/d/chat/completions?api-version=2024-10-21",
    ),
    (
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/m/invoke?X-Amz-Signature=s&x-amz-date=d&sig=1&signature=2&code=3&apikey=4&access_token=5&API_KEY=6&auth_x=7&keep=me",
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/m/invoke?keep=me",
    ),
    ("http://[::1]:8080/v1?model=a%20b", "http://[::1]:8080/v1?model=a%20b"),
    ("not a url", ""),
])
def test_query_strings_keep_everything_but_credentials(url, expected):
    assert t.sanitize_url(url) == expected


# ── Headers ──

def test_request_headers_are_an_allowlist():
    headers = httpx.Headers({
        "Content-Type": "application/json",
        "Anthropic-Version": "2023-06-01",
        "anthropic-beta": "tools-2024",
        "OpenAI-Beta": "assistants=v2",
        "x-goog-api-client": "genai-py/1",
        "X-Stainless-Lang": "python",
        "Authorization": "Bearer sk-secret",
        "x-api-key": "secret",
        "user-agent": "OpenAI/Python",
    })
    assert t.request_headers(headers) == {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "tools-2024",
        "openai-beta": "assistants=v2",
        "x-goog-api-client": "genai-py/1",
        "x-stainless-lang": "python",
    }


def test_response_headers_are_a_denylist_that_keeps_token_counts():
    headers = [
        ("Set-Cookie", "a=b"), ("Cookie", "c"), ("Authorization", "x"), ("Proxy-Authorization", "y"),
        ("X-API-Key", "k"), ("api-key", "k"), ("x-session-token", "t"),
        ("X-Request-Id", "req_1"), ("x-amzn-bedrock-input-token-count", "120"),
        ("anthropic-ratelimit-tokens-remaining", "1000"), ("openai-processing-ms", "40"),
        ("character-cost", "12"),
    ]
    assert t.response_headers(headers) == {
        "x-request-id": "req_1",
        "x-amzn-bedrock-input-token-count": "120",
        "openai-processing-ms": "40",
        "character-cost": "12",
    }


# ── Bodies ──

def test_body_rules():
    assert t.body_skeleton(b"") is None
    assert t.body_skeleton(None) is None
    assert t.body_skeleton(b"\xff\xd8\xff", "image/jpeg") == {"$bytes": 3}
    assert t.body_skeleton(b"RIFF....", "audio/wav") == {"$bytes": 8}
    assert t.body_skeleton(b"\x00\x01", "application/octet-stream") == {"$bytes": 2}
    assert t.body_skeleton(b"plain words", "text/plain") == {"$s": 11}
    assert t.body_skeleton(b"\xff\xfe\x00", "") == {"$bytes": 3}
    assert t.body_skeleton(b'{"model":"m","input":"hi"}', "application/json") == {
        "model": "m", "input": {"$s": 2},
    }
    assert t.body_skeleton(b"model=m&text=hello", "application/x-www-form-urlencoded") == {
        "model": "m", "text": {"$s": 5},
    }


def test_multipart_bytes_name_fields_and_count_files():
    body = (
        b"--BOUND\r\n"
        b'Content-Disposition: form-data; name="model"\r\n\r\n'
        b"whisper-1\r\n"
        b"--BOUND\r\n"
        b'Content-Disposition: form-data; name="prompt"\r\n\r\n'
        b"spell it right\r\n"
        b"--BOUND\r\n"
        b'Content-Disposition: form-data; name="file"; filename="meeting.wav"\r\n'
        b"Content-Type: audio/wav\r\n\r\n"
        + b"\x00" * 100 + b"\r\n"
        b"--BOUND--\r\n"
    )
    assert t.body_skeleton(body, "multipart/form-data; boundary=BOUND") == {"$multipart": [
        {"name": "model", "value": "whisper-1"},
        {"name": "prompt", "value": {"$s": 14}},
        {"name": "file", "filename": {"$s": 11}, "contentType": "audio/wav", "bytes": 100},
    ]}


def test_multipart_from_an_unread_httpx_stream():
    request = httpx.Request(
        "POST", "https://api.openai.com/v1/audio/transcriptions",
        data={"model": "whisper-1", "language": "en"},
        files={"file": ("a.mp3", b"\x01" * 42, "audio/mpeg")},
    )
    assert t.multipart_from_httpx(request.stream) == {"$multipart": [
        {"name": "model", "value": "whisper-1"},
        {"name": "language", "value": "en"},
        {"name": "file", "filename": {"$s": 5}, "contentType": "audio/mpeg", "bytes": 42},
    ]}


def test_multipart_from_aiohttp_form_data():
    aiohttp = pytest.importorskip("aiohttp")
    import io
    form = aiohttp.FormData()
    form.add_field("model_id", "scribe_v1")
    form.add_field("file", io.BytesIO(b"\x00" * 7), filename="x.wav", content_type="audio/wav")
    assert t.multipart_from_aiohttp(form) == {"$multipart": [
        {"name": "model_id", "value": "scribe_v1"},
        {"name": "file", "filename": {"$s": 5}, "contentType": "audio/wav", "bytes": 7},
    ]}
    plain = aiohttp.FormData({"model": "m", "text": "hi"})
    assert t.multipart_from_aiohttp(plain) == {"model": "m", "text": {"$s": 2}}


def test_error_block_is_the_raw_class_name_and_errno():
    try:
        raise ConnectionResetError(54, "reset")
    except ConnectionResetError as exc:
        error = exc
    import errno
    assert t.error_block(error) == {"name": "ConnectionResetError", "code": errno.errorcode[54]}
    assert t.error_block(httpx.ReadTimeout("slow")) == {"name": "ReadTimeout", "code": None}


# ── Streams ──

def _sse(*records: str) -> bytes:
    return "".join(f"data: {record}\n\n" for record in records).encode()


def test_sse_skips_done_and_event_lines_and_keeps_non_json_as_lengths():
    collector = t.StreamCollector("text/event-stream")
    collector.feed(b"event: message_start\n")
    collector.feed(b'data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":5}}}\n\n')
    collector.feed(b": keep-alive comment\n\n")
    collector.feed(b"data: not json\n\n")
    collector.feed(b"data: [DONE]\n\n")
    stream, body = collector.finish()
    assert body is None
    assert stream["format"] == "sse"
    assert stream["chunks"] == [
        {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 5}}},
        {"$s": 8},
    ]
    assert stream["chunkCount"] == 2
    assert stream["firstChunkMs"] is not None and stream["lastChunkMs"] >= stream["firstChunkMs"]


def test_sse_records_split_across_network_reads():
    collector = t.StreamCollector("text/event-stream")
    data = _sse('{"id":"a"}', '{"id":"b"}')
    for index in range(0, len(data), 3):
        collector.feed(data[index:index + 3])
    stream, _ = collector.finish()
    assert stream["chunks"] == [{"id": "a"}, {"id": "b"}]


def test_stream_chunk_caps_keep_first_20_and_last_50_over_200():
    collector = t.StreamCollector("text/event-stream")
    collector.feed(_sse(*[json.dumps({"index": i}) for i in range(200)]))
    stream, _ = collector.finish()
    assert stream["chunkCount"] == 200 and len(stream["chunks"]) == 200

    collector = t.StreamCollector("text/event-stream")
    collector.feed(_sse(*[json.dumps({"index": i}) for i in range(500)]))
    stream, _ = collector.finish()
    assert stream["chunkCount"] == 500
    assert [chunk["index"] for chunk in stream["chunks"]] == list(range(20)) + list(range(450, 500))


def test_ndjson_and_json_array_streams():
    collector = t.StreamCollector("application/x-ndjson")
    collector.feed(b'{"model":"llama3","done":false}\n{"model":"llama3","done":true,"eval_count":9}\n')
    stream, _ = collector.finish()
    assert stream["format"] == "ndjson"
    assert stream["chunks"][1] == {"model": "llama3", "done": True, "eval_count": 9}

    collector = t.StreamCollector("application/json")
    collector.feed(b'[{"usageMetadata":{"promptTokenCount":3}},')
    collector.feed(b'{"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":4}}]')
    stream, _ = collector.finish()
    assert stream["format"] == "json-array"
    assert stream["chunkCount"] == 2


def test_a_json_object_on_a_stream_is_a_response_body():
    collector = t.StreamCollector("application/json")
    collector.feed(b'{"error":{"type":"invalid_request_error","message":"bad"}}')
    stream, body = collector.finish()
    assert stream is None
    assert body == {"error": {"type": "invalid_request_error", "message": {"$s": 3}}}


def test_binary_streams_count_bytes_or_use_content_length():
    collector = t.StreamCollector("audio/mpeg")
    collector.feed(b"\x00" * 10)
    collector.feed(b"\x00" * 5)
    assert collector.finish() == (None, {"$bytes": 15})
    unread = t.StreamCollector("audio/mpeg", content_length=4096)
    assert unread.finish() == (None, {"$bytes": 4096})


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


def test_bedrock_event_stream_frames_decode_base64_bytes():
    inner = json.dumps({"type": "message_delta", "usage": {"output_tokens": 7}}).encode()
    chunk = _frame(
        {":event-type": "chunk", ":message-type": "event", ":content-type": "application/json"},
        json.dumps({"bytes": base64.b64encode(inner).decode(), "p": "abcd"}).encode(),
    )
    metadata = _frame(
        {":event-type": "metadata", ":message-type": "event"},
        json.dumps({"usage": {"inputTokens": 3, "outputTokens": 7}, "metrics": {"latencyMs": 20}}).encode(),
    )
    collector = t.StreamCollector("application/vnd.amazon.eventstream")
    data = chunk + metadata
    collector.feed(data[:10])
    collector.feed(data[10:])
    stream, _ = collector.finish()
    assert stream["format"] == "aws-eventstream"
    assert stream["chunks"] == [
        {"chunk": {"bytes": {"type": "message_delta", "usage": {"output_tokens": 7}}, "p": {"$s": 4}}},
        {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 7}, "metrics": {"latencyMs": 20}}},
    ]


def test_build_assembles_the_contract_shape():
    data = t.build(
        method="POST", url="https://api.openai.com/v1/chat/completions?key=x",
        request_header_values={"content-type": "application/json", "authorization": "Bearer x"},
        request_body={"model": "m"}, request_bytes=13, status=200,
        response_header_values={"x-request-id": "r", "set-cookie": "c"},
        response_body={"id": "x"}, response_bytes=10, stream=None, error=None,
        started_at="2026-10-11T09:00:00.000Z", duration_ms=12.3456,
    )
    assert data == {
        "v": 1,
        "request": {
            "method": "POST", "url": "https://api.openai.com/v1/chat/completions",
            "headers": {"content-type": "application/json"}, "body": {"model": "m"}, "bytes": 13,
        },
        "response": {"status": 200, "headers": {"x-request-id": "r"}, "body": {"id": "x"}, "bytes": 10},
        "stream": None,
        "error": None,
        "startedAt": "2026-10-11T09:00:00.000Z",
        "durationMs": 12.35,
    }
