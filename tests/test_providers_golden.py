"""Golden raw telemetry per provider shape, through the real patched transports."""
from __future__ import annotations

import asyncio
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
import requests

import costkey.patch as patch_module
from costkey.providers import apply_runtime_hosts, clear_runtime_hosts, match_provider
from costkey.transport import Transport

TIMING = ("startedAt", "durationMs")


def golden(telemetry: dict) -> dict:
    """Telemetry without the clock-dependent fields."""
    data = {key: value for key, value in telemetry.items() if key not in TIMING}
    if data.get("stream"):
        data["stream"] = {k: v for k, v in data["stream"].items() if k not in ("firstChunkMs", "lastChunkMs")}
    if data.get("response"):
        headers = dict(data["response"]["headers"])
        headers.pop("content-length", None)
        data["response"] = {**data["response"], "headers": headers}
    return data


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def only_call(sink) -> dict:
    calls = sink.calls()
    assert len(calls) == 1, [c.url for c in calls]
    return calls[0]


# ── OpenAI ──

def test_openai_chat_completion(sink):
    def handler(request):
        return httpx.Response(200, headers={"x-request-id": "req_9", "openai-processing-ms": "41"}, json={
            "id": "chatcmpl-1", "object": "chat.completion", "created": 1760000000, "model": "gpt-4o-2024-08-06",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello there"},
                         "finish_reason": "stop", "logprobs": None}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12,
                      "prompt_tokens_details": {"cached_tokens": 0}},
            "service_tier": "default", "system_fingerprint": "fp_1",
        })

    with mock_client(handler) as http:
        http.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": "Bearer sk-secretsecretsecretsecret", "OpenAI-Beta": "x"},
            json={"model": "gpt-4o", "messages": [{"role": "system", "content": "be brief"},
                                                  {"role": "user", "content": "hi"}], "max_tokens": 50},
        )
    call = only_call(sink)
    assert golden(call.telemetry) == {
        "v": 1,
        "request": {
            "method": "POST",
            "url": "https://api.openai.com/v1/chat/completions",
            "headers": {"content-type": "application/json", "openai-beta": "x"},
            "body": {
                "model": "gpt-4o",
                "messages": [{"role": {"$s": 6}, "content": {"$s": 8}}, {"role": {"$s": 4}, "content": {"$s": 2}}],
                "max_tokens": 50,
            },
            "bytes": len(json.dumps({"model": "gpt-4o", "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}], "max_tokens": 50}, separators=(",", ":"))),
        },
        "response": {
            "status": 200,
            "headers": {"x-request-id": "req_9", "openai-processing-ms": "41", "content-type": "application/json"},
            "body": {
                "id": "chatcmpl-1", "object": "chat.completion", "created": 1760000000, "model": "gpt-4o-2024-08-06",
                "choices": [{"index": 0, "message": {"role": {"$s": 9}, "content": {"$s": 11}},
                             "finish_reason": "stop", "logprobs": None}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12,
                          "prompt_tokens_details": {"cached_tokens": 0}},
                "service_tier": "default", "system_fingerprint": "fp_1",
            },
            "bytes": call.telemetry["response"]["bytes"],
        },
        "stream": None,
        "error": None,
    }
    assert call.telemetry["response"]["bytes"] > 0
    assert call.telemetry["startedAt"].endswith("Z")
    assert call.telemetry["durationMs"] >= 0


def test_legacy_fields_carry_no_derived_values(sink):
    with mock_client(lambda r: httpx.Response(200, json={"model": "gpt-4o", "usage": {"prompt_tokens": 1},
                                                          "id": "chatcmpl-9"})) as http:
        http.post("https://api.openai.com/v1/chat/completions?api-version=1", json={"model": "gpt-4o"})
    wire = Transport("https://x.test/api/v1/events", "k", 50, 5, False)._serialize(only_call(sink))
    assert wire["model"] is None
    assert wire["usage"] is None
    assert wire["costUsd"] is None
    assert wire["streamTiming"] is None
    assert wire["systemPrompt"] is None
    assert wire["metadata"] is None
    assert "providerRequestId" not in wire and "requestModel" not in wire and "chargeId" not in wire
    assert wire["provider"] == "openai"
    assert wire["url"] == "https://api.openai.com/v1/chat/completions"
    assert wire["telemetry"]["request"]["url"] == "https://api.openai.com/v1/chat/completions?api-version=1"


def test_openai_chat_stream_skips_done(sink):
    chunks = [
        {"id": "c1", "object": "chat.completion.chunk", "model": "gpt-4o",
         "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}]},
        {"id": "c1", "object": "chat.completion.chunk", "model": "gpt-4o",
         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"id": "c1", "object": "chat.completion.chunk", "model": "gpt-4o", "choices": [],
         "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}},
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    with mock_client(handler) as http:
        with http.stream("POST", "https://api.openai.com/v1/chat/completions",
                         json={"model": "gpt-4o", "stream": True, "stream_options": {"include_usage": True}}) as response:
            for _ in response.iter_lines():
                pass
    call = only_call(sink)
    data = golden(call.telemetry)
    assert data["response"]["body"] is None
    assert data["response"]["bytes"] == len(body.encode())
    assert data["stream"] == {
        "format": "sse",
        "chunks": [
            {"id": "c1", "object": "chat.completion.chunk", "model": "gpt-4o",
             "choices": [{"index": 0, "delta": {"role": {"$s": 9}, "content": {"$s": 2}}, "finish_reason": None}]},
            {"id": "c1", "object": "chat.completion.chunk", "model": "gpt-4o",
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"id": "c1", "object": "chat.completion.chunk", "model": "gpt-4o", "choices": [],
             "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}},
        ],
        "chunkCount": 3,
    }
    assert call.streaming is True
    stream = call.telemetry["stream"]
    assert 0 <= stream["firstChunkMs"] <= stream["lastChunkMs"]


def test_openai_transcription_multipart_has_null_bytes(sink):
    with mock_client(lambda r: httpx.Response(200, json={"text": "hello", "usage": {"type": "duration", "seconds": 3}})) as http:
        http.post("https://api.openai.com/v1/audio/transcriptions",
                  data={"model": "whisper-1", "response_format": "json"},
                  files={"file": ("meeting.mp3", b"\x00" * 256, "audio/mpeg")})
    request = only_call(sink).telemetry["request"]
    assert request["bytes"] is None
    assert request["body"] == {"$multipart": [
        {"name": "model", "value": "whisper-1"},
        {"name": "response_format", "value": "json"},
        {"name": "file", "filename": {"$s": 11}, "contentType": "audio/mpeg", "bytes": 256},
    ]}
    assert only_call(sink).telemetry["response"]["body"] == {
        "text": {"$s": 5}, "usage": {"type": "duration", "seconds": 3},
    }


def test_openai_speech_binary_response(sink):
    audio = b"ID3" + b"\x00" * 1000
    with mock_client(lambda r: httpx.Response(200, headers={"content-type": "audio/mpeg"}, content=audio)) as http:
        http.post("https://api.openai.com/v1/audio/speech", json={"model": "tts-1", "voice": "alloy", "input": "Say hi"})
    telemetry = only_call(sink).telemetry
    assert telemetry["request"]["body"] == {"model": "tts-1", "voice": {"$s": 5}, "input": {"$s": 6}}
    assert telemetry["response"]["body"] == {"$bytes": len(audio)}
    assert telemetry["stream"] is None


def test_openai_responses_api_keeps_response_id(sink):
    events = [
        {"type": "response.created", "response": {"id": "resp_1", "object": "response", "status": "in_progress"}},
        {"type": "response.output_text.delta", "item_id": "msg_1", "response_id": "resp_1", "delta": "Hi"},
        {"type": "response.completed", "response": {"id": "resp_1", "status": "completed",
                                                    "usage": {"input_tokens": 5, "output_tokens": 1}}},
    ]
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    with mock_client(lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())) as http:
        http.post("https://api.openai.com/v1/responses", json={"model": "gpt-5", "input": "hi", "stream": True})
    stream = only_call(sink).telemetry["stream"]
    assert stream["chunks"][1] == {"type": "response.output_text.delta", "item_id": {"$s": 5},
                                   "response_id": "resp_1", "delta": {"$s": 2}}
    assert stream["chunks"][2]["response"]["usage"] == {"input_tokens": 5, "output_tokens": 1}


# ── Anthropic ──

def test_anthropic_stream_ignores_event_lines(sink):
    body = (
        "event: message_start\n"
        'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5","usage":{"input_tokens":25,"cache_read_input_tokens":100,"output_tokens":1}}}\n\n'
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n'
        "event: message_delta\n"
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":12}}\n\n'
        "event: message_stop\n"
        'data: {"type":"message_stop"}\n\n'
    )

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream", "request-id": "req_a"}, content=body.encode())

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            async with http.stream("POST", "https://api.anthropic.com/v1/messages",
                                   headers={"anthropic-version": "2023-06-01", "x-api-key": "sk-ant-secret"},
                                   json={"model": "claude-sonnet-4-5", "max_tokens": 64, "stream": True,
                                         "system": "be brief", "messages": [{"role": "user", "content": "hi"}]}) as response:
                async for _ in response.aiter_lines():
                    pass

    asyncio.run(run())
    data = golden(only_call(sink).telemetry)
    assert data["request"]["headers"] == {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    assert data["request"]["body"]["system"] == {"$s": 8}
    assert data["response"]["headers"]["request-id"] == "req_a"
    assert data["stream"] == {
        "format": "sse",
        "chunks": [
            {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": {"$s": 9},
                                                  "model": "claude-sonnet-4-5",
                                                  "usage": {"input_tokens": 25, "cache_read_input_tokens": 100,
                                                            "output_tokens": 1}}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": {"$s": 5}}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 12}},
            {"type": "message_stop"},
        ],
        "chunkCount": 4,
    }
    assert "sk-ant-secret" not in json.dumps(only_call(sink).telemetry)


# ── Google ──

def test_gemini_query_key_is_dropped_and_alt_sse_kept(sink):
    body = 'data: {"candidates":[{"content":{"parts":[{"text":"Hi"}],"role":"model"}}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1},"modelVersion":"gemini-2.5-flash","responseId":"r1"}\n\n'
    with mock_client(lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())) as http:
        http.post("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=AIzaSECRET",
                  json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]})
    data = golden(only_call(sink).telemetry)
    assert data["request"]["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    )
    assert data["stream"]["chunks"] == [{
        "candidates": [{"content": {"parts": [{"text": {"$s": 2}}], "role": {"$s": 5}}}],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1},
        "modelVersion": "gemini-2.5-flash", "responseId": "r1",
    }]
    assert "AIzaSECRET" not in json.dumps(only_call(sink).telemetry)


def test_gemini_json_array_stream(sink):
    body = json.dumps([
        {"candidates": [{"content": {"parts": [{"text": "a"}]}}], "usageMetadata": {"promptTokenCount": 3}},
        {"candidates": [{"content": {"parts": [{"text": "b"}]}, "finishReason": "STOP"}],
         "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2}},
    ])
    with mock_client(lambda r: httpx.Response(200, headers={"content-type": "application/json"}, content=body.encode())) as http:
        http.post("https://aiplatform.googleapis.com/v1/projects/p/locations/global/publishers/google/models/gemini-2.5-pro:streamGenerateContent",
                  json={"contents": []})
    stream = only_call(sink).telemetry["stream"]
    assert stream["format"] == "json-array"
    assert stream["chunkCount"] == 2
    assert stream["chunks"][1]["candidates"][0]["finishReason"] == "STOP"


# ── Bedrock (boto3) ──

class _Meta:
    endpoint_url = "https://bedrock-runtime.us-east-1.amazonaws.com"

    class service_model:
        service_name = "bedrock-runtime"


class _BedrockClient:
    meta = _Meta()


def _bedrock_meta(**headers):
    return {"HTTPStatusCode": 200, "RequestId": "rid", "HTTPHeaders": {
        "content-type": "application/json", "x-amzn-requestid": "rid", **headers}}


def test_bedrock_converse(sink):
    def original(client, operation, params):
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "Hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 7, "outputTokens": 2, "totalTokens": 9},
            "metrics": {"latencyMs": 120},
            "ResponseMetadata": _bedrock_meta(),
        }

    patch_module._bedrock_api_call(original, _BedrockClient(), "Converse", "converse", {
        "modelId": "us.anthropic.claude-sonnet-4-5-v1:0",
        "messages": [{"role": "user", "content": [{"text": "hi"}, {"image": {"format": "png", "source": {"bytes": b"\x89PNG" * 4}}}]}],
        "inferenceConfig": {"maxTokens": 100},
    })
    data = golden(only_call(sink).telemetry)
    assert data["request"] == {
        "method": "POST",
        "url": "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-4-5-v1%3A0/converse",
        "headers": {"content-type": "application/json"},
        "body": {
            "messages": [{"role": {"$s": 4}, "content": [
                {"text": {"$s": 2}},
                {"image": {"format": {"$s": 3}, "source": {"bytes": {"$bytes": 16}}}},
            ]}],
            "inferenceConfig": {"maxTokens": 100},
        },
        "bytes": None,
    }
    assert data["response"]["status"] == 200
    assert data["response"]["headers"] == {"content-type": "application/json", "x-amzn-requestid": "rid"}
    assert data["response"]["body"] == {
        "output": {"message": {"role": {"$s": 9}, "content": [{"text": {"$s": 2}}]}},
        "stopReason": {"$s": 8},
        "usage": {"inputTokens": 7, "outputTokens": 2, "totalTokens": 9},
        "metrics": {"latencyMs": 120},
    }


def test_bedrock_invoke_model_keeps_count_headers(sink):
    from botocore.response import StreamingBody

    payload = json.dumps({"id": "msg_1", "type": "message", "content": [{"type": "text", "text": "Hi"}],
                          "usage": {"input_tokens": 5, "output_tokens": 1}}).encode()

    def original(client, operation, params):
        return {"body": StreamingBody(io.BytesIO(payload), len(payload)), "contentType": "application/json",
                "ResponseMetadata": _bedrock_meta(**{"x-amzn-bedrock-input-token-count": "5",
                                                      "x-amzn-bedrock-output-token-count": "1"})}

    request_body = json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 10,
                               "messages": [{"role": "user", "content": "hi"}]}).encode()
    response = patch_module._bedrock_api_call(original, _BedrockClient(), "InvokeModel", "invoke", {
        "modelId": "anthropic.claude-3-haiku", "body": request_body, "contentType": "application/json",
    })
    assert response["body"].read() == payload  # the caller still reads the body
    data = golden(only_call(sink).telemetry)
    assert data["request"]["bytes"] == len(request_body)
    assert data["request"]["body"] == {"anthropic_version": {"$s": 18}, "max_tokens": 10,
                                       "messages": [{"role": {"$s": 4}, "content": {"$s": 2}}]}
    assert data["response"]["headers"]["x-amzn-bedrock-input-token-count"] == "5"
    assert data["response"]["headers"]["x-amzn-bedrock-output-token-count"] == "1"
    assert data["response"]["body"]["usage"] == {"input_tokens": 5, "output_tokens": 1}
    assert data["response"]["bytes"] == len(payload)


def test_bedrock_converse_stream_events(sink):
    events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Hi"}, "contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 1, "totalTokens": 4}, "metrics": {"latencyMs": 9}}},
    ]

    def original(client, operation, params):
        return {"stream": iter(events), "ResponseMetadata": _bedrock_meta(**{"content-type": "application/vnd.amazon.eventstream"})}

    response = patch_module._bedrock_api_call(original, _BedrockClient(), "ConverseStream", "converse-stream",
                                              {"modelId": "m", "messages": []})
    assert list(response["stream"]) == events
    stream = only_call(sink).telemetry["stream"]
    assert stream["format"] == "aws-eventstream"
    assert stream["chunks"][3] == {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 1, "totalTokens": 4},
                                                "metrics": {"latencyMs": 9}}}


def test_bedrock_invoke_stream_decodes_chunk_bytes(sink):
    inner = {"type": "message_delta", "usage": {"output_tokens": 4}}
    events = [{"chunk": {"bytes": json.dumps(inner).encode()}}]

    def original(client, operation, params):
        return {"body": iter(events), "ResponseMetadata": _bedrock_meta()}

    response = patch_module._bedrock_api_call(original, _BedrockClient(), "InvokeModelWithResponseStream",
                                              "invoke-with-response-stream", {"modelId": "m", "body": b"{}"})
    list(response["body"])
    assert only_call(sink).telemetry["stream"]["chunks"] == [{"chunk": {"bytes": inner}}]


def test_bedrock_client_error_sends_the_raw_exception_name(sink):
    class ValidationException(Exception):
        def __init__(self):
            super().__init__("bad")
            self.response = {"Error": {"Code": "ValidationException", "Message": "bad input"},
                             "ResponseMetadata": {"HTTPStatusCode": 400, "HTTPHeaders": {"x-amzn-errortype": "ValidationException"}}}

    def original(client, operation, params):
        raise ValidationException()

    with pytest.raises(ValidationException):
        patch_module._bedrock_api_call(original, _BedrockClient(), "Converse", "converse", {"modelId": "m"})
    telemetry = only_call(sink).telemetry
    assert telemetry["error"] == {"name": "ValidationException", "code": None}
    assert telemetry["response"]["status"] == 400
    assert telemetry["response"]["body"] == {"Error": {"Code": {"$s": 19}, "Message": {"$s": 9}}}


# ── Speech, search, embeddings (requests and httpx) ──

class _CannedAdapter(requests.adapters.BaseAdapter):
    def __init__(self, status, body, headers):
        super().__init__()
        self._status, self._body, self._headers = status, body, headers
        self.sent = []

    def send(self, request, **kwargs):
        self.sent.append(request)
        response = requests.Response()
        response.status_code = self._status
        response.headers.update(self._headers)
        response.raw = io.BytesIO(self._body)
        response.url = request.url
        response.request = request
        return response

    def close(self):
        pass


def _session(status, body, headers):
    session = requests.Session()
    adapter = _CannedAdapter(status, body, headers)
    session.mount("https://", adapter)
    return session, adapter


def test_deepgram_binary_upload_with_model_in_query(sink):
    session, adapter = _session(200, json.dumps({
        "metadata": {"request_id": "dg1", "duration": 3.5, "channels": 1},
        "results": {"channels": [{"alternatives": [{"transcript": "hello", "confidence": 0.9}]}]},
    }).encode(), {"content-type": "application/json", "dg-request-id": "dg1"})
    session.post("https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true",
                 data=b"RIFF" + b"\x00" * 60, headers={"Content-Type": "audio/wav", "Authorization": "Token dg-secret"})
    telemetry = only_call(sink).telemetry
    assert telemetry["request"]["url"] == "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true"
    assert telemetry["request"]["headers"] == {"content-type": "audio/wav"}
    assert telemetry["request"]["body"] == {"$bytes": 64}
    assert telemetry["request"]["bytes"] == 64
    assert telemetry["response"]["body"]["metadata"] == {"request_id": "dg1", "duration": 3.5, "channels": 1}
    assert telemetry["response"]["headers"]["dg-request-id"] == "dg1"
    assert "dg-secret" not in json.dumps(telemetry)


def test_elevenlabs_tts_audio_response_keeps_character_cost(sink):
    with mock_client(lambda r: httpx.Response(200, headers={"content-type": "audio/mpeg", "character-cost": "11",
                                                            "request-id": "el1"}, content=b"\xff" * 300)) as http:
        http.post("https://api.elevenlabs.io/v1/text-to-speech/voice123", json={"text": "Hello there", "model_id": "eleven_v3"})
    telemetry = only_call(sink).telemetry
    assert telemetry["request"]["body"] == {"text": {"$s": 11}, "model_id": "eleven_v3"}
    assert telemetry["response"]["body"] == {"$bytes": 300}
    assert telemetry["response"]["headers"]["character-cost"] == "11"


def test_assemblyai_poll_keeps_speech_model(sink):
    session, _ = _session(200, json.dumps({"id": "tr_1", "status": "completed", "speech_model": "universal",
                                            "audio_duration": 61, "text": "secret words"}).encode(),
                          {"content-type": "application/json"})
    session.get("https://api.assemblyai.com/v2/transcript/tr_1", headers={"authorization": "aai-secret"})
    telemetry = only_call(sink).telemetry
    assert telemetry["request"] == {
        "method": "GET", "url": "https://api.assemblyai.com/v2/transcript/tr_1", "headers": {},
        "body": None, "bytes": None,
    }
    assert telemetry["response"]["body"] == {"id": "tr_1", "status": "completed", "speech_model": "universal",
                                             "audio_duration": 61, "text": {"$s": 12}}


def test_tavily_body_api_key_never_ships_and_search_depth_is_kept(sink):
    session, _ = _session(200, json.dumps({"results": [{"url": "https://a", "content": "c"}],
                                            "usage": {"credits": 2}}).encode(), {"content-type": "application/json"})
    session.post("https://api.tavily.com/search", json={"query": "q", "api_key": "tvly-secret",
                                                          "search_depth": "advanced", "include_usage": True})
    telemetry = only_call(sink).telemetry
    assert telemetry["request"]["body"] == {"query": {"$s": 1}, "api_key": {"$s": 11},
                                            "search_depth": "advanced", "include_usage": True}
    assert telemetry["response"]["body"]["usage"] == {"credits": 2}
    assert "tvly-secret" not in json.dumps(telemetry)


def test_voyage_embeddings_do_not_ship_vectors(sink):
    with mock_client(lambda r: httpx.Response(200, json={
        "object": "list", "model": "voyage-3",
        "data": [{"object": "embedding", "embedding": [0.5] * 1024, "index": 0}],
        "usage": {"total_tokens": 8},
    })) as http:
        http.post("https://api.voyageai.com/v1/embeddings", json={"input": ["a", "b"], "model": "voyage-3"})
    telemetry = only_call(sink).telemetry
    assert telemetry["request"]["body"] == {"input": [{"$s": 1}, {"$s": 1}], "model": "voyage-3"}
    assert telemetry["response"]["body"]["data"][0]["embedding"] == [0.5] * 8 + [{"$more": 1016}]


def test_cohere_v2_chat(sink):
    with mock_client(lambda r: httpx.Response(200, json={
        "id": "co1", "finish_reason": "COMPLETE",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
        "usage": {"billed_units": {"input_tokens": 3, "output_tokens": 1}, "tokens": {"input_tokens": 70, "output_tokens": 1}},
    })) as http:
        http.post("https://api.cohere.com/v2/chat", json={"model": "command-a-03-2025", "messages": []})
    body = only_call(sink).telemetry["response"]["body"]
    assert body["usage"] == {"billed_units": {"input_tokens": 3, "output_tokens": 1},
                             "tokens": {"input_tokens": 70, "output_tokens": 1}}
    assert body["finish_reason"] == "COMPLETE"


def test_ollama_native_ndjson_stream(sink):
    lines = [{"model": "llama3", "message": {"role": "assistant", "content": "Hi"}, "done": False},
             {"model": "llama3", "done": True, "prompt_eval_count": 12, "eval_count": 3}]
    body = "\n".join(json.dumps(line) for line in lines) + "\n"
    with mock_client(lambda r: httpx.Response(200, headers={"content-type": "application/x-ndjson"}, content=body.encode())) as http:
        with http.stream("POST", "http://localhost:11434/api/chat", json={"model": "llama3", "messages": []}) as response:
            for _ in response.iter_bytes():
                pass
    stream = only_call(sink).telemetry["stream"]
    assert stream["format"] == "ndjson"
    assert stream["chunks"][1] == {"model": "llama3", "done": True, "prompt_eval_count": 12, "eval_count": 3}


# ── Capture rules ──

def test_hosts_outside_the_list_and_the_agents_trace_exporter_are_not_captured(sink):
    with mock_client(lambda r: httpx.Response(200, json={})) as http:
        http.post("https://example.com/v1/chat/completions", json={})
        http.post("https://api.openai.com/v1/traces/ingest", json={"data": []})
    assert sink.calls() == []
    assert match_provider("https://api.openai.com/v1/traces/ingest") is None
    assert match_provider("https://api.openai.com/v1/chat/completions") == "openai"


def test_gateways_keep_their_own_label():
    assert match_provider("https://api.portkey.ai/v1/chat/completions") == "portkey"
    assert match_provider("https://oai.helicone.ai/v1/chat/completions") == "helicone"


def test_runtime_hosts_widen_capture_only():
    try:
        apply_runtime_hosts([{"id": "acme-llm", "hostname": "https://llm.acme.internal/v1", "usage_input_path": "x"}])
        assert match_provider("https://llm.acme.internal/v1/generate") == "acme-llm"
        apply_runtime_hosts([{"name": "wild", "hostname": "*.acme.ai"}])
        assert match_provider("https://eu.acme.ai/x") == "wild"
        assert match_provider("https://llm.acme.internal/v1/generate") is None
    finally:
        clear_runtime_hosts()


def test_transport_errors_are_sent_with_the_raw_exception_name(sink):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with mock_client(handler) as http:
        with pytest.raises(httpx.ConnectError):
            http.post("https://api.anthropic.com/v1/messages", json={"model": "claude"})
    telemetry = only_call(sink).telemetry
    assert telemetry["response"] is None
    assert telemetry["error"] == {"name": "ConnectError", "code": None}


def test_stream_errors_mid_iteration_keep_the_raw_name(sink):
    class Broken(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"id":"a"}\n\n'
            raise httpx.ReadError("dropped")

    with mock_client(lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Broken())) as http:
        with pytest.raises(httpx.ReadError):
            with http.stream("POST", "https://api.openai.com/v1/chat/completions", json={"stream": True}) as response:
                for _ in response.iter_lines():
                    pass
    telemetry = only_call(sink).telemetry
    assert telemetry["error"]["name"] == "ReadError"
    assert telemetry["stream"]["chunks"] == [{"id": "a"}]


def test_content_capture_is_opt_in_and_telemetry_stays_a_skeleton(content_sink):
    with mock_client(lambda r: httpx.Response(200, json={"id": "x", "choices": [{"message": {"content": "answer"}}]})) as http:
        http.post("https://api.openai.com/v1/chat/completions",
                  json={"model": "m", "messages": [{"role": "user", "content": "question"}], "api_key": "sk-abcdefghijklmnopqrstuvwxyz"})
    call = only_call(content_sink)
    assert call.request_body["messages"][0]["content"] == "question"
    assert call.request_body["api_key"] == "[REDACTED]"
    assert call.response_body["choices"][0]["message"]["content"] == "answer"
    assert call.telemetry["request"]["body"]["messages"][0]["content"] == {"$s": 8}


def test_stack_frames_are_raw_and_innermost_first(sink):
    def my_business_function():
        with mock_client(lambda r: httpx.Response(200, json={})) as http:
            http.post("https://api.openai.com/v1/chat/completions", json={})

    my_business_function()
    frames = only_call(sink).call_site.frames
    names = [frame.function_name for frame in frames]
    assert "my_business_function" in names
    # Library and SDK frames are not filtered: the server does that.
    files = [frame.file_name for frame in frames]
    assert any("httpx" in name for name in files)
    assert any("costkey" in name for name in files)
    assert names.index("my_business_function") > names.index("patched_send")


def test_admission_receives_the_request_skeleton(sink):
    seen = []
    patch_module._state.admission = lambda payload: seen.append(payload) or {"allowed": True}
    with mock_client(lambda r: httpx.Response(200, json={})) as http:
        http.post("https://api.openai.com/v1/chat/completions?key=s",
                  json={"model": "gpt-4o", "max_tokens": 9, "messages": [{"role": "user", "content": "hi"}]})
    payload = seen[0]
    assert payload["request"] == {
        "method": "POST",
        "url": "https://api.openai.com/v1/chat/completions",
        "headers": {"content-type": "application/json"},
        "body": {"model": "gpt-4o", "max_tokens": 9, "messages": [{"role": {"$s": 4}, "content": {"$s": 2}}]},
        "bytes": payload["request"]["bytes"],
    }
    assert "usageBounds" not in payload and "model" not in payload and "streaming" not in payload
    assert payload["id"] == only_call(sink).id


# ── aiohttp ──

class _Server:
    """A local provider stand-in, captured through a runtime host entry."""

    def __init__(self, routes):
        routes_ = routes

        class Handler(BaseHTTPRequestHandler):
            def _reply(self):
                length = int(self.headers.get("content-length") or 0)
                self.rfile.read(length)
                status, headers, body = routes_[self.path.split("?")[0]]
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_POST = do_GET = _reply

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def test_aiohttp_json_and_sse(sink):
    aiohttp = pytest.importorskip("aiohttp")
    sse = b'data: {"id":"s1","usage":{"output_tokens":2}}\n\ndata: [DONE]\n\n'
    server = _Server({
        "/v1/json": (200, {"content-type": "application/json", "x-request-id": "a1"}, b'{"id":"j1","usage":{"input_tokens":3}}'),
        "/v1/sse": (200, {"content-type": "text/event-stream"}, sse),
    })
    apply_runtime_hosts([{"id": "local", "hostname": "127.0.0.1"}])

    async def run():
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{server.base}/v1/json", json={"model": "m", "input": "x"},
                                    params={"api-version": "1", "key": "secret"}) as response:
                assert (await response.json())["id"] == "j1"
            async with session.post(f"{server.base}/v1/sse", json={"stream": True}) as response:
                async for _ in response.content:
                    pass

    try:
        asyncio.run(run())
    finally:
        clear_runtime_hosts()
        server.close()
    first, second = sink.calls()
    assert first.provider == "local"
    assert first.telemetry["request"]["url"] == f"{server.base}/v1/json?api-version=1"
    assert first.telemetry["request"]["body"] == {"model": "m", "input": {"$s": 1}}
    assert first.telemetry["response"]["body"] == {"id": "j1", "usage": {"input_tokens": 3}}
    assert first.telemetry["response"]["headers"]["x-request-id"] == "a1"
    assert second.telemetry["stream"]["chunks"] == [{"id": "s1", "usage": {"output_tokens": 2}}]


# ── httpx2 (openai 3.x, anthropic 1.x and mistralai 3.x send through it) ──

def test_httpx2_clients_are_captured(sink):
    httpx2 = pytest.importorskip("httpx2")

    def handler(request):
        return httpx2.Response(200, json={
            "id": "msg_1", "type": "message", "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "Hi"}],
            "usage": {"input_tokens": 12, "output_tokens": 3},
        })

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as http:
        http.post("https://api.anthropic.com/v1/messages",
                  json={"model": "claude-sonnet-4-5", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]})
    call = only_call(sink)
    telemetry = call.telemetry
    assert telemetry["request"]["url"] == "https://api.anthropic.com/v1/messages"
    assert telemetry["response"]["body"]["usage"] == {"input_tokens": 12, "output_tokens": 3}


def test_httpx2_async_clients_are_captured(sink):
    httpx2 = pytest.importorskip("httpx2")

    def handler(request):
        return httpx2.Response(200, json={"id": "chatcmpl-1", "object": "chat.completion", "model": "gpt-5",
                                          "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1}})

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
            await http.post("https://api.openai.com/v1/chat/completions", json={"model": "gpt-5", "messages": []})

    asyncio.run(run())
    assert only_call(sink).telemetry["response"]["body"]["usage"]["prompt_tokens"] == 5
