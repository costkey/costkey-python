"""aiohttp and botocore capture, grounding units and non-token providers."""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import json
import socket
import struct
import zlib
from contextlib import asynccontextmanager

import httpx
import pytest

import costkey.patch as patch_module
from costkey.providers import (
    BedrockExtractor, GoogleExtractor, call_model, call_usage, find_extractor,
)
from costkey.transport import Transport
from test_devx import capture_events

needs_aiohttp = pytest.mark.skipif(importlib.util.find_spec("aiohttp") is None, reason="aiohttp not installed")
needs_botocore = pytest.mark.skipif(importlib.util.find_spec("botocore") is None, reason="botocore not installed")
needs_requests = pytest.mark.skipif(importlib.util.find_spec("requests") is None, reason="requests not installed")


@pytest.fixture()
def patched():
    with capture_events() as transport:
        patch_module._patch_httpx()
        patch_module._patch_aiohttp()
        patch_module._patch_botocore()
        patch_module._state.patched = True
        try:
            yield transport
        finally:
            patch_module.unpatch()


def _wire(event):
    return Transport("https://example.test", "key", 50, 5, False)._serialize(event)


def _calls(transport):
    return [event for event in transport.events if event.event_type == "call"]


def _sse(*events):
    return "".join(f"data: {json.dumps(event)}\r\n\r\n" for event in events).encode()


# ── Gemini grounding ──

def _grounded(queries=(), chunks=(), **metadata):
    return {
        "modelVersion": "gemini-3-pro",
        "candidates": [{"groundingMetadata": {"webSearchQueries": list(queries), "groundingChunks": list(chunks), **metadata}}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4, "totalTokenCount": 14},
    }


def test_gemini_grounding_units_count_distinct_queries_and_maps():
    usage = GoogleExtractor().extract_usage(_grounded(
        queries=["fd rates", "fd rates", "rbi repo"],
        chunks=[{"web": {"uri": "https://a"}}, {"maps": {"uri": "https://maps"}}],
    ))
    assert usage.native_units == {"searchQueries": 2, "groundingRequests": 1, "mapsGroundingRequests": 1}

    maps_only = GoogleExtractor().extract_usage(_grounded(googleMapsWidgetContextToken="tok"))
    assert maps_only.native_units == {"mapsGroundingRequests": 1}

    web_chunks_only = GoogleExtractor().extract_usage(_grounded(chunks=[{"web": {"uri": "https://a"}}]))
    assert web_chunks_only.native_units == {"groundingRequests": 1}


def test_gemini_retrieval_grounding_is_not_billed_as_search():
    usage = GoogleExtractor().extract_usage(_grounded(
        chunks=[{"retrievedContext": {"uri": "gs://docs"}}],
        retrievalQueries=["policy"],
        searchEntryPoint={"renderedContent": "<div/>"},
    ))
    assert not usage.native_units


def test_gemini_stream_unions_queries_across_chunks():
    text = _sse(_grounded(queries=["a", "b"]), _grounded(queries=["b", "c"])).decode()
    usage = patch_module._extract_sse_usage(text, GoogleExtractor())
    assert usage.native_units["searchQueries"] == 3
    assert usage.native_units["groundingRequests"] == 1
    assert (usage.input_tokens, usage.output_tokens) == (10, 4)


# ── aiohttp ──

class _LocalResolver:
    """Resolve provider hostnames to a local test server."""

    async def resolve(self, host, port=0, family=socket.AF_INET):
        return [{"hostname": host, "host": "127.0.0.1", "port": port,
                 "family": socket.AF_INET, "proto": 0, "flags": socket.AI_NUMERICHOST}]

    async def close(self):
        return None


@asynccontextmanager
async def _provider_server(handler):
    from aiohttp import web

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield port
    finally:
        await runner.cleanup()


def _session():
    import aiohttp
    return aiohttp.ClientSession(connector=aiohttp.TCPConnector(resolver=_LocalResolver()))


VERTEX = "http://us-central1-aiplatform.googleapis.com:{port}/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.5-flash"
GEMINI_CHUNKS = [
    {"candidates": [{"content": {"parts": [{"text": "Hel"}]}}], "modelVersion": "gemini-2.5-flash",
     "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 1}},
    {"candidates": [{"content": {"parts": [{"text": "lo"}]},
                     "groundingMetadata": {"webSearchQueries": ["q1"], "groundingChunks": [{"web": {}}]}}],
     "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 9, "totalTokenCount": 21}},
]


async def _gemini_handler(request):
    from aiohttp import web

    body = await request.json()
    assert "x-costkey-trace-id" not in request.headers
    if request.path.endswith(":streamGenerateContent"):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for chunk in GEMINI_CHUNKS:
            await response.write(_sse(chunk))
            await asyncio.sleep(0.01)
        await response.write_eof()
        return response
    return web.json_response({
        **_grounded(queries=["q1", "q2"]),
        "modelVersion": "gemini-2.5-flash",
        "echo": body["contents"][0]["parts"][0]["text"],
    })


def _gemini_body(text="hi"):
    return json.dumps({"contents": [{"role": "user", "parts": [{"text": text}]}]}).encode()


@needs_aiohttp
def test_aiohttp_vertex_generate_content_is_captured_and_body_still_readable(patched):
    async def run():
        async with _provider_server(_gemini_handler) as port, _session() as session:
            async with session.post(VERTEX.format(port=port) + ":generateContent", data=_gemini_body(),
                                    headers={"x-costkey-trace-id": "trace-a"}) as response:
                payload = await response.json()
                text = await response.text()
            # Callers that read the content stream directly see the body too.
            async with session.post(VERTEX.format(port=port) + ":generateContent", data=_gemini_body("again")) as response:
                raw = await response.content.read()
        return payload, text, raw

    payload, text, raw = asyncio.run(run())
    assert payload["echo"] == "hi"
    assert json.loads(text) == payload
    assert json.loads(raw)["echo"] == "again"

    calls = _calls(patched)
    assert len(calls) == 2
    event = calls[0]
    assert event.provider.value == "google"
    assert event.model == "gemini-2.5-flash"
    assert event.status_code == 200
    assert event.streaming is False
    assert event.context["traceId"] == "trace-a"
    assert event.duration_ms > 0
    assert event.call_site.frames[0].file_name.endswith("test_provider_coverage.py")
    assert (event.usage.input_tokens, event.usage.output_tokens) == (10, 4)
    wire = _wire(event)
    assert wire["usage"]["nativeUnits"] == {"searchQueries": 2, "groundingRequests": 1}
    assert wire["url"].endswith("/models/gemini-2.5-flash:generateContent")


@needs_aiohttp
def test_aiohttp_vertex_sse_stream_is_teed_and_emitted_once(patched):
    async def run():
        lines = []
        async with _provider_server(_gemini_handler) as port, _session() as session:
            response = await session.post(VERTEX.format(port=port) + ":streamGenerateContent?alt=sse",
                                          data=_gemini_body())
            # google-genai's aiohttp path: readline() until EOF, then release().
            while True:
                line = await response.content.readline()
                if not line:
                    break
                lines.append(line)
            response.release()
        return lines

    lines = asyncio.run(run())
    assert [line for line in lines if line.startswith(b"data:")] == [
        f"data: {json.dumps(chunk)}\r\n".encode() for chunk in GEMINI_CHUNKS
    ]
    calls = _calls(patched)
    assert len(calls) == 1
    event = calls[0]
    assert event.streaming is True
    assert event.model == "gemini-2.5-flash"
    assert (event.usage.input_tokens, event.usage.output_tokens, event.usage.total_tokens) == (12, 9, 21)
    assert event.usage.native_units == {"searchQueries": 1, "groundingRequests": 1}
    assert event.stream_timing.chunk_count >= 2
    assert event.stream_timing.ttft is not None


@needs_aiohttp
def test_aiohttp_stream_iterators_are_recorded(patched):
    async def run():
        async with _provider_server(_gemini_handler) as port, _session() as session:
            url = VERTEX.format(port=port) + ":streamGenerateContent?alt=sse"
            async with session.post(url, data=_gemini_body()) as response:
                chunked = b"".join([chunk async for chunk in response.content.iter_chunked(7)])
            async with session.post(url, data=_gemini_body()) as response:
                lines = [line async for line in response.content]
            async with session.post(url, data=_gemini_body()) as response:
                whole = await response.read()
        return chunked, lines, whole

    chunked, lines, whole = asyncio.run(run())
    assert chunked == whole == b"".join(lines) == _sse(*GEMINI_CHUNKS)
    calls = _calls(patched)
    assert len(calls) == 3
    assert all(call.usage.output_tokens == 9 for call in calls)


@needs_aiohttp
def test_aiohttp_stream_released_early_emits_once_with_partial_usage(patched):
    async def run():
        async with _provider_server(_gemini_handler) as port, _session() as session:
            async with session.post(VERTEX.format(port=port) + ":streamGenerateContent?alt=sse",
                                    data=_gemini_body()) as response:
                await response.content.readline()
            response.close()

    asyncio.run(run())
    calls = _calls(patched)
    assert len(calls) == 1
    assert calls[0].usage.input_tokens == 12


@needs_aiohttp
def test_aiohttp_json_kwarg_and_unmatched_hosts(patched):
    from aiohttp import web

    async def handler(request):
        body = await request.json()
        return web.json_response({"id": "chatcmpl-1", "model": body["model"],
                                  "usage": {"prompt_tokens": 3, "completion_tokens": 2}})

    async def run():
        async with _provider_server(handler) as port, _session() as session:
            async with session.post(f"http://api.openai.com:{port}/v1/chat/completions",
                                    json={"model": "gpt-5", "messages": []}) as response:
                await response.json()
            async with session.post(f"http://example.com:{port}/v1/chat/completions",
                                    json={"model": "gpt-5", "messages": []}) as response:
                await response.json()

    asyncio.run(run())
    calls = _calls(patched)
    assert len(calls) == 1
    assert calls[0].model == "gpt-5"
    assert calls[0].provider_request_id == "chatcmpl-1"
    assert calls[0].usage.total_tokens == 5


@needs_aiohttp
def test_aiohttp_audio_response_is_not_preread(patched):
    from aiohttp import web

    audio = b"ID3" + bytes(range(256)) * 64

    async def handler(request):
        return web.Response(body=audio, content_type="audio/mpeg", headers={"character-cost": "11"})

    async def run():
        async with _provider_server(handler) as port, _session() as session:
            async with session.post(f"http://api.elevenlabs.io:{port}/v1/text-to-speech/voice-1/stream",
                                    json={"text": "hello world", "model_id": "eleven_flash_v2_5"}) as response:
                assert patched.events, "emitted before the caller reads the audio"
                return b"".join([chunk async for chunk in response.content.iter_chunked(1024)])

    assert asyncio.run(run()) == audio
    event = _calls(patched)[0]
    assert event.provider.value == "elevenlabs"
    assert event.model == "eleven_flash_v2_5"
    assert event.usage.native_units == {"characters": 11}
    assert event.response_body is None


@needs_aiohttp
def test_aiohttp_patch_is_idempotent_and_unpatch_restores():
    import aiohttp

    original = aiohttp.ClientSession._request
    try:
        patch_module._patch_aiohttp()
        patched_request = aiohttp.ClientSession._request
        assert patched_request is not original
        patch_module._patch_aiohttp()
        assert aiohttp.ClientSession._request is patched_request
    finally:
        patch_module._unpatch_aiohttp()
    assert aiohttp.ClientSession._request is original
    assert patch_module._state._original_aiohttp_request is None


# ── Speech, search and scraping providers ──

def _post(transport_handler, url, **kwargs):
    with httpx.Client(transport=httpx.MockTransport(transport_handler)) as http:
        return http.post(url, **kwargs)


def test_elevenlabs_tts_counts_request_characters_or_header(patched):
    _post(lambda r: httpx.Response(200, content=b"\xff\xfb audio", headers={"content-type": "audio/mpeg"}),
          "https://api.elevenlabs.io/v1/text-to-speech/v1", json={"text": "नमस्ते दुनिया"})
    _post(lambda r: httpx.Response(200, json={"audio_base64": "QUJD" * 5000, "alignment": {}},
                                   headers={"x-character-count": "7"}),
          "https://api.us.elevenlabs.io/v1/text-to-speech/v1/with-timestamps",
          json={"text": "ignored by header", "model_id": "eleven_v3"})
    _post(lambda r: httpx.Response(401, json={"detail": "bad key"}),
          "https://api.elevenlabs.io/v1/text-to-speech/v1", json={"text": "not billed"})
    first, second, failed = _calls(patched)
    assert (first.model, first.usage.native_units) == ("eleven_multilingual_v2", {"characters": 13})
    assert (second.model, second.usage.native_units) == ("eleven_v3", {"characters": 7})
    assert _wire(second)["responseBody"]["audio_base64"] == "[omitted 20000 chars]"
    assert failed.usage is None


def test_elevenlabs_speech_to_text_reads_multipart_model_and_duration(patched):
    _post(lambda r: httpx.Response(200, json={"text": "hi", "audio_duration_secs": 12.5}),
          "https://api.elevenlabs.io/v1/speech-to-text",
          data={"model_id": "scribe_v2"}, files={"file": ("a.wav", b"RIFF" + b"\0" * 64)})
    _post(lambda r: httpx.Response(200, json={"text": "hi", "words": [{"end": 1.0}, {"end": 3.25}]}),
          "https://api.elevenlabs.io/v1/speech-to-text", files={"file": ("a.wav", b"RIFF")})
    first, second = _calls(patched)
    assert (first.model, first.usage.native_units) == ("scribe_v2", {"audioSeconds": 12.5})
    assert (second.model, second.usage.native_units) == ("scribe_v1", {"audioSeconds": 3.25})


def test_rumik_counts_characters_without_parsing_wav(patched):
    _post(lambda r: httpx.Response(200, content=b"RIFF\x00\x00WAVEfmt ", headers={"content-type": "audio/wav"}),
          "https://silk-api.rumik.ai/v1/tts", json={"text": "Namaste!", "speaker": "a"})
    event = _calls(patched)[0]
    assert (event.provider.value, event.model) == ("rumik", "mulberry")
    assert _wire(event)["usage"]["nativeUnits"] == {"characters": 8}


def test_exa_models_units_and_reported_cost(patched):
    def handler(request):
        if request.url.path == "/contents":
            return httpx.Response(200, json={"results": [{}, {}, {}], "costDollars": {"total": 0.003}})
        return httpx.Response(200, json={"results": [], "costDollars": {"total": 0.012}})

    _post(handler, "https://api.exa.ai/search", json={"query": "x", "numResults": 20})
    _post(handler, "https://api.exa.ai/search", json={"query": "x", "type": "deep-reasoning"})
    _post(handler, "https://api.exa.ai/findSimilar", json={"url": "https://a"})
    _post(handler, "https://api.exa.ai/contents", json={"ids": ["a", "b", "c"]})
    _post(handler, "https://api.exa.ai/answer", json={"query": "x"})
    search, deep, similar, contents, answer = _calls(patched)
    assert (search.model, search.usage.native_units) == ("search", {"searchQueries": 1, "searchResults": 20})
    assert (deep.model, deep.usage.native_units["searchResults"]) == ("search-deep-reasoning", 10)
    assert similar.model == "find-similar"
    assert (contents.model, contents.usage.native_units) == ("contents", {"pages": 3})
    assert answer.model == "answer"
    wire = _wire(search)
    assert wire["usage"]["costUsd"] == 0.012
    assert wire["costUsd"] == 0.012
    assert call_model(find_extractor("https://api.exa.ai/search"), "https://api.exa.ai/search", {"type": "instant"}, None) == "search-instant"


def test_firecrawl_credits_defaults_and_job_polls(patched):
    def handler(request):
        path = request.url.path
        if path == "/v2/scrape":
            return httpx.Response(200, json={"success": True, "data": {"metadata": {"creditsUsed": 5}}})
        if path == "/v2/crawl/job-1":
            return httpx.Response(200, json={"status": "completed", "creditsUsed": 40})
        if path == "/v2/search":
            return httpx.Response(200, json={"success": True, "data": {"web": []}, "creditsUsed": 2})
        return httpx.Response(200, json={"success": True, "links": []})

    _post(handler, "https://api.firecrawl.dev/v2/scrape", json={"url": "https://a"})
    _post(handler, "https://api.firecrawl.dev/v2/map", json={"url": "https://a"})
    _post(handler, "https://api.firecrawl.dev/v2/batch/scrape", json={"urls": []})
    _post(handler, "https://api.firecrawl.dev/v2/crawl/job-1", json={})
    _post(handler, "https://api.firecrawl.dev/v2/search", json={"query": "x", "limit": 3})
    scrape, map_, batch, poll, search = _calls(patched)
    assert (scrape.model, scrape.usage.native_units) == ("scrape", {"credits": 5})
    assert (map_.model, map_.usage.native_units) == ("map", {"credits": 1})
    assert (batch.model, batch.usage) == ("batch-scrape", None)
    assert (poll.model, poll.usage) == ("crawl", None)
    assert search.usage.native_units == {"credits": 2, "searchQueries": 1, "searchResults": 3}


def test_typesafe_tokens_and_default_model(patched):
    _post(lambda r: httpx.Response(200, json={"usage": {"input_tokens": 30, "output_tokens": 7}}),
          "https://api.typesafe.ai/v1/systemone", json={"state": {}, "questions": {}})
    event = _calls(patched)[0]
    assert (event.provider.value, event.model) == ("typesafe", "jev-latest")
    assert (event.usage.input_tokens, event.usage.output_tokens, event.usage.total_tokens) == (30, 7, 37)


def test_httpx_multipart_upload_to_a_provider_is_not_broken(patched):
    seen = []

    def handler(request):
        seen.append(request.read())
        return httpx.Response(200, json={"text": "ok", "usage": {"input_tokens": 1, "output_tokens": 1}})

    response = _post(handler, "https://api.openai.com/v1/audio/transcriptions",
                     data={"model": "gpt-4o-transcribe"}, files={"file": ("a.wav", b"RIFF")})
    assert response.json()["text"] == "ok"
    assert b"RIFF" in seen[0]
    assert _calls(patched)[0].model == "gpt-4o-transcribe"


# ── requests ──

def _requests_session(status, headers, body):
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.response import HTTPResponse

    class FakeAdapter(HTTPAdapter):
        def send(self, request, **_kwargs):
            raw = HTTPResponse(body=io.BytesIO(body), headers=headers, status=status, preload_content=False)
            return self.build_response(request, raw)

    session = requests.Session()
    session.mount("https://", FakeAdapter())
    return session


@needs_requests
def test_requests_streams_are_recorded_as_the_caller_reads(patched):
    patch_module._patch_requests()
    openai_sse = _sse(
        {"id": "chatcmpl-9", "model": "gpt-5", "choices": [{"delta": {"content": "hi"}}]},
        {"id": "chatcmpl-9", "choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 3}},
    ) + b"data: [DONE]\n\n"
    url = "https://api.openai.com/v1/chat/completions"
    session = _requests_session(200, {"content-type": "text/event-stream"}, openai_sse)
    response = session.post(url, json={"model": "gpt-5", "stream": True}, stream=True)
    assert not _calls(patched), "nothing is read before the caller reads"
    lines = list(response.iter_lines())
    assert lines[0].startswith(b"data:")

    session = _requests_session(200, {"content-type": "text/event-stream"}, openai_sse)
    session.post(url, json={"model": "gpt-5", "stream": True})

    session = _requests_session(200, {"content-type": "application/vnd.amazon.eventstream"}, CONVERSE_STREAM)
    response = session.post("https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-lite-v1%3A0/converse-stream",
                            json={"messages": []}, stream=True)
    assert b"".join(response.iter_content(64)) == CONVERSE_STREAM

    session = _requests_session(200, {"content-type": "application/json"},
                                json.dumps({"model": "gpt-5", "usage": {"prompt_tokens": 2, "completion_tokens": 1}}).encode())
    with session.post(url, json={"model": "gpt-5"}, stream=True) as response:
        assert response.json()["model"] == "gpt-5"

    iterated, preread, bedrock, json_stream = _calls(patched)
    assert iterated.provider_request_id == "chatcmpl-9"
    for event in (iterated, preread):
        assert (event.model, event.usage.input_tokens, event.usage.output_tokens) == ("gpt-5", 8, 3)
    assert (bedrock.model, bedrock.usage.input_tokens, bedrock.usage.output_tokens) == ("amazon.nova-lite-v1:0", 40, 9)
    assert json_stream.usage.total_tokens == 3


# ── Bedrock ──

def _event_frame(event_type: str, payload: dict, message_type: str = "event") -> bytes:
    headers = b""
    for name, value in ((":event-type", event_type), (":content-type", "application/json"),
                        (":message-type", message_type)):
        headers += bytes([len(name)]) + name.encode() + b"\x07" + struct.pack(">H", len(value)) + value.encode()
    body = json.dumps(payload).encode()
    prelude = struct.pack(">II", 16 + len(headers) + len(body), len(headers))
    message = prelude + struct.pack(">I", zlib.crc32(prelude)) + headers + body
    return message + struct.pack(">I", zlib.crc32(message))


CONVERSE_STREAM = b"".join([
    _event_frame("messageStart", {"role": "assistant"}),
    _event_frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "hi"}}),
    _event_frame("messageStop", {"stopReason": "end_turn"}),
    _event_frame("metadata", {"usage": {"inputTokens": 40, "outputTokens": 9, "totalTokens": 49,
                                        "cacheReadInputTokens": 30}, "metrics": {"latencyMs": 120}}),
])


def _chunk(payload: dict) -> bytes:
    return _event_frame("chunk", {"bytes": base64.b64encode(json.dumps(payload).encode()).decode()})


INVOKE_STREAM = b"".join([
    _chunk({"type": "message_start", "message": {"model": "claude", "usage": {"input_tokens": 25, "output_tokens": 1}}}),
    _chunk({"type": "content_block_delta", "delta": {"text": "hello"}}),
    _chunk({"type": "message_delta", "usage": {"output_tokens": 16},
            "amazon-bedrock-invocationMetrics": {"inputTokenCount": 25, "outputTokenCount": 16,
                                                 "cacheReadInputTokenCount": 5, "invocationLatency": 300}}),
])


def test_event_stream_decoder_reads_converse_and_invoke_usage():
    converse = patch_module._extract_event_stream_usage(CONVERSE_STREAM, BedrockExtractor())
    assert (converse.input_tokens, converse.output_tokens, converse.cache_read_tokens) == (40, 9, 30)
    invoke = patch_module._extract_event_stream_usage(INVOKE_STREAM, BedrockExtractor())
    assert (invoke.input_tokens, invoke.output_tokens, invoke.cache_read_tokens) == (25, 16, 5)
    # Exceptions and a truncated trailing frame are ignored.
    noisy = INVOKE_STREAM + _event_frame("throttlingException", {"message": "slow"}, "exception") + CONVERSE_STREAM[:20]
    assert patch_module._extract_event_stream_usage(noisy, BedrockExtractor()).output_tokens == 16


def test_httpx_bedrock_event_stream_gets_usage(patched):
    url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-sonnet-4-5-v1%3A0/converse-stream"
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(
        200, content=CONVERSE_STREAM, headers={"content-type": "application/vnd.amazon.eventstream"},
    ))) as http:
        with http.stream("POST", url, json={"messages": []}) as response:
            received = b"".join(response.iter_bytes())
    assert received == CONVERSE_STREAM
    event = _calls(patched)[0]
    assert event.provider.value == "bedrock"
    assert event.model == "anthropic.claude-sonnet-4-5-v1:0"
    assert (event.usage.input_tokens, event.usage.output_tokens, event.usage.cache_read_tokens) == (40, 9, 30)


@needs_aiohttp
def test_aiohttp_bedrock_invoke_stream_gets_usage(patched):
    from aiohttp import web

    async def handler(request):
        response = web.StreamResponse(headers={"Content-Type": "application/vnd.amazon.eventstream"})
        await response.prepare(request)
        await response.write(INVOKE_STREAM)
        await response.write_eof()
        return response

    async def run():
        async with _provider_server(handler) as port, _session() as session:
            url = f"http://bedrock-runtime.us-east-1.amazonaws.com:{port}/model/anthropic.claude-3-haiku/invoke-with-response-stream"
            async with session.post(url, json={"messages": []}) as response:
                return b"".join([chunk async for chunk, _ in response.content.iter_chunks()])

    assert asyncio.run(run()) == INVOKE_STREAM
    event = _calls(patched)[0]
    assert event.model == "anthropic.claude-3-haiku"
    assert (event.usage.input_tokens, event.usage.output_tokens) == (25, 16)


def test_bedrock_invoke_model_headers_are_authoritative():
    usage = call_usage(
        BedrockExtractor(), "https://bedrock-runtime.us-east-1.amazonaws.com/model/meta.llama3/invoke",
        None, {"generation": "hi", "prompt_token_count": 1, "generation_token_count": 2},
        {"x-amzn-bedrock-input-token-count": "17", "x-amzn-bedrock-output-token-count": "4"},
    )
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (17, 4, 21)


class _RawBody:
    def __init__(self, data: bytes) -> None:
        self._data = io.BytesIO(data)

    def stream(self, **_kwargs):
        while chunk := self._data.read(37):
            yield chunk

    def read(self, amt=None, **_kwargs):
        return self._data.read(amt)

    def close(self):
        return None


def _bedrock_client(responses):
    from botocore.awsrequest import AWSResponse
    import botocore.session

    session = botocore.session.get_session()
    client = session.create_client(
        "bedrock-runtime", region_name="us-east-1",
        aws_access_key_id="testing", aws_secret_access_key="testing",
    )

    def respond(request, **_kwargs):
        status, headers, body = responses.pop(0)
        return AWSResponse(request.url, status, {"content-length": str(len(body)), **headers}, _RawBody(body))

    client.meta.events.register("before-send.bedrock-runtime.*", respond)
    return client


@needs_botocore
def test_botocore_converse_and_invoke_model(patched):
    invoke_body = json.dumps({"content": [{"text": "hi"}], "usage": {"input_tokens": 1, "output_tokens": 1}}).encode()
    client = _bedrock_client([
        (200, {"content-type": "application/json"}, json.dumps({
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 11, "outputTokens": 3, "totalTokens": 14, "cacheWriteInputTokens": 6},
            "metrics": {"latencyMs": 50},
        }).encode()),
        (200, {"content-type": "application/json", "x-amzn-bedrock-input-token-count": "21",
               "x-amzn-bedrock-output-token-count": "8"}, invoke_body),
    ])
    converse = client.converse(
        modelId="anthropic.claude-sonnet-4-5",
        messages=[{"role": "user", "content": [{"text": "hello"}, {"image": {"format": "png", "source": {"bytes": b"\x89PNG"}}}]}],
        system=[{"text": "be brief"}],
    )
    invoked = client.invoke_model(modelId="anthropic.claude-3-haiku", body=json.dumps({"messages": []}))
    assert converse["usage"]["inputTokens"] == 11
    assert invoked["body"].read() == invoke_body, "the caller still reads the whole body"

    first, second = _calls(patched)
    assert (first.provider.value, first.model, first.status_code) == ("bedrock", "anthropic.claude-sonnet-4-5", 200)
    assert (first.usage.input_tokens, first.usage.output_tokens, first.usage.cache_creation_tokens) == (11, 3, 6)
    assert first.system_prompt == "be brief"
    wire = _wire(first)
    json.dumps(wire, allow_nan=False)
    assert "4 bytes" in json.dumps(wire["requestBody"])
    assert wire["url"] == "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-sonnet-4-5/converse"
    assert (second.model, second.usage.input_tokens, second.usage.output_tokens) == ("anthropic.claude-3-haiku", 21, 8)


@needs_botocore
def test_botocore_streams_capture_usage_and_early_close(patched):
    stream_headers = {"content-type": "application/vnd.amazon.eventstream"}
    client = _bedrock_client([
        (200, stream_headers, CONVERSE_STREAM),
        (200, stream_headers, INVOKE_STREAM),
        (200, stream_headers, INVOKE_STREAM),
    ])
    converse = client.converse_stream(modelId="amazon.nova-pro-v1:0", messages=[{"role": "user", "content": [{"text": "hi"}]}])
    events = list(converse["stream"])
    assert events[-1]["metadata"]["usage"]["outputTokens"] == 9

    invoked = client.invoke_model_with_response_stream(modelId="anthropic.claude-3-haiku", body=b"{}")
    chunks = [json.loads(event["chunk"]["bytes"]) for event in invoked["body"]]
    assert chunks[0]["type"] == "message_start"

    early = client.invoke_model_with_response_stream(modelId="anthropic.claude-3-haiku", body=b"{}")
    next(iter(early["body"]))
    early["body"].close()

    converse_event, invoke_event, early_event = _calls(patched)
    assert converse_event.streaming and converse_event.model == "amazon.nova-pro-v1:0"
    assert (converse_event.usage.input_tokens, converse_event.usage.output_tokens) == (40, 9)
    assert (invoke_event.usage.input_tokens, invoke_event.usage.output_tokens) == (25, 16)
    assert invoke_event.stream_timing.chunk_count == 3
    assert early_event.usage.input_tokens == 25


@needs_botocore
def test_botocore_error_is_recorded_and_raised(patched):
    from botocore.exceptions import ClientError

    client = _bedrock_client([
        (400, {"content-type": "application/json", "x-amzn-errortype": "ValidationException"},
         json.dumps({"message": "bad model"}).encode()),
    ])
    with pytest.raises(ClientError):
        client.converse(modelId="nope", messages=[])
    event = _calls(patched)[0]
    assert (event.status_code, event.usage, event.model) == (400, None, "nope")


@needs_botocore
def test_botocore_other_services_pass_through_and_unpatch_restores():
    import botocore.session
    from botocore.client import BaseClient

    original = BaseClient._make_api_call
    with capture_events() as transport:
        patch_module._patch_botocore()
        try:
            patched_call = BaseClient._make_api_call
            assert patched_call is not original
            patch_module._patch_botocore()
            assert BaseClient._make_api_call is patched_call
            s3 = botocore.session.get_session().create_client(
                "s3", region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing",
            )
            s3.meta.events.register(
                "before-call.s3.ListBuckets",
                lambda **_kw: (type("Http", (), {"status_code": 200})(), {"Buckets": []}),
            )
            assert s3.list_buckets() == {"Buckets": []}
            assert transport.events == []
        finally:
            patch_module.unpatch()
    assert BaseClient._make_api_call is original
    assert patch_module._state._original_botocore_call is None


def test_httpx_stream_closed_unread_reports_once(patched):
    body = _sse({"type": "message_start", "message": {"usage": {"input_tokens": 4, "output_tokens": 1}}})
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(
        200, content=body, headers={"content-type": "text/event-stream"},
    ))) as http:
        with http.stream("POST", "https://api.anthropic.com/v1/messages", json={"model": "claude", "stream": True}):
            pass
        with http.stream("POST", "https://api.anthropic.com/v1/messages", json={"model": "claude", "stream": True}) as response:
            for line in response.iter_lines():
                if line:
                    break
    unread, partial = _calls(patched)
    assert unread.usage is None and unread.model == "claude"
    assert partial.usage.input_tokens == 4


def test_gemini_tool_use_prompt_tokens_are_billed_as_input():
    from costkey.providers import GoogleExtractor
    usage = GoogleExtractor().extract_usage({"usageMetadata": {"promptTokenCount": 100, "toolUsePromptTokenCount": 900, "candidatesTokenCount": 20, "totalTokenCount": 1020}})
    assert usage.input_tokens == 1000
    assert usage.total_tokens == 1020
    assert "toolUsePromptTokenCount" not in (usage.native_units or {})
