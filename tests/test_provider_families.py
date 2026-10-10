"""OpenAI-compatible hosts, Vertex partner models, speech, embeddings and search."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from costkey.providers import call_model, call_usage, find_extractor
from test_provider_coverage import (  # noqa: F401  (patched is a fixture)
    _calls, _provider_server, _requests_session, _session, _sse, _wire,
    needs_aiohttp, needs_requests, patched,
)


def _send(handler, method, url, **kwargs):
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        return http.request(method, url, **kwargs)


def _json(body, status=200, **headers):
    return lambda _request: httpx.Response(status, json=body, headers=headers or None)


AUDIO = b"ID3" + bytes(range(256)) * 16


# ── OpenAI-compatible hosts ──

OPENAI_COMPATIBLE = [
    ("deepinfra", "api.deepinfra.com", "/v1/openai/chat/completions"),
    ("nvidia", "integrate.api.nvidia.com", "/v1/chat/completions"),
    ("sambanova", "api.sambanova.ai", "/v1/chat/completions"),
    ("novita", "api.novita.ai", "/v3/openai/chat/completions"),
    ("hyperbolic", "api.hyperbolic.xyz", "/v1/chat/completions"),
    ("nebius", "api.studio.nebius.com", "/v1/chat/completions"),
    ("nebius", "api.studio.nebius.ai", "/v1/embeddings"),
    ("nebius", "api.tokenfactory.nebius.com", "/v1/chat/completions"),
    ("alibaba", "dashscope.aliyuncs.com", "/compatible-mode/v1/chat/completions"),
    ("alibaba", "dashscope-intl.aliyuncs.com", "/compatible-mode/v1/chat/completions"),
    ("alibaba", "dashscope-us.aliyuncs.com", "/compatible-mode/v1/responses"),
    ("moonshot", "api.moonshot.ai", "/v1/chat/completions"),
    ("moonshot", "api.moonshot.cn", "/v1/chat/completions"),
    ("zai", "api.z.ai", "/api/paas/v4/chat/completions"),
    ("zai", "open.bigmodel.cn", "/api/paas/v4/chat/completions"),
    ("minimax", "api.minimax.io", "/v1/chat/completions"),
    ("minimax", "api.minimaxi.com", "/v1/text/chatcompletion_v2"),
    ("minimax", "api.minimax.chat", "/v1/chat/completions"),
    ("huggingface", "router.huggingface.co", "/v1/chat/completions"),
    ("huggingface", "api-inference.huggingface.co", "/models/x/v1/chat/completions"),
    ("baseten", "inference.baseten.co", "/v1/chat/completions"),
    ("lambda", "api.lambda.ai", "/v1/completions"),
    ("featherless", "api.featherless.ai", "/v1/chat/completions"),
    ("friendli", "api.friendli.ai", "/serverless/v1/chat/completions"),
    ("ai21", "api.ai21.com", "/studio/v1/chat/completions"),
    ("upstage", "api.upstage.ai", "/v1/chat/completions"),
    ("sarvam", "api.sarvam.ai", "/v1/chat/completions"),
    ("cloudflare", "api.cloudflare.com", "/client/v4/accounts/acct/ai/v1/chat/completions"),
    ("vercel", "ai-gateway.vercel.sh", "/v1/chat/completions"),
    ("ollama", "localhost:11434", "/v1/chat/completions"),
    ("ollama", "127.0.0.1:11434", "/v1/embeddings"),
]

OPENAI_USAGE = {
    "prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
    "prompt_tokens_details": {"cached_tokens": 100},
    "completion_tokens_details": {"reasoning_tokens": 12},
}


@pytest.mark.parametrize("provider,host,path", OPENAI_COMPATIBLE)
def test_openai_compatible_hosts_report_openai_usage(patched, provider, host, path):
    scheme = "http" if host.endswith(":11434") else "https"
    _send(_json({"id": "cmpl-1", "model": "served-model", "usage": OPENAI_USAGE}),
          "POST", f"{scheme}://{host}{path}", json={"model": "requested-model", "messages": []})
    event = _calls(patched)[0]
    assert event.provider.value == provider
    assert event.model == "served-model"
    assert event.request_model == "requested-model"
    usage = event.usage
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (120, 30, 150)
    assert (usage.cache_read_tokens, usage.reasoning_tokens) == (100, 12)


def test_openai_compatible_sse_stream_merges_final_usage(patched):
    body = _sse(
        {"id": "c1", "model": "deepseek-ai/DeepSeek-V3", "choices": [{"delta": {"content": "hi"}}]},
        {"id": "c1", "choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4,
                                              "prompt_tokens_details": {"cached_tokens": 2}}},
    ) + b"data: [DONE]\n\n"
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(
        200, content=body, headers={"content-type": "text/event-stream"},
    ))) as http:
        with http.stream("POST", "https://api.deepinfra.com/v1/openai/chat/completions",
                         json={"model": "deepseek-ai/DeepSeek-V3", "stream": True}) as response:
            list(response.iter_lines())
    event = _calls(patched)[0]
    assert (event.provider.value, event.streaming, event.model) == ("deepinfra", True, "deepseek-ai/DeepSeek-V3")
    assert (event.usage.input_tokens, event.usage.output_tokens, event.usage.cache_read_tokens) == (9, 4, 2)


@needs_aiohttp
def test_openai_compatible_hosts_over_aiohttp(patched):
    from aiohttp import web

    async def handler(request):
        body = await request.json()
        return web.json_response({"model": body["model"], "usage": {"prompt_tokens": 5, "completion_tokens": 2}})

    async def run():
        async with _provider_server(handler) as port, _session() as session:
            for host in ("integrate.api.nvidia.com", "api.moonshot.ai"):
                async with session.post(f"http://{host}:{port}/v1/chat/completions",
                                        json={"model": "kimi-k2", "messages": []}) as response:
                    await response.json()

    asyncio.run(run())
    nvidia, moonshot = _calls(patched)
    assert (nvidia.provider.value, moonshot.provider.value) == ("nvidia", "moonshot")
    assert moonshot.usage.total_tokens == 7


def test_gateways_and_unlisted_hosts_keep_their_handling():
    for url, provider in (
        ("https://api.portkey.ai/v1/chat/completions", "openai"),
        ("https://oai.helicone.ai/v1/chat/completions", "openai"),
        ("https://anthropic.helicone.ai/v1/messages", "anthropic"),
        ("https://gateway.helicone.ai/v1/chat/completions", "openai"),
    ):
        assert find_extractor(url).provider.value == provider
    for url in (
        "https://api.cloudflare.com/client/v4/zones",
        "http://localhost:8080/api/chat",
        "http://example.com:11434/api/chat",
        "https://deepinfra.com/v1/chat/completions",
    ):
        assert find_extractor(url) is None, url


# ── MiniMax and Sarvam speech ──

def test_minimax_t2a_counts_request_characters_and_never_reads_audio(patched):
    hex_audio = {"data": {"audio": "49443303" * 4000, "status": 2}, "extra_info": {"usage_characters": 99}}
    _send(_json(hex_audio), "POST", "https://api.minimax.io/v1/t2a_v2?GroupId=g",
          json={"model": "speech-02-hd", "text": "Hello, नमस्ते", "voice_setting": {}})
    _send(_json({"base_resp": {"status_code": 1004}}, status=401), "POST",
          "https://api.minimax.io/v1/t2a_v2", json={"model": "speech-02-hd", "text": "denied"})
    event, failed = _calls(patched)
    assert (event.provider.value, event.model) == ("minimax", "speech-02-hd")
    assert event.usage.native_units == {"characters": 13}
    assert event.response_body is None
    assert failed.usage is None


def test_sarvam_tts_and_stt(patched):
    _send(_json({"request_id": "r1", "audios": ["UklGR" * 4000]}), "POST", "https://api.sarvam.ai/text-to-speech",
          json={"inputs": ["नमस्ते", "दुनिया"], "target_language_code": "hi-IN"})
    _send(_json({"audios": []}), "POST", "https://api.sarvam.ai/text-to-speech",
          json={"text": "Hello", "model": "bulbul:v3"})
    _send(_json({"transcript": "hi", "duration": 4.5}), "POST", "https://api.sarvam.ai/speech-to-text",
          data={"model": "saarika:v2"}, files={"file": ("a.wav", b"RIFF")})
    _send(_json({"transcript": "hi"}), "POST", "https://api.sarvam.ai/speech-to-text-translate",
          files={"file": ("a.wav", b"RIFF")})
    tts, tts_text, stt, translate = _calls(patched)
    assert (tts.provider.value, tts.model, tts.usage.native_units) == ("sarvam", "bulbul:v2", {"characters": 12})
    assert tts.response_body is None
    assert (tts_text.model, tts_text.usage.native_units) == ("bulbul:v3", {"characters": 5})
    assert (stt.model, stt.usage.native_units) == ("saarika:v2", {"audioSeconds": 4.5})
    assert (translate.model, translate.usage) == ("saarika:v2.5", None)


# ── Cloudflare and Ollama native routes ──

def test_cloudflare_native_run_takes_model_from_path_and_result_usage(patched):
    url = "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/meta/llama-3.1-8b-instruct"
    _send(_json({"success": True, "result": {"response": "hi", "usage": {
        "prompt_tokens": 14, "completion_tokens": 6, "total_tokens": 20}}}),
        "POST", url, json={"messages": []})
    stream = _sse({"response": "h", "p": "abc"}, {"response": "", "usage": {"prompt_tokens": 3, "completion_tokens": 8}})
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(
        200, content=stream, headers={"content-type": "text/event-stream"},
    ))) as http:
        with http.stream("POST", url.replace("@cf/meta", "%40cf/meta"), json={"messages": [], "stream": True}) as response:
            list(response.iter_bytes())
    native, streamed = _calls(patched)
    assert (native.provider.value, native.model) == ("cloudflare", "@cf/meta/llama-3.1-8b-instruct")
    assert (native.usage.input_tokens, native.usage.output_tokens) == (14, 6)
    assert (streamed.model, streamed.usage.input_tokens, streamed.usage.output_tokens) == (
        "@cf/meta/llama-3.1-8b-instruct", 3, 8)


OLLAMA_NDJSON = b"".join(json.dumps(chunk).encode() + b"\n" for chunk in (
    {"model": "llama3.2", "message": {"role": "assistant", "content": "Hel"}, "done": False},
    {"model": "llama3.2", "message": {"role": "assistant", "content": "lo"}, "done": False},
    {"model": "llama3.2", "message": {"role": "assistant", "content": ""}, "done": True,
     "prompt_eval_count": 26, "eval_count": 290, "total_duration": 4883583458},
))


def test_ollama_native_ndjson_and_json(patched):
    def handler(request):
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"model": "qwen3", "response": "ok", "done": True,
                                             "prompt_eval_count": 11, "eval_count": 3})
        return httpx.Response(200, content=OLLAMA_NDJSON, headers={"content-type": "application/x-ndjson"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        # The ollama client streams /api/chat line by line.
        with http.stream("POST", "http://localhost:11434/api/chat", json={"model": "llama3.2", "messages": []}) as response:
            lines = list(response.iter_lines())
        # A caller that does not stream still gets the whole NDJSON body.
        http.post("http://127.0.0.1:11434/api/chat", json={"model": "llama3.2", "messages": []})
        http.post("http://localhost:11434/api/generate", json={"model": "qwen3", "prompt": "x", "stream": False})
    assert len(lines) == 3
    streamed, preread, generated = _calls(patched)
    for event in (streamed, preread):
        assert (event.provider.value, event.model, event.streaming) == ("ollama", "llama3.2", True)
        assert (event.usage.input_tokens, event.usage.output_tokens, event.usage.total_tokens) == (26, 290, 316)
    assert (generated.model, generated.usage.input_tokens, generated.usage.output_tokens) == ("qwen3", 11, 3)
    assert "total_duration" not in (generated.usage.native_units or {})


@needs_aiohttp
def test_ollama_ndjson_stream_over_aiohttp(patched):
    from aiohttp import web

    async def handler(request):
        response = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
        await response.prepare(request)
        for line in OLLAMA_NDJSON.splitlines(keepends=True):
            await response.write(line)
        await response.write_eof()
        return response

    async def run():
        async with _provider_server(handler) as port, _session() as session:
            async with session.post(f"http://localhost:{port}/api/chat", json={"model": "llama3.2"}) as response:
                return [line async for line in response.content]

    from costkey import providers

    class AnyPortOllama(providers.OllamaExtractor):
        # Ollama matches only port 11434; the test server's port is random.
        def match(self, url):
            return providers._host(url) == "localhost"

    providers._extractors.insert(0, AnyPortOllama())
    try:
        assert len(asyncio.run(run())) == 3
    finally:
        providers._extractors.pop(0)
    event = _calls(patched)[0]
    assert (event.provider.value, event.streaming) == ("ollama", True)
    assert (event.model, event.usage.input_tokens, event.usage.output_tokens) == ("llama3.2", 26, 290)


# ── Vertex partner models ──

VERTEX_ANTHROPIC = (
    "https://us-east5-aiplatform.googleapis.com/v1/projects/p/locations/us-east5/"
    "publishers/anthropic/models/claude-sonnet-4-5@20250929"
)


def test_vertex_anthropic_raw_predict_uses_anthropic_usage(patched):
    _send(_json({"id": "msg_1", "type": "message", "model": "claude-sonnet-4-5-20250929",
                 "usage": {"input_tokens": 50, "output_tokens": 20, "cache_read_input_tokens": 400,
                           "cache_creation_input_tokens": 60}}),
          "POST", VERTEX_ANTHROPIC + ":rawPredict", json={"anthropic_version": "vertex-2023-10-16", "messages": []})
    stream = _sse(
        {"type": "message_start", "message": {"model": "claude-sonnet-4-5-20250929",
                                              "usage": {"input_tokens": 50, "output_tokens": 1,
                                                        "cache_read_input_tokens": 400}}},
        {"type": "content_block_delta", "delta": {"text": "hi"}},
        {"type": "message_delta", "usage": {"output_tokens": 33}},
    )
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(
        200, content=stream, headers={"content-type": "text/event-stream"},
    ))) as http:
        with http.stream("POST", VERTEX_ANTHROPIC + ":streamRawPredict",
                         json={"anthropic_version": "vertex-2023-10-16", "stream": True}) as response:
            list(response.iter_lines())
    raw, streamed = _calls(patched)
    assert (raw.provider.value, raw.model) == ("google", "claude-sonnet-4-5@20250929")
    usage = raw.usage
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens, usage.cache_creation_tokens) == (50, 20, 400, 60)
    assert (streamed.model, streamed.streaming) == ("claude-sonnet-4-5@20250929", True)
    assert (streamed.usage.input_tokens, streamed.usage.output_tokens, streamed.usage.cache_read_tokens) == (50, 33, 400)


def test_vertex_openai_style_partner_endpoints(patched):
    base = "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1"
    _send(_json({"model": "mistral-large", "usage": {"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13}}),
          "POST", base + "/publishers/mistralai/models/mistral-large-2411@001:rawPredict",
          json={"model": "mistral-large", "messages": []})
    _send(_json({"model": "meta/llama-4-maverick", "usage": {"prompt_tokens": 7, "completion_tokens": 2,
                                                             "prompt_tokens_details": {"cached_tokens": 4}}}),
          "POST", base + "/endpoints/openapi/chat/completions",
          json={"model": "meta/llama-4-maverick-17b-128e-instruct-maas", "messages": []})
    mistral, llama = _calls(patched)
    assert (mistral.provider.value, mistral.model) == ("google", "mistral-large-2411@001")
    assert (mistral.usage.input_tokens, mistral.usage.output_tokens) == (8, 5)
    assert llama.model == "meta/llama-4-maverick"
    assert (llama.usage.input_tokens, llama.usage.output_tokens, llama.usage.cache_read_tokens) == (7, 2, 4)


# ── OpenAI audio ──

def test_openai_speech_counts_input_characters_without_reading_audio(patched):
    audio = lambda r: httpx.Response(200, content=AUDIO, headers={"content-type": "audio/mpeg"})
    _send(audio, "POST", "https://api.openai.com/v1/audio/speech",
          json={"model": "gpt-4o-mini-tts", "input": "Hello there", "voice": "alloy"})
    _send(audio, "POST", "https://res.openai.azure.com/openai/deployments/tts/audio/speech?api-version=2025-03-01",
          json={"model": "tts-1-hd", "input": "abc", "voice": "alloy"})
    # Some proxies label audio as JSON; the speech endpoint is still never parsed.
    _send(lambda r: httpx.Response(200, content=b"{\"not\": \"usage\"}", headers={"content-type": "application/json"}),
          "POST", "https://api.openai.com/v1/audio/speech", json={"model": "tts-1", "input": "x"})
    _send(_json({"error": {"message": "bad"}}, status=400), "POST", "https://api.openai.com/v1/audio/speech",
          json={"model": "tts-1", "input": "not billed"})
    speech, azure, mislabeled, failed = _calls(patched)
    assert (speech.provider.value, speech.model, speech.usage.native_units) == ("openai", "gpt-4o-mini-tts", {"characters": 11})
    assert (azure.model, azure.usage.native_units) == ("tts-1-hd", {"characters": 3})
    assert mislabeled.response_body is None and mislabeled.usage.native_units == {"characters": 1}
    assert failed.usage is None


def test_openai_transcription_usage_shapes(patched):
    url = "https://api.openai.com/v1/audio/transcriptions"
    upload = {"files": {"file": ("a.wav", b"RIFF" + b"\0" * 32)}}
    _send(_json({"text": "hi", "usage": {"type": "duration", "seconds": 61}}), "POST", url,
          data={"model": "whisper-1"}, **upload)
    _send(_json({"text": "hi", "usage": {"type": "tokens", "input_tokens": 70, "output_tokens": 12, "total_tokens": 82,
                                         "input_token_details": {"audio_tokens": 70, "text_tokens": 0}}}),
          "POST", url, data={"model": "gpt-4o-transcribe"}, **upload)
    _send(_json({"task": "transcribe", "language": "english", "duration": 8.47, "text": "hi"}), "POST",
          "https://api.openai.com/v1/audio/translations", data={"model": "whisper-1", "response_format": "verbose_json"}, **upload)
    _send(lambda r: httpx.Response(200, text="plain transcript", headers={"content-type": "text/plain"}),
          "POST", url, data={"model": "whisper-1", "response_format": "text"}, **upload)
    duration, tokens, verbose, plain = _calls(patched)
    assert (duration.model, duration.usage.native_units) == ("whisper-1", {"audioSeconds": 61})
    assert duration.usage.input_tokens is None
    assert (tokens.model, tokens.usage.input_tokens, tokens.usage.output_tokens) == ("gpt-4o-transcribe", 70, 12)
    assert not tokens.usage.native_units
    assert (verbose.model, verbose.usage.native_units) == ("whisper-1", {"audioSeconds": 8.47})
    assert plain.usage is None


# ── Deepgram, AssemblyAI, Cartesia ──

def test_deepgram_listen_and_speak(patched):
    _send(_json({"metadata": {"request_id": "r", "duration": 25.93, "channels": 1}, "results": {}}), "POST",
          "https://api.deepgram.com/v1/listen?model=nova-2-medical&smart_format=true",
          content=AUDIO, headers={"content-type": "audio/wav"})
    _send(_json({"metadata": {"duration": 3.0}}), "POST", "https://api.deepgram.com/v1/listen",
          json={"url": "https://example.com/a.wav"})
    _send(lambda r: httpx.Response(200, content=AUDIO, headers={"content-type": "audio/mpeg"}), "POST",
          "https://api.deepgram.com/v1/speak?model=aura-2-thalia-en", json={"text": "Good morning"})
    listen, default_model, speak = _calls(patched)
    assert (listen.provider.value, listen.model, listen.usage.native_units) == ("deepgram", "nova-2-medical", {"audioSeconds": 25.93})
    assert "model=" not in listen.url, "queries never reach telemetry"
    assert (default_model.model, default_model.usage.native_units) == ("nova-3", {"audioSeconds": 3.0})
    assert (speak.model, speak.usage.native_units) == ("aura-2-thalia-en", {"characters": 12})


@needs_aiohttp
def test_deepgram_model_from_aiohttp_params(patched):
    from aiohttp import web

    async def handler(request):
        assert request.query["model"] == "nova-3-general"
        return web.json_response({"metadata": {"duration": 9.5}})

    async def run():
        async with _provider_server(handler) as port, _session() as session:
            async with session.post(f"http://api.deepgram.com:{port}/v1/listen", data=b"RIFF",
                                    params={"model": "nova-3-general"}) as response:
                await response.json()

    asyncio.run(run())
    event = _calls(patched)[0]
    assert (event.model, event.usage.native_units) == ("nova-3-general", {"audioSeconds": 9.5})


def test_assemblyai_bills_only_the_completed_transcript_get(patched):
    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json={"id": "t1", "status": "queued", "speech_model": None})
        if request.url.path == "/v2/transcript/t1":
            status = "completed" if request.method in ("GET", "DELETE") else "processing"
            return httpx.Response(200, json={"id": "t1", "status": status, "audio_duration": 281,
                                             "speech_model": "universal"})
        return httpx.Response(200, json={"id": "t2", "status": "processing", "audio_duration": None})

    _send(handler, "POST", "https://api.assemblyai.com/v2/transcript", json={"audio_url": "https://a"})
    _send(handler, "GET", "https://api.assemblyai.com/v2/transcript/t2")
    _send(handler, "GET", "https://api.assemblyai.com/v2/transcript/t1")
    _send(handler, "DELETE", "https://api.assemblyai.com/v2/transcript/t1")
    created, polling, completed, deleted = _calls(patched)
    assert (created.provider.value, created.model, created.usage) == ("assemblyai", "best", None)
    assert polling.usage is None
    assert (completed.model, completed.usage.native_units) == ("universal", {"audioSeconds": 281})
    assert deleted.usage is None
    assert call_usage(find_extractor("https://api.assemblyai.com/v2/transcript/t1"),
                      "https://api.assemblyai.com/v2/transcript/t1", None,
                      {"status": "completed", "audio_duration": 4}, None, 200) is None, "no method, no usage"


@needs_requests
def test_assemblyai_completed_get_over_requests(patched):
    from costkey import patch as patch_module
    patch_module._patch_requests()
    body = json.dumps({"id": "t1", "status": "completed", "audio_duration": 12.5, "speech_model": "slam-1"}).encode()
    session = _requests_session(200, {"content-type": "application/json"}, body)
    session.get("https://api.assemblyai.com/v2/transcript/t1")
    event = _calls(patched)[0]
    assert (event.model, event.usage.native_units) == ("slam-1", {"audioSeconds": 12.5})


def test_cartesia_counts_transcript_and_does_not_buffer_sse_audio(patched):
    _send(lambda r: httpx.Response(200, content=AUDIO, headers={"content-type": "audio/wav"}), "POST",
          "https://api.cartesia.ai/tts/bytes", json={"model_id": "sonic-2", "transcript": "Welcome back"})
    sse_audio = _sse({"type": "chunk", "data": "UklGR" * 4000}, {"type": "done"})
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(
        200, content=sse_audio, headers={"content-type": "text/event-stream"},
    ))) as http:
        with http.stream("POST", "https://api.cartesia.ai/tts/sse",
                         json={"model_id": "sonic-turbo", "transcript": "Hi"}) as response:
            assert b"".join(response.iter_bytes()) == sse_audio
    _send(_json({"voices": []}), "GET", "https://api.cartesia.ai/voices")
    tts, sse, voices = _calls(patched)
    assert (tts.provider.value, tts.model, tts.usage.native_units) == ("cartesia", "sonic-2", {"characters": 12})
    assert (sse.model, sse.usage.native_units, sse.response_body) == ("sonic-turbo", {"characters": 2}, None)
    assert voices.usage is None


# ── Embeddings, rerank and search ──

def test_voyage_and_jina_total_tokens_are_input(patched):
    _send(_json({"object": "list", "data": [], "model": "voyage-3.5", "usage": {"total_tokens": 42}}), "POST",
          "https://api.voyageai.com/v1/embeddings", json={"model": "voyage-3.5", "input": ["a"]})
    _send(_json({"data": [], "usage": {"total_tokens": 300}}), "POST",
          "https://api.voyageai.com/v1/rerank", json={"model": "rerank-2.5", "query": "q", "documents": []})
    _send(_json({"data": [], "usage": {"text_tokens": 5, "image_pixels": 2000, "total_tokens": 1005}}), "POST",
          "https://api.voyageai.com/v1/multimodalembeddings", json={"model": "voyage-multimodal-3", "inputs": []})
    _send(_json({"model": "jina-embeddings-v3", "usage": {"total_tokens": 17, "prompt_tokens": 17}}), "POST",
          "https://api.jina.ai/v1/embeddings", json={"model": "jina-embeddings-v3", "input": ["a"]})
    _send(_json({"results": [], "usage": {"total_tokens": 90}}), "POST",
          "https://api.jina.ai/v1/rerank", json={"model": "jina-reranker-v2-base-multilingual", "query": "q"})
    embed, rerank, multimodal, jina_embed, jina_rerank = _calls(patched)
    assert (embed.provider.value, embed.model, embed.usage.input_tokens, embed.usage.total_tokens) == ("voyage", "voyage-3.5", 42, 42)
    assert (rerank.model, rerank.usage.input_tokens, rerank.usage.output_tokens) == ("rerank-2.5", 300, None)
    assert (multimodal.usage.input_tokens, multimodal.usage.native_units) == (1005, {})
    assert (jina_embed.provider.value, jina_embed.usage.input_tokens) == ("jina", 17)
    assert (jina_rerank.model, jina_rerank.usage.input_tokens) == ("jina-reranker-v2-base-multilingual", 90)


def test_cohere_billed_units_for_embed_rerank_and_v2_chat(patched):
    _send(_json({"id": "e1", "embeddings": {"float": []}, "meta": {"billed_units": {"input_tokens": 64}}}), "POST",
          "https://api.cohere.com/v2/embed", json={"model": "embed-v4.0", "texts": ["a"]})
    _send(_json({"id": "r1", "results": [], "meta": {"billed_units": {"search_units": 2}}}), "POST",
          "https://api.cohere.com/v2/rerank", json={"model": "rerank-v3.5", "query": "q", "documents": []})
    _send(_json({"id": "c1", "message": {}, "usage": {"billed_units": {"input_tokens": 5, "output_tokens": 9},
                                                      "tokens": {"input_tokens": 71, "output_tokens": 9}}}),
          "POST", "https://api.cohere.com/v2/chat", json={"model": "command-a-03-2025", "messages": []})
    embed, rerank, chat = _calls(patched)
    assert (embed.model, embed.usage.input_tokens, embed.usage.output_tokens) == ("embed-v4.0", 64, None)
    assert (rerank.model, rerank.usage.native_units) == ("rerank-v3.5", {"searchQueries": 2})
    assert (chat.usage.input_tokens, chat.usage.output_tokens, chat.usage.total_tokens) == (71, 9, 80)


def test_cohere_v2_stream_message_end_usage():
    from costkey import patch as patch_module
    text = _sse({"type": "content-delta", "delta": {"message": {"content": {"text": "hi"}}}},
                {"type": "message-end", "delta": {"finish_reason": "COMPLETE", "usage": {
                    "billed_units": {"input_tokens": 3, "output_tokens": 4},
                    "tokens": {"input_tokens": 60, "output_tokens": 4}}}}).decode()
    usage = patch_module._extract_sse_usage(text, find_extractor("https://api.cohere.com/v2/chat"))
    assert (usage.input_tokens, usage.output_tokens) == (60, 4)


def test_tavily_search_depth_queries_and_credits(patched):
    def handler(request):
        body = json.loads(request.content)
        usage = {"usage": {"credits": 2}} if body.get("include_usage") else {}
        return httpx.Response(200, json={"results": [], **usage})

    _send(handler, "POST", "https://api.tavily.com/search", json={"query": "x"})
    _send(handler, "POST", "https://api.tavily.com/search",
          json={"query": "x", "search_depth": "advanced", "include_usage": True})
    _send(handler, "POST", "https://api.tavily.com/extract", json={"urls": ["https://a"], "include_usage": True})
    _send(handler, "POST", "https://api.tavily.com/extract", json={"urls": ["https://a"]})
    _send(_json({"detail": "limit"}, status=429), "POST", "https://api.tavily.com/search", json={"query": "x"})
    basic, advanced, extract, extract_unreported, failed = _calls(patched)
    assert (basic.provider.value, basic.model, basic.usage.native_units) == ("tavily", "search", {"searchQueries": 1})
    assert (advanced.model, advanced.usage.native_units) == ("search-advanced", {"searchQueries": 1, "credits": 2})
    assert (extract.model, extract.usage.native_units) == ("extract", {"credits": 2})
    assert extract_unreported.usage is None
    assert failed.usage is None


@needs_requests
def test_tavily_over_requests(patched):
    from costkey import patch as patch_module
    patch_module._patch_requests()
    session = _requests_session(200, {"content-type": "application/json"},
                                json.dumps({"results": [], "usage": {"credits": 1}}).encode())
    session.post("https://api.tavily.com/search", json={"query": "x", "api_key": "tvly-secret", "include_usage": True})
    event = _calls(patched)[0]
    assert (event.model, event.usage.native_units) == ("search", {"searchQueries": 1, "credits": 1})
    assert event.request_body["api_key"] == "[REDACTED]"
    assert _wire(event)["usage"]["nativeUnits"] == {"searchQueries": 1, "credits": 1}


def test_request_derived_models_without_a_response():
    assert call_model(find_extractor("https://api.deepgram.com/v1/speak"), "https://api.deepgram.com/v1/speak",
                      {"text": "x"}, None) == "aura-asteria-en"
    assert call_model(find_extractor("https://api.sarvam.ai/text-to-speech"), "https://api.sarvam.ai/text-to-speech",
                      {"inputs": ["x"]}, None) == "bulbul:v2"
