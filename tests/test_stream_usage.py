"""Streamed chat calls ask the provider to report usage, like the TS SDK."""
import json

import httpx

from costkey import providers


def test_openai_chat_stream_asks_for_usage():
    body = {"model": "gpt-5", "messages": [], "stream": True}
    assert providers.inject_stream_usage("https://api.openai.com/v1/chat/completions", body)["stream_options"] == {"include_usage": True}


def test_keeps_existing_stream_options():
    body = {"model": "gpt-5", "messages": [], "stream": True, "stream_options": {"x": 1}}
    out = providers.inject_stream_usage("https://api.openai.com/v1/chat/completions", body)
    assert out["stream_options"] == {"x": 1, "include_usage": True}


def test_leaves_other_requests_alone():
    plain = {"model": "gpt-5", "messages": []}
    assert providers.inject_stream_usage("https://api.openai.com/v1/chat/completions", plain) is plain
    responses = {"model": "gpt-5", "input": "x", "stream": True}
    assert providers.inject_stream_usage("https://api.openai.com/v1/responses", responses) is responses
    anthropic = {"model": "claude", "messages": [], "stream": True}
    assert providers.inject_stream_usage("https://api.anthropic.com/v1/messages", anthropic) is anthropic


def test_openrouter_always_and_azure_chat():
    assert providers.stream_usage_mode("https://openrouter.ai/api/v1/chat/completions") == "always"
    assert providers.stream_usage_mode("https://x.openai.azure.com/openai/deployments/d/chat/completions") == "chat"


def test_bytes_bodies():
    raw = json.dumps({"model": "gpt-5", "messages": [], "stream": True}).encode()
    out = providers.inject_stream_usage_bytes("https://api.openai.com/v1/chat/completions", raw)
    assert json.loads(out)["stream_options"]["include_usage"] is True
    assert providers.inject_stream_usage_bytes("https://api.openai.com/v1/chat/completions", b"not json") == b"not json"


def test_httpx_request_is_rewritten():
    from costkey.patch import _strip_httpx_context_headers
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions",
                            json={"model": "gpt-5", "messages": [], "stream": True})
    sent = _strip_httpx_context_headers(request)
    assert json.loads(sent.content)["stream_options"] == {"include_usage": True}
    assert int(sent.headers["content-length"]) == len(sent.content)
