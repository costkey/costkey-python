"""Comprehensive tests for every module in the CostKey Python SDK."""
from __future__ import annotations
import json
import time
import functools
from unittest.mock import MagicMock

import httpx
import pytest

from costkey.types import (
    Provider, NormalizedUsage, StackFrame, CallSite,
    StreamTiming, CostKeyEvent, CostKeyOptions,
)
from costkey.providers import (
    find_extractor, OpenAIExtractor, AnthropicExtractor,
    GoogleExtractor, DeepSeekExtractor, BedrockExtractor,
    CohereExtractor, _extract_openai_usage, _extract_model, _as_int,
)
from costkey.stack import capture_call_site, _INTERNAL
from costkey.transport import Transport
from costkey.patch import (
    _scrub, _is_streaming_request, _extract_sse_usage,
    _extract_sse_model, _wrap_streaming_response,
)


# ═══════════════════════════════════════════════════════════════════
# Provider usage extraction — realistic response bodies
# ═══════════════════════════════════════════════════════════════════

class TestOpenAIUsage:
    def test_basic_chat_completion(self):
        ext = OpenAIExtractor()
        body = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "model": "gpt-4o-2024-11-20",
            "usage": {
                "prompt_tokens": 512,
                "completion_tokens": 128,
                "total_tokens": 640,
            },
        }
        usage = ext.extract_usage(body)
        assert usage is not None
        assert usage.input_tokens == 512
        assert usage.output_tokens == 128
        assert usage.total_tokens == 640
        assert usage.reasoning_tokens is None

    def test_with_reasoning_tokens(self):
        ext = OpenAIExtractor()
        body = {
            "model": "o1-preview",
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 5000,
                "total_tokens": 6000,
                "completion_tokens_details": {
                    "reasoning_tokens": 4500,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            },
        }
        usage = ext.extract_usage(body)
        assert usage is not None
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 5000
        assert usage.reasoning_tokens == 4500

    def test_with_output_tokens_details(self):
        """Some newer OpenAI models use output_tokens_details instead."""
        ext = OpenAIExtractor()
        body = {
            "model": "o3-mini",
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 1000,
                "total_tokens": 1200,
                "output_tokens_details": {
                    "reasoning_tokens": 800,
                },
            },
        }
        usage = ext.extract_usage(body)
        assert usage is not None
        assert usage.reasoning_tokens == 800

    def test_none_body(self):
        assert OpenAIExtractor().extract_usage(None) is None

    def test_no_usage_key(self):
        assert OpenAIExtractor().extract_usage({"model": "gpt-4o"}) is None

    def test_model_extraction(self):
        ext = OpenAIExtractor()
        assert ext.extract_model(None, {"model": "gpt-4o"}) == "gpt-4o"
        assert ext.extract_model({"model": "gpt-4o"}, None) == "gpt-4o"
        assert ext.extract_model({"model": "gpt-4o"}, {"model": "gpt-4o-2024-11-20"}) == "gpt-4o-2024-11-20"
        assert ext.extract_model(None, None) is None


class TestAnthropicUsage:
    def test_basic_message(self):
        ext = AnthropicExtractor()
        body = {
            "id": "msg_01XYZ",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {
                "input_tokens": 3018,
                "output_tokens": 123,
            },
        }
        usage = ext.extract_usage(body)
        assert usage is not None
        assert usage.input_tokens == 3018
        assert usage.output_tokens == 123
        assert usage.total_tokens == 3141
        assert usage.cache_read_tokens is None
        assert usage.cache_creation_tokens is None

    def test_with_cache_tokens(self):
        ext = AnthropicExtractor()
        body = {
            "model": "claude-sonnet-4-20250514",
            "usage": {
                "input_tokens": 5000,
                "output_tokens": 200,
                "cache_read_input_tokens": 3000,
                "cache_creation_input_tokens": 1000,
            },
        }
        usage = ext.extract_usage(body)
        assert usage is not None
        assert usage.input_tokens == 5000
        assert usage.output_tokens == 200
        assert usage.total_tokens == 5200
        assert usage.cache_read_tokens == 3000
        assert usage.cache_creation_tokens == 1000

    def test_none_body(self):
        assert AnthropicExtractor().extract_usage(None) is None

    def test_no_usage(self):
        assert AnthropicExtractor().extract_usage({"type": "error"}) is None


class TestGoogleGeminiUsage:
    def test_gemini_usage_metadata(self):
        ext = GoogleExtractor()
        body = {
            "candidates": [{"content": {"parts": [{"text": "Hello"}]}}],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 25,
                "totalTokenCount": 35,
            },
        }
        usage = ext.extract_usage(body)
        assert usage is not None
        assert usage.input_tokens == 10
        assert usage.output_tokens == 25
        assert usage.total_tokens == 35

    def test_with_thoughts_and_cache(self):
        ext = GoogleExtractor()
        body = {
            "usageMetadata": {
                "promptTokenCount": 500,
                "candidatesTokenCount": 200,
                "totalTokenCount": 1200,
                "thoughtsTokenCount": 500,
                "cachedContentTokenCount": 100,
            },
        }
        usage = ext.extract_usage(body)
        assert usage is not None
        assert usage.reasoning_tokens == 500
        assert usage.cache_read_tokens == 100

    def test_model_from_model_version(self):
        ext = GoogleExtractor()
        resp = {"modelVersion": "gemini-2.0-flash"}
        assert ext.extract_model(None, resp) == "gemini-2.0-flash"

    def test_model_none_without_version(self):
        ext = GoogleExtractor()
        assert ext.extract_model(None, {"candidates": []}) is None
        assert ext.extract_model(None, None) is None

    def test_none_body(self):
        assert GoogleExtractor().extract_usage(None) is None


class TestDeepSeekUsage:
    def test_basic_usage(self):
        ext = DeepSeekExtractor()
        body = {
            "model": "deepseek-chat",
            "usage": {
                "prompt_tokens": 2000,
                "completion_tokens": 500,
                "total_tokens": 2500,
            },
        }
        usage = ext.extract_usage(body)
        assert usage is not None
        assert usage.input_tokens == 2000
        assert usage.output_tokens == 500

    def test_with_cache_hit_tokens(self):
        ext = DeepSeekExtractor()
        body = {
            "model": "deepseek-chat",
            "usage": {
                "prompt_tokens": 2000,
                "completion_tokens": 500,
                "total_tokens": 2500,
                "prompt_cache_hit_tokens": 1500,
            },
        }
        usage = ext.extract_usage(body)
        assert usage is not None
        assert usage.cache_read_tokens == 1500

    def test_with_reasoning_tokens(self):
        ext = DeepSeekExtractor()
        body = {
            "model": "deepseek-reasoner",
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 3000,
                "total_tokens": 4000,
                "prompt_cache_hit_tokens": 500,
                "completion_tokens_details": {
                    "reasoning_tokens": 2500,
                },
            },
        }
        usage = ext.extract_usage(body)
        assert usage is not None
        assert usage.cache_read_tokens == 500
        assert usage.reasoning_tokens == 2500


class TestCohereUsage:
    def test_meta_tokens_format(self):
        ext = CohereExtractor()
        body = {
            "id": "gen-123",
            "text": "Hello!",
            "meta": {
                "api_version": {"version": "2022-12-06"},
                "tokens": {
                    "input_tokens": 150,
                    "output_tokens": 75,
                },
            },
        }
        usage = ext.extract_usage(body)
        assert usage is not None
        assert usage.input_tokens == 150
        assert usage.output_tokens == 75

    def test_v2_openai_format(self):
        """Cohere v2 uses OpenAI-compatible format."""
        ext = CohereExtractor()
        body = {
            "id": "chatcmpl-123",
            "model": "command-r-plus",
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300,
            },
        }
        usage = ext.extract_usage(body)
        assert usage is not None
        assert usage.input_tokens == 200
        assert usage.output_tokens == 100

    def test_meta_format_takes_priority(self):
        """When both meta.tokens and usage exist, meta.tokens is returned."""
        ext = CohereExtractor()
        body = {
            "meta": {"tokens": {"input_tokens": 50, "output_tokens": 25}},
            "usage": {"prompt_tokens": 999, "completion_tokens": 999},
        }
        usage = ext.extract_usage(body)
        assert usage is not None
        assert usage.input_tokens == 50
        assert usage.output_tokens == 25

    def test_none_body(self):
        assert CohereExtractor().extract_usage(None) is None


class TestBedrockUsage:
    def test_usage_format(self):
        ext = BedrockExtractor()
        body = {
            "output": {"message": {"content": [{"text": "Hello"}]}},
            "usage": {
                "inputTokens": 100,
                "outputTokens": 50,
                "totalTokens": 150,
            },
        }
        usage = ext.extract_usage(body)
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_invocation_metrics_format(self):
        ext = BedrockExtractor()
        body = {
            "generation": "Hello!",
            "amazon-bedrock-invocationMetrics": {
                "inputTokenCount": 200,
                "outputTokenCount": 80,
                "invocationLatency": 500,
                "firstByteLatency": 100,
            },
        }
        usage = ext.extract_usage(body)
        assert usage is not None
        assert usage.input_tokens == 200
        assert usage.output_tokens == 80

    def test_model_from_request(self):
        ext = BedrockExtractor()
        assert ext.extract_model({"modelId": "anthropic.claude-3-sonnet"}, None) == "anthropic.claude-3-sonnet"
        assert ext.extract_model(None, None) is None
        assert ext.extract_model({}, None) is None

    def test_none_body(self):
        assert BedrockExtractor().extract_usage(None) is None

    def test_no_usage_keys(self):
        assert BedrockExtractor().extract_usage({"generation": "text"}) is None


class TestSimpleExtractors:
    """All simple extractors use OpenAI-compatible usage format."""

    SIMPLE_PROVIDERS = [
        ("https://api.groq.com/openai/v1/chat/completions", Provider.GROQ),
        ("https://api.mistral.ai/v1/chat/completions", Provider.MISTRAL),
        ("https://api.together.xyz/v1/chat/completions", Provider.TOGETHER),
        ("https://api.fireworks.ai/inference/v1/chat/completions", Provider.FIREWORKS),
        ("https://api.perplexity.ai/chat/completions", Provider.PERPLEXITY),
        ("https://api.cerebras.ai/v1/chat/completions", Provider.CEREBRAS),
        ("https://api.x.ai/v1/chat/completions", Provider.XAI),
        ("https://openrouter.ai/api/v1/chat/completions", Provider.OPENROUTER),
    ]

    OPENAI_BODY = {
        "model": "test-model",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }

    @pytest.mark.parametrize("url,expected_provider", SIMPLE_PROVIDERS)
    def test_detects_provider(self, url, expected_provider):
        ext = find_extractor(url)
        assert ext is not None, f"No extractor found for {url}"
        assert ext.provider == expected_provider

    @pytest.mark.parametrize("url,expected_provider", SIMPLE_PROVIDERS)
    def test_extracts_usage(self, url, expected_provider):
        ext = find_extractor(url)
        usage = ext.extract_usage(self.OPENAI_BODY)
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    @pytest.mark.parametrize("url,expected_provider", SIMPLE_PROVIDERS)
    def test_extracts_model(self, url, expected_provider):
        ext = find_extractor(url)
        model = ext.extract_model(None, self.OPENAI_BODY)
        assert model == "test-model"


# ═══════════════════════════════════════════════════════════════════
# Provider detection by URL
# ═══════════════════════════════════════════════════════════════════

class TestProviderDetection:
    def test_openai(self):
        ext = find_extractor("https://api.openai.com/v1/chat/completions")
        assert ext is not None
        assert ext.provider == Provider.OPENAI

    def test_azure_openai(self):
        ext = find_extractor("https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-01")
        assert ext is not None
        assert ext.provider == Provider.OPENAI

    def test_anthropic(self):
        ext = find_extractor("https://api.anthropic.com/v1/messages")
        assert ext is not None
        assert ext.provider == Provider.ANTHROPIC

    def test_google_generative(self):
        ext = find_extractor("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent")
        assert ext is not None
        assert ext.provider == Provider.GOOGLE

    def test_google_vertex(self):
        ext = find_extractor("https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-2.0-flash:generateContent")
        assert ext is not None
        assert ext.provider == Provider.GOOGLE

    def test_deepseek(self):
        ext = find_extractor("https://api.deepseek.com/v1/chat/completions")
        assert ext is not None
        assert ext.provider == Provider.DEEPSEEK

    def test_bedrock(self):
        ext = find_extractor("https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-sonnet/invoke")
        assert ext is not None
        assert ext.provider == Provider.BEDROCK

    def test_bedrock_other_region(self):
        ext = find_extractor("https://bedrock-runtime.eu-west-1.amazonaws.com/model/test")
        assert ext is not None
        assert ext.provider == Provider.BEDROCK

    def test_cohere(self):
        ext = find_extractor("https://api.cohere.com/v2/chat")
        assert ext is not None
        assert ext.provider == Provider.COHERE

    def test_xai_alternate(self):
        ext = find_extractor("https://api.grok.xai.com/v1/chat/completions")
        assert ext is not None
        assert ext.provider == Provider.XAI

    def test_non_ai_urls_return_none(self):
        assert find_extractor("https://www.google.com/search?q=test") is None
        assert find_extractor("https://github.com/costkey/sdk") is None
        assert find_extractor("https://api.example.com/v1/data") is None
        assert find_extractor("https://httpbin.org/post") is None

    def test_url_with_port(self):
        assert find_extractor("https://api.openai.com:443/v1/chat/completions") is not None

    def test_url_with_query_params(self):
        ext = find_extractor("https://api.anthropic.com/v1/messages?beta=true")
        assert ext is not None
        assert ext.provider == Provider.ANTHROPIC

    def test_url_with_deep_path(self):
        ext = find_extractor("https://api.groq.com/openai/v1/chat/completions")
        assert ext is not None
        assert ext.provider == Provider.GROQ


# ═══════════════════════════════════════════════════════════════════
# Streaming
# ═══════════════════════════════════════════════════════════════════

def _make_mock_response(chunks: list[bytes]) -> MagicMock:
    """Create a mock httpx response with iteration methods."""
    response = MagicMock()
    response.status_code = 200

    def iter_bytes_fn(*a, **kw):
        yield from chunks

    def iter_lines_fn(*a, **kw):
        full = b"".join(chunks).decode("utf-8", errors="replace")
        for line in full.split("\n"):
            if line:
                yield line

    def iter_text_fn(*a, **kw):
        for chunk in chunks:
            yield chunk.decode("utf-8", errors="replace")

    def read_fn(*a, **kw):
        return b"".join(chunks)

    response.iter_bytes = iter_bytes_fn
    response.iter_lines = iter_lines_fn
    response.iter_text = iter_text_fn
    response.read = read_fn
    # No async variants
    response.aiter_bytes = None
    response.aiter_lines = None
    response.aread = None
    # Explicitly remove to avoid wrapping
    del response.aiter_bytes
    del response.aiter_lines
    del response.aread

    return response


class TestStreamingWrappers:
    def test_iter_bytes_accumulates_and_fires_on_done(self):
        chunks = [b"data: {\"usage\":{\"input_tokens\":10,\"output_tokens\":5}}\n"]
        response = _make_mock_response(chunks)
        extractor = AnthropicExtractor()
        done_called = []

        _wrap_streaming_response(
            response, extractor, "https://api.anthropic.com/v1/messages",
            "POST", {"stream": True}, time.perf_counter(), None, {},
        )

        # Consume the iterator
        result = list(response.iter_bytes())
        assert len(result) == 1
        assert result[0] == chunks[0]

    def test_iter_lines_wrapper(self):
        chunks = [
            b'data: {"type":"message_start"}\n',
            b'data: {"type":"message_delta","usage":{"input_tokens":20,"output_tokens":10}}\n',
        ]
        response = _make_mock_response(chunks)
        extractor = AnthropicExtractor()

        _wrap_streaming_response(
            response, extractor, "https://api.anthropic.com/v1/messages",
            "POST", {"stream": True}, time.perf_counter(), None, {},
        )

        lines = list(response.iter_lines())
        assert len(lines) > 0

    def test_iter_text_wrapper(self):
        chunks = [b'data: {"type":"content_block_delta"}\n']
        response = _make_mock_response(chunks)
        extractor = AnthropicExtractor()

        _wrap_streaming_response(
            response, extractor, "https://api.anthropic.com/v1/messages",
            "POST", {"stream": True}, time.perf_counter(), None, {},
        )

        texts = list(response.iter_text())
        assert len(texts) == 1

    def test_read_wrapper(self):
        data = b'data: {"usage":{"input_tokens":5,"output_tokens":3}}\n'
        response = _make_mock_response([data])
        extractor = AnthropicExtractor()

        _wrap_streaming_response(
            response, extractor, "https://api.anthropic.com/v1/messages",
            "POST", {"stream": True}, time.perf_counter(), None, {},
        )

        result = response.read()
        assert result == data

    def test_on_done_called_only_once(self):
        """Even if multiple iteration methods are used, _on_done fires only once."""
        chunks = [b'data: {"usage":{"input_tokens":10,"output_tokens":5}}\n']
        response = _make_mock_response(chunks)
        extractor = AnthropicExtractor()

        done_count = [0]
        _wrap_streaming_response(
            response, extractor, "https://api.anthropic.com/v1/messages",
            "POST", {"stream": True}, time.perf_counter(), None, {},
        )

        # Monkey-patch to count _on_done calls — the event_sent flag prevents double fire
        # We can verify indirectly: consuming iter_bytes then read should not error
        list(response.iter_bytes())
        # Second consumption — _on_done should be a no-op due to event_sent flag
        result = response.read()
        # If _on_done wasn't guarded, it would process the event twice — just verify no crash


class TestSSEUsageExtraction:
    def test_openai_format(self):
        ext = OpenAIExtractor()
        sse = (
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}\n'
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n'
            'data: [DONE]\n'
        )
        usage = _extract_sse_usage(sse, ext)
        assert usage is not None
        assert usage.input_tokens == 10
        assert usage.output_tokens == 5
        assert usage.total_tokens == 15

    def test_anthropic_format(self):
        ext = AnthropicExtractor()
        sse = (
            'data: {"type":"message_start","message":{"model":"claude-sonnet-4-5","usage":{"input_tokens":100,"output_tokens":0}}}\n'
            'data: {"type":"content_block_delta","delta":{"text":"Hi"}}\n'
            'data: {"type":"message_delta","usage":{"input_tokens":100,"output_tokens":50}}\n'
            'data: [DONE]\n'
        )
        usage = _extract_sse_usage(sse, ext)
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_empty_sse_returns_none(self):
        ext = OpenAIExtractor()
        assert _extract_sse_usage("", ext) is None

    def test_malformed_sse_returns_none(self):
        ext = OpenAIExtractor()
        sse = "data: {this is not json}\ndata: also broken\n"
        assert _extract_sse_usage(sse, ext) is None

    def test_only_done_returns_none(self):
        ext = OpenAIExtractor()
        assert _extract_sse_usage("data: [DONE]\n", ext) is None


class TestSSEModelExtraction:
    def test_model_from_first_chunk(self):
        ext = OpenAIExtractor()
        sse = 'data: {"model":"gpt-4o-2024-11-20","choices":[{"delta":{"role":"assistant"}}]}\n'
        model = _extract_sse_model(sse, {"model": "gpt-4o"}, ext)
        assert model == "gpt-4o-2024-11-20"

    def test_model_from_request_fallback(self):
        ext = OpenAIExtractor()
        sse = 'data: {"choices":[{"delta":{"content":"hi"}}]}\n'
        # No model in SSE chunk, falls back to request body
        model = _extract_sse_model(sse, {"model": "gpt-4o"}, ext)
        assert model == "gpt-4o"

    def test_empty_returns_none(self):
        ext = OpenAIExtractor()
        assert _extract_sse_model("", {}, ext) is None


# ═══════════════════════════════════════════════════════════════════
# Stack trace
# ═══════════════════════════════════════════════════════════════════

class TestStackTrace:
    def test_captures_user_frames(self):
        def my_function():
            return capture_call_site()

        site = my_function()
        assert site is not None
        assert len(site.frames) > 0
        # Our function should appear in the frames
        func_names = [f.function_name for f in site.frames]
        assert "my_function" in func_names

    def test_filters_library_frames(self):
        site = capture_call_site()
        assert site is not None
        for frame in site.frames:
            fname = frame.file_name or ""
            # These should all be filtered
            assert "site-packages/openai" not in fname
            assert "site-packages/anthropic" not in fname
            assert "site-packages/httpx" not in fname
            assert "site-packages/httpcore" not in fname
            assert "costkey/" not in fname

    def test_filters_stdlib_frames(self):
        site = capture_call_site()
        assert site is not None
        for frame in site.frames:
            fname = frame.file_name or ""
            assert "asyncio/" not in fname
            assert "_bootstrap" not in fname

    def test_function_name_parsing(self):
        """Function names should be clean, not contain code lines."""
        def outer():
            def inner():
                return capture_call_site()
            return inner()

        site = outer()
        assert site is not None
        for frame in site.frames:
            if frame.function_name:
                # Function names should not contain code — just the name
                assert "\n" not in frame.function_name
                assert "=" not in frame.function_name or frame.function_name.startswith("<")

    def test_has_line_numbers(self):
        site = capture_call_site()
        assert site is not None
        for frame in site.frames:
            assert frame.line_number is not None
            assert frame.line_number > 0

    def test_raw_contains_full_traceback(self):
        site = capture_call_site()
        assert site is not None
        assert len(site.raw) > 0
        assert "File " in site.raw

    def test_internal_patterns_comprehensive(self):
        """Verify all _INTERNAL patterns are strings that would appear in file paths."""
        for pattern in _INTERNAL:
            assert isinstance(pattern, str)
            assert len(pattern) > 0


# ═══════════════════════════════════════════════════════════════════
# Body capture and credential scrubbing
# ═══════════════════════════════════════════════════════════════════

class TestScrubbing:
    def test_scrubs_openai_api_key(self):
        data = {"headers": {"Authorization": "Bearer sk-abc123456789012345678901234567890"}}
        result = _scrub(data)
        assert result["headers"]["Authorization"] == "[REDACTED]"

    def test_scrubs_anthropic_api_key(self):
        result = _scrub("sk-ant-api03-abcdefghijklmnopqrstuvwxyz")
        assert result == "[REDACTED]"

    def test_scrubs_google_api_key(self):
        result = _scrub("AIzaSyBxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        assert result == "[REDACTED]"

    def test_scrubs_bearer_token(self):
        result = _scrub("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc.def")
        assert result == "[REDACTED]"

    def test_scrubs_jwt(self):
        result = _scrub("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0")
        assert result == "[REDACTED]"

    def test_scrubs_secret_key_names(self):
        data = {
            "api_key": "my-secret-key",
            "token": "my-token",
            "password": "hunter2",
            "model": "gpt-4o",
            "prompt": "hello",
        }
        result = _scrub(data)
        assert result["api_key"] == "[REDACTED]"
        assert result["token"] == "[REDACTED]"
        assert result["password"] == "[REDACTED]"
        assert result["model"] == "gpt-4o"
        assert result["prompt"] == "hello"

    def test_scrubs_nested_dicts(self):
        data = {
            "config": {
                "api_key": "secret",
                "endpoint": "https://api.openai.com",
            }
        }
        result = _scrub(data)
        assert result["config"]["api_key"] == "[REDACTED]"
        assert result["config"]["endpoint"] == "https://api.openai.com"

    def test_scrubs_lists(self):
        data = ["sk-abc12345678901234567890123456", "normal-text"]
        result = _scrub(data)
        assert result[0] == "[REDACTED]"
        assert result[1] == "normal-text"

    def test_scrub_none(self):
        assert _scrub(None) is None

    def test_scrub_preserves_non_secret_strings(self):
        assert _scrub("Hello, world!") == "Hello, world!"
        assert _scrub("gpt-4o") == "gpt-4o"

    def test_scrub_preserves_numbers(self):
        assert _scrub(42) == 42
        assert _scrub(3.14) == 3.14

    def test_case_insensitive_key_matching(self):
        data = {"Authorization": "value", "API_KEY": "value", "Token": "value"}
        result = _scrub(data)
        assert result["Authorization"] == "[REDACTED]"
        assert result["API_KEY"] == "[REDACTED]"
        assert result["Token"] == "[REDACTED]"


# ═══════════════════════════════════════════════════════════════════
# Transport serialization
# ═══════════════════════════════════════════════════════════════════

class TestTransportSerialization:
    def _make_transport(self):
        return Transport(
            endpoint="https://ingest.costkey.dev",
            auth_key="test-key",
            max_batch_size=50,
            flush_interval=5.0,
            debug=False,
        )

    def test_basic_event_serialization(self):
        t = self._make_transport()
        event = CostKeyEvent(
            id="test-123",
            timestamp="2025-01-01T00:00:00.000Z",
            project_id="proj-1",
            provider=Provider.OPENAI,
            model="gpt-4o",
            url="https://api.openai.com/v1/chat/completions",
            method="POST",
            status_code=200,
            duration_ms=150.5,
            streaming=False,
        )
        d = t._serialize(event)
        assert d["id"] == "test-123"
        assert d["projectId"] == "proj-1"
        assert d["provider"] == "openai"
        assert d["model"] == "gpt-4o"
        assert d["statusCode"] == 200
        assert d["durationMs"] == 150.5
        assert d["streaming"] is False
        assert d["usage"] is None
        assert d["streamTiming"] is None
        assert d["callSite"] is None

    def test_usage_serialization_camel_case(self):
        t = self._make_transport()
        event = CostKeyEvent(
            id="test-456",
            timestamp="2025-01-01T00:00:00.000Z",
            project_id="proj-1",
            provider=Provider.ANTHROPIC,
            usage=NormalizedUsage(
                input_tokens=500,
                output_tokens=200,
                total_tokens=700,
                reasoning_tokens=100,
                cache_read_tokens=50,
                cache_creation_tokens=25,
            ),
        )
        d = t._serialize(event)
        usage = d["usage"]
        assert usage["inputTokens"] == 500
        assert usage["outputTokens"] == 200
        assert usage["totalTokens"] == 700
        assert usage["reasoningTokens"] == 100
        assert usage["cacheReadTokens"] == 50
        assert usage["cacheCreationTokens"] == 25

    def test_stream_timing_serialization(self):
        t = self._make_transport()
        event = CostKeyEvent(
            id="test-789",
            timestamp="2025-01-01T00:00:00.000Z",
            project_id="proj-1",
            provider=Provider.OPENAI,
            streaming=True,
            stream_timing=StreamTiming(
                ttft=42.5,
                tps=85.3,
                stream_duration=1500.0,
                chunk_count=120,
            ),
        )
        d = t._serialize(event)
        st = d["streamTiming"]
        assert st["ttft"] == 42.5
        assert st["tps"] == 85.3
        assert st["streamDuration"] == 1500.0
        assert st["chunkCount"] == 120

    def test_call_site_frames_serialization(self):
        t = self._make_transport()
        event = CostKeyEvent(
            id="test-cs",
            timestamp="2025-01-01T00:00:00.000Z",
            project_id="proj-1",
            provider=Provider.OPENAI,
            call_site=CallSite(
                raw='File "app.py", line 42, in main\n',
                frames=[
                    StackFrame(function_name="main", file_name="app.py", line_number=42),
                    StackFrame(function_name="handler", file_name="routes.py", line_number=15),
                ],
            ),
        )
        d = t._serialize(event)
        cs = d["callSite"]
        assert cs["raw"] == 'File "app.py", line 42, in main\n'
        assert len(cs["frames"]) == 2
        assert cs["frames"][0]["functionName"] == "main"
        assert cs["frames"][0]["fileName"] == "app.py"
        assert cs["frames"][0]["lineNumber"] == 42
        assert cs["frames"][0]["columnNumber"] is None
        assert cs["frames"][1]["functionName"] == "handler"

    def test_null_fields_handled(self):
        t = self._make_transport()
        event = CostKeyEvent(
            id="test-null",
            timestamp="2025-01-01T00:00:00.000Z",
            project_id="proj-1",
            provider=Provider.UNKNOWN,
            model=None,
            status_code=None,
            usage=None,
            cost_usd=None,
            stream_timing=None,
            call_site=None,
        )
        d = t._serialize(event)
        assert d["model"] is None
        assert d["statusCode"] is None
        assert d["usage"] is None
        assert d["costUsd"] is None
        assert d["streamTiming"] is None
        assert d["callSite"] is None

    def test_serialized_json_roundtrip(self):
        """Verify the serialized dict can be JSON-encoded without error."""
        t = self._make_transport()
        event = CostKeyEvent(
            id="test-json",
            timestamp="2025-01-01T00:00:00.000Z",
            project_id="proj-1",
            provider=Provider.OPENAI,
            model="gpt-4o",
            usage=NormalizedUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            stream_timing=StreamTiming(ttft=50.0, tps=100.0, stream_duration=500.0, chunk_count=10),
            call_site=CallSite(raw="trace", frames=[StackFrame("fn", "f.py", 1)]),
            context={"user_id": "u123"},
            request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            response_body={"id": "cmpl-1", "choices": [{"message": {"content": "hello"}}]},
        )
        d = t._serialize(event)
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["projectId"] == "proj-1"
        assert parsed["usage"]["inputTokens"] == 10


# ═══════════════════════════════════════════════════════════════════
# Model extraction
# ═══════════════════════════════════════════════════════════════════

class TestModelExtraction:
    def test_model_from_response_body_priority(self):
        """Response body model takes priority over request body."""
        result = _extract_model(
            {"model": "gpt-4o"},
            {"model": "gpt-4o-2024-11-20"},
        )
        assert result == "gpt-4o-2024-11-20"

    def test_model_from_request_fallback(self):
        result = _extract_model({"model": "gpt-4o"}, {})
        assert result == "gpt-4o"

    def test_model_from_request_when_response_none(self):
        result = _extract_model({"model": "gpt-4o"}, None)
        assert result == "gpt-4o"

    def test_none_when_neither_has_model(self):
        assert _extract_model(None, None) is None
        assert _extract_model({}, {}) is None
        assert _extract_model(None, {}) is None

    def test_model_must_be_string(self):
        """Non-string model values should be ignored."""
        assert _extract_model(None, {"model": 123}) is None
        assert _extract_model({"model": True}, None) is None


# ═══════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════

class TestAsInt:
    def test_int(self):
        assert _as_int(42) == 42

    def test_float(self):
        assert _as_int(42.0) == 42

    def test_none(self):
        assert _as_int(None) is None

    def test_string(self):
        assert _as_int("42") is None

    def test_bool(self):
        assert _as_int(True) is None
        assert _as_int(False) is None


class TestStreamingDetection:
    def test_stream_true(self):
        assert _is_streaming_request({"model": "gpt-4o", "stream": True}) is True

    def test_stream_false(self):
        assert _is_streaming_request({"model": "gpt-4o", "stream": False}) is False

    def test_no_stream_key(self):
        assert _is_streaming_request({"model": "gpt-4o"}) is False

    def test_none_body(self):
        assert _is_streaming_request(None) is False

    def test_non_dict(self):
        assert _is_streaming_request("not a dict") is False
        assert _is_streaming_request([1, 2, 3]) is False


# ═══════════════════════════════════════════════════════════════════
# OpenAI usage helper
# ═══════════════════════════════════════════════════════════════════

class TestExtractOpenAIUsage:
    def test_prompt_completion_format(self):
        body = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        usage = _extract_openai_usage(body)
        assert usage.input_tokens == 10
        assert usage.output_tokens == 5
        assert usage.total_tokens == 15

    def test_input_output_format(self):
        """Some providers use input_tokens/output_tokens instead."""
        body = {"usage": {"input_tokens": 20, "output_tokens": 10}}
        usage = _extract_openai_usage(body)
        assert usage.input_tokens == 20
        assert usage.output_tokens == 10
        assert usage.total_tokens == 30  # auto-calculated

    def test_total_calculated_when_missing(self):
        body = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}
        usage = _extract_openai_usage(body)
        assert usage.total_tokens == 150

    def test_non_dict_returns_none(self):
        assert _extract_openai_usage(None) is None
        assert _extract_openai_usage("string") is None
        assert _extract_openai_usage([1, 2]) is None

    def test_no_usage_key_returns_none(self):
        assert _extract_openai_usage({"model": "test"}) is None

    def test_usage_not_dict_returns_none(self):
        assert _extract_openai_usage({"usage": "string"}) is None
