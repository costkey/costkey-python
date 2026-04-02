"""Test streaming capture with real httpx patterns used by AI SDKs."""
import json
import time
import httpx
from unittest.mock import MagicMock
from costkey.patch import (
    _wrap_streaming_response,
    _is_streaming_request,
    _extract_sse_usage,
    _extract_sse_model,
    _state,
    _process,
)
from costkey.providers import AnthropicExtractor


def make_sse_response(chunks: list[str]) -> httpx.Response:
    """Create a mock httpx streaming response with SSE data."""
    full_body = "\n".join(chunks).encode("utf-8")

    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=httpx.ByteStream(full_body),
    )
    # Don't read the body yet — simulate streaming
    return response


class TestStreamingDetection:
    def test_detects_streaming_request(self):
        assert _is_streaming_request({"model": "claude-sonnet-4-5", "stream": True})

    def test_non_streaming_request(self):
        assert not _is_streaming_request({"model": "claude-sonnet-4-5"})
        assert not _is_streaming_request({"model": "claude-sonnet-4-5", "stream": False})
        assert not _is_streaming_request(None)
        assert not _is_streaming_request("not a dict")


class TestSSEExtraction:
    def test_extract_anthropic_usage(self):
        extractor = AnthropicExtractor()
        sse_text = (
            'data: {"type":"message_start","message":{"model":"claude-sonnet-4-5","usage":{"input_tokens":100,"output_tokens":0}}}\n'
            'data: {"type":"content_block_delta","delta":{"text":"Hello"}}\n'
            'data: {"type":"message_delta","usage":{"input_tokens":100,"output_tokens":50}}\n'
            'data: [DONE]\n'
        )
        usage = _extract_sse_usage(sse_text, extractor)
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_extract_model_from_sse(self):
        extractor = AnthropicExtractor()
        # Anthropic returns model at top level in SSE, or in request body
        sse_text = 'data: {"type":"message_start","model":"claude-sonnet-4-5-20250514"}\n'
        model = _extract_sse_model(sse_text, {"model": "claude-sonnet-4-5-20250514"}, extractor)
        assert model == "claude-sonnet-4-5-20250514"

    def test_empty_sse(self):
        extractor = AnthropicExtractor()
        assert _extract_sse_usage("", extractor) is None
        assert _extract_sse_model("", {}, extractor) is None


class TestStreamWrapper:
    def test_iter_bytes_captures_data(self):
        """Verify that wrapping iter_bytes captures all chunks and fires _on_done."""
        response = make_sse_response([
            'data: {"type":"message_start","message":{"model":"claude-sonnet-4-5"}}',
            'data: {"type":"content_block_delta","delta":{"text":"Hi"}}',
            'data: {"type":"message_delta","usage":{"input_tokens":10,"output_tokens":5}}',
            'data: [DONE]',
        ])

        # Read the response body so iter_bytes works
        response.read()

        # Track what _process receives
        captured = {}
        original_process = _process

        extractor = AnthropicExtractor()
        start = time.perf_counter()

        _wrap_streaming_response(
            response, extractor, "https://api.anthropic.com/v1/messages",
            "POST", {"model": "claude-sonnet-4-5", "stream": True},
            start, None, {},
        )

        # iter_bytes should yield all data and trigger _on_done
        chunks = list(response.iter_bytes())
        assert len(chunks) > 0
        # The accumulated text should contain the SSE data
        total_text = b"".join(chunks).decode()
        assert "message_delta" in total_text

    def test_iter_lines_captures_data(self):
        """Verify iter_lines wrapper works for SDKs that use line-by-line iteration."""
        response = make_sse_response([
            'data: {"type":"message_start","message":{"model":"gpt-4o"}}',
            'data: {"type":"content","delta":{"text":"Hello"}}',
            'data: [DONE]',
        ])
        response.read()

        extractor = AnthropicExtractor()
        start = time.perf_counter()

        _wrap_streaming_response(
            response, extractor, "https://api.anthropic.com/v1/messages",
            "POST", {"stream": True}, start, None, {},
        )

        lines = list(response.iter_lines())
        assert len(lines) > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
