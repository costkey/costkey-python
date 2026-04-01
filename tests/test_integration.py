"""Integration tests — verify real-world behavior."""
from costkey.providers import find_extractor
from costkey.pricing import compute_cost
from costkey.stack import capture_call_site


def test_anthropic_response():
    """Test with real Anthropic API response format."""
    ext = find_extractor("https://api.anthropic.com/v1/messages")
    assert ext is not None
    assert ext.provider.value == "anthropic"

    response = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-0-20250514",
        "content": [{"type": "text", "text": "Hello!"}],
        "usage": {"input_tokens": 3018, "output_tokens": 123},
    }

    usage = ext.extract_usage(response)
    assert usage is not None
    assert usage.input_tokens == 3018
    assert usage.output_tokens == 123

    model = ext.extract_model(None, response)
    assert model == "claude-sonnet-4-0-20250514"

    cost = compute_cost(model, usage)
    assert cost is not None
    assert cost > 0, f"Cost must be > 0, got {cost}"
    print(f"  Anthropic cost: ${cost}")


def test_openai_response():
    """Test with real OpenAI API response format."""
    ext = find_extractor("https://api.openai.com/v1/chat/completions")
    assert ext is not None

    response = {
        "id": "chatcmpl-abc",
        "model": "gpt-4o-2024-11-20",
        "usage": {"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700},
    }

    usage = ext.extract_usage(response)
    cost = compute_cost(ext.extract_model(None, response), usage)
    assert cost is not None and cost > 0
    print(f"  OpenAI cost: ${cost}")


def test_groq_response():
    """Test Groq response format (uses input_tokens/output_tokens)."""
    ext = find_extractor("https://api.groq.com/openai/v1/chat/completions")
    assert ext is not None

    response = {
        "model": "llama-3.3-70b-versatile",
        "usage": {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
    }

    usage = ext.extract_usage(response)
    cost = compute_cost(ext.extract_model(None, response), usage)
    assert cost is not None and cost > 0
    print(f"  Groq cost: ${cost}")


def test_deepseek_response():
    """Test DeepSeek with cache tokens."""
    ext = find_extractor("https://api.deepseek.com/v1/chat/completions")
    assert ext is not None

    response = {
        "model": "deepseek-chat",
        "usage": {
            "prompt_tokens": 2000,
            "completion_tokens": 500,
            "total_tokens": 2500,
            "prompt_cache_hit_tokens": 1500,
        },
    }

    usage = ext.extract_usage(response)
    assert usage.cache_read_tokens == 1500
    cost = compute_cost("deepseek-chat", usage)
    assert cost is not None and cost > 0
    print(f"  DeepSeek cost: ${cost}")


def test_stack_trace_filters_libraries():
    """Stack trace should only show user code, not library internals."""
    def my_handler():
        def my_inner():
            return capture_call_site()
        return my_inner()

    site = my_handler()
    assert site is not None, "Stack trace should not be None"
    assert len(site.frames) > 0, "Should have at least one frame"

    for frame in site.frames:
        fname = frame.file_name or ""
        func = frame.function_name or ""
        assert "site-packages" not in fname, f"Library frame leaked: {fname}"
        assert "threading" not in fname, f"Threading frame leaked: {fname}"
        assert "_bootstrap" not in func, f"Bootstrap func leaked: {func}"
        assert "/lib/python" not in fname or "site-packages" in fname, f"Stdlib frame leaked: {fname}"

    print(f"  Stack: {len(site.frames)} frames, top: {site.frames[0].function_name}()")


def test_all_providers_detect():
    """All 14 providers should be detected."""
    urls = [
        ("OpenAI", "https://api.openai.com/v1/chat"),
        ("Anthropic", "https://api.anthropic.com/v1/messages"),
        ("Google", "https://generativelanguage.googleapis.com/v1/models"),
        ("OpenRouter", "https://openrouter.ai/api/v1/chat"),
        ("xAI", "https://api.x.ai/v1/chat"),
        ("Groq", "https://api.groq.com/openai/v1/chat"),
        ("Mistral", "https://api.mistral.ai/v1/chat"),
        ("DeepSeek", "https://api.deepseek.com/v1/chat"),
        ("Together", "https://api.together.xyz/v1/chat"),
        ("Fireworks", "https://api.fireworks.ai/inference/v1/chat"),
        ("Perplexity", "https://api.perplexity.ai/chat"),
        ("Cerebras", "https://api.cerebras.ai/v1/chat"),
        ("Bedrock", "https://bedrock-runtime.us-east-1.amazonaws.com/model/test"),
        ("Cohere", "https://api.cohere.com/v2/chat"),
    ]

    for name, url in urls:
        ext = find_extractor(url)
        assert ext is not None, f"{name} not detected for {url}"

    # Non-AI should return None
    assert find_extractor("https://api.example.com/data") is None
    print(f"  All 14 providers detected, non-AI correctly ignored")


if __name__ == "__main__":
    tests = [
        test_anthropic_response,
        test_openai_response,
        test_groq_response,
        test_deepseek_response,
        test_stack_trace_filters_libraries,
        test_all_providers_detect,
    ]

    for test in tests:
        print(f"Running {test.__name__}...")
        test()

    print(f"\n✅ ALL {len(tests)} TESTS PASSED")
