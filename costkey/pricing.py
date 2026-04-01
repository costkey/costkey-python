"""Model pricing — cost per 1M tokens in USD."""
from __future__ import annotations
from costkey.types import NormalizedUsage

# (input_per_1M, output_per_1M, cache_read_per_1M, cache_write_per_1M)
_PRICING: dict[str, tuple[float, float, float | None, float | None]] = {
    # OpenAI
    "gpt-4o": (2.5, 10, None, None),
    "gpt-4o-mini": (0.15, 0.6, None, None),
    "gpt-4-turbo": (10, 30, None, None),
    "gpt-4": (30, 60, None, None),
    "gpt-3.5-turbo": (0.5, 1.5, None, None),
    "o1": (15, 60, None, None),
    "o1-mini": (3, 12, None, None),
    "o1-pro": (150, 600, None, None),
    "o3": (10, 40, None, None),
    "o3-mini": (1.1, 4.4, None, None),
    "o4-mini": (1.1, 4.4, None, None),

    # Anthropic
    "claude-opus-4-0-20250514": (15, 75, 1.5, 18.75),
    "claude-sonnet-4-0-20250514": (3, 15, 0.3, 3.75),
    "claude-sonnet-4-5-20250514": (3, 15, 0.3, 3.75),
    "claude-haiku-3-5-20241022": (0.8, 4, 0.08, 1),
    "claude-3-5-sonnet-20241022": (3, 15, 0.3, 3.75),
    "claude-3-5-haiku-20241022": (0.8, 4, 0.08, 1),
    "claude-3-opus-20240229": (15, 75, 1.5, 18.75),
    # Short names (some APIs return these)
    "claude-opus-4": (15, 75, 1.5, 18.75),
    "claude-sonnet-4": (3, 15, 0.3, 3.75),

    # Google Gemini
    "gemini-2.0-flash": (0.1, 0.4, None, None),
    "gemini-2.0-flash-lite": (0.02, 0.1, None, None),
    "gemini-1.5-pro": (1.25, 5, None, None),
    "gemini-1.5-flash": (0.075, 0.3, None, None),
    "gemini-2.5-pro": (1.25, 10, None, None),
    "gemini-2.5-flash": (0.15, 0.6, None, None),

    # xAI (Grok)
    "grok-3": (3, 15, None, None),
    "grok-3-mini": (0.3, 0.5, None, None),
    "grok-4": (3, 15, None, None),

    # Groq
    "llama-3.3-70b-versatile": (0.59, 0.79, None, None),
    "llama-3.1-8b-instant": (0.05, 0.08, None, None),
    "mixtral-8x7b-32768": (0.24, 0.24, None, None),

    # Mistral
    "mistral-large-latest": (2, 6, None, None),
    "mistral-small-latest": (0.2, 0.6, None, None),
    "codestral-latest": (0.3, 0.9, None, None),

    # DeepSeek
    "deepseek-chat": (0.14, 0.28, 0.014, None),
    "deepseek-reasoner": (0.55, 2.19, 0.055, None),

    # Cohere
    "command-r-plus": (2.5, 10, None, None),
    "command-r": (0.15, 0.6, None, None),

    # Perplexity
    "sonar-pro": (3, 15, None, None),
    "sonar": (1, 1, None, None),
}


def _find_pricing(model: str) -> tuple[float, float, float | None, float | None] | None:
    if model in _PRICING:
        return _PRICING[model]
    # Try progressively shorter prefixes
    parts = model.split("-")
    for i in range(len(parts) - 1, 0, -1):
        prefix = "-".join(parts[:i])
        if prefix in _PRICING:
            return _PRICING[prefix]
    return None


def compute_cost(model: str, usage: NormalizedUsage) -> float | None:
    pricing = _find_pricing(model)
    if pricing is None:
        return None

    inp, out, cache_r, cache_w = pricing
    cost = 0.0
    if usage.input_tokens is not None:
        cost += (usage.input_tokens / 1_000_000) * inp
    if usage.output_tokens is not None:
        cost += (usage.output_tokens / 1_000_000) * out
    if usage.cache_read_tokens is not None and cache_r is not None:
        cost += (usage.cache_read_tokens / 1_000_000) * cache_r
    if usage.cache_creation_tokens is not None and cache_w is not None:
        cost += (usage.cache_creation_tokens / 1_000_000) * cache_w

    return round(cost, 6)


def register_pricing(model: str, input_per_1m: float, output_per_1m: float,
                      cache_read_per_1m: float | None = None,
                      cache_write_per_1m: float | None = None) -> None:
    _PRICING[model] = (input_per_1m, output_per_1m, cache_read_per_1m, cache_write_per_1m)
