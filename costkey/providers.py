"""Provider extractors — detect AI providers by URL and extract usage from responses."""
from __future__ import annotations
from urllib.parse import urlparse
from typing import Any, Protocol
from costkey.types import Provider, NormalizedUsage


class ProviderExtractor(Protocol):
    provider: Provider
    def match(self, url: str) -> bool: ...
    def extract_usage(self, body: Any) -> NormalizedUsage | None: ...
    def extract_model(self, request_body: Any, response_body: Any) -> str | None: ...


def _as_int(val: Any) -> int | None:
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return int(val)
    return None


def _extract_openai_usage(body: Any) -> NormalizedUsage | None:
    """Standard OpenAI-compatible usage extraction — works for many providers."""
    if not isinstance(body, dict):
        return None
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    input_t = _as_int(usage.get("prompt_tokens")) or _as_int(usage.get("input_tokens"))
    output_t = _as_int(usage.get("completion_tokens")) or _as_int(usage.get("output_tokens"))
    total_t = _as_int(usage.get("total_tokens"))
    if total_t is None and input_t is not None and output_t is not None:
        total_t = input_t + output_t
    return NormalizedUsage(input_tokens=input_t, output_tokens=output_t, total_tokens=total_t)


def _extract_model(request_body: Any, response_body: Any) -> str | None:
    if isinstance(response_body, dict) and isinstance(response_body.get("model"), str):
        return response_body["model"]
    if isinstance(request_body, dict) and isinstance(request_body.get("model"), str):
        return request_body["model"]
    return None


class OpenAIExtractor:
    provider = Provider.OPENAI
    def match(self, url: str) -> bool:
        host = urlparse(url).hostname or ""
        return host == "api.openai.com" or host.endswith(".openai.azure.com")
    def extract_usage(self, body: Any) -> NormalizedUsage | None:
        usage = _extract_openai_usage(body)
        if usage and isinstance(body, dict):
            u = body.get("usage", {})
            details = u.get("completion_tokens_details") or u.get("output_tokens_details") or {}
            if isinstance(details, dict):
                usage.reasoning_tokens = _as_int(details.get("reasoning_tokens"))
        return usage
    def extract_model(self, req: Any, resp: Any) -> str | None:
        return _extract_model(req, resp)


class AnthropicExtractor:
    provider = Provider.ANTHROPIC
    def match(self, url: str) -> bool:
        return (urlparse(url).hostname or "") == "api.anthropic.com"
    def extract_usage(self, body: Any) -> NormalizedUsage | None:
        if not isinstance(body, dict): return None
        usage = body.get("usage")
        if not isinstance(usage, dict): return None
        input_t = _as_int(usage.get("input_tokens"))
        output_t = _as_int(usage.get("output_tokens"))
        total_t = (input_t or 0) + (output_t or 0) if input_t is not None or output_t is not None else None
        return NormalizedUsage(input_tokens=input_t, output_tokens=output_t, total_tokens=total_t,
            cache_read_tokens=_as_int(usage.get("cache_read_input_tokens")),
            cache_creation_tokens=_as_int(usage.get("cache_creation_input_tokens")))
    def extract_model(self, req: Any, resp: Any) -> str | None:
        return _extract_model(req, resp)


class GoogleExtractor:
    provider = Provider.GOOGLE
    def match(self, url: str) -> bool:
        host = urlparse(url).hostname or ""
        return host == "generativelanguage.googleapis.com" or host.endswith("-aiplatform.googleapis.com")
    def extract_usage(self, body: Any) -> NormalizedUsage | None:
        if not isinstance(body, dict): return None
        meta = body.get("usageMetadata")
        if not isinstance(meta, dict): return None
        input_t = _as_int(meta.get("promptTokenCount"))
        output_t = _as_int(meta.get("candidatesTokenCount"))
        total_t = _as_int(meta.get("totalTokenCount"))
        return NormalizedUsage(input_tokens=input_t, output_tokens=output_t, total_tokens=total_t,
            reasoning_tokens=_as_int(meta.get("thoughtsTokenCount")),
            cache_read_tokens=_as_int(meta.get("cachedContentTokenCount")))
    def extract_model(self, req: Any, resp: Any) -> str | None:
        if isinstance(resp, dict) and isinstance(resp.get("modelVersion"), str):
            return resp["modelVersion"]
        return None


# ── Additional providers (OpenAI-compatible) ──

def _make_simple_extractor(hostname: str, prov: Provider = Provider.UNKNOWN):
    class SimpleExtractor:
        provider = prov
        def match(self, url: str) -> bool:
            return (urlparse(url).hostname or "") == hostname
        def extract_usage(self, body: Any) -> NormalizedUsage | None:
            return _extract_openai_usage(body)
        def extract_model(self, req: Any, resp: Any) -> str | None:
            return _extract_model(req, resp)
    return SimpleExtractor()


class DeepSeekExtractor:
    provider = Provider.DEEPSEEK
    def match(self, url: str) -> bool:
        return (urlparse(url).hostname or "") == "api.deepseek.com"
    def extract_usage(self, body: Any) -> NormalizedUsage | None:
        usage = _extract_openai_usage(body)
        if usage and isinstance(body, dict):
            u = body.get("usage", {})
            if isinstance(u, dict):
                usage.cache_read_tokens = _as_int(u.get("prompt_cache_hit_tokens"))
                details = u.get("completion_tokens_details") or {}
                if isinstance(details, dict):
                    usage.reasoning_tokens = _as_int(details.get("reasoning_tokens"))
        return usage
    def extract_model(self, req: Any, resp: Any) -> str | None:
        return _extract_model(req, resp)


class BedrockExtractor:
    provider = Provider.BEDROCK
    def match(self, url: str) -> bool:
        host = urlparse(url).hostname or ""
        return "bedrock-runtime" in host and host.endswith(".amazonaws.com")
    def extract_usage(self, body: Any) -> NormalizedUsage | None:
        if not isinstance(body, dict): return None
        usage = body.get("usage")
        if isinstance(usage, dict):
            return NormalizedUsage(
                input_tokens=_as_int(usage.get("inputTokens")),
                output_tokens=_as_int(usage.get("outputTokens")),
                total_tokens=_as_int(usage.get("totalTokens")))
        metrics = body.get("amazon-bedrock-invocationMetrics")
        if isinstance(metrics, dict):
            return NormalizedUsage(
                input_tokens=_as_int(metrics.get("inputTokenCount")),
                output_tokens=_as_int(metrics.get("outputTokenCount")))
        return None
    def extract_model(self, req: Any, resp: Any) -> str | None:
        if isinstance(req, dict) and isinstance(req.get("modelId"), str):
            return req["modelId"]
        return None


class CohereExtractor:
    provider = Provider.COHERE
    def match(self, url: str) -> bool:
        return (urlparse(url).hostname or "") == "api.cohere.com"
    def extract_usage(self, body: Any) -> NormalizedUsage | None:
        if isinstance(body, dict):
            meta = body.get("meta")
            if isinstance(meta, dict):
                tokens = meta.get("tokens")
                if isinstance(tokens, dict):
                    return NormalizedUsage(
                        input_tokens=_as_int(tokens.get("input_tokens")),
                        output_tokens=_as_int(tokens.get("output_tokens")))
        return _extract_openai_usage(body)
    def extract_model(self, req: Any, resp: Any) -> str | None:
        return _extract_model(req, resp)


# Registry
_extractors: list[ProviderExtractor] = [
    OpenAIExtractor(),
    AnthropicExtractor(),
    GoogleExtractor(),
    DeepSeekExtractor(),
    BedrockExtractor(),
    CohereExtractor(),
    _make_simple_extractor("openrouter.ai", Provider.OPENROUTER),
    _make_simple_extractor("api.x.ai", Provider.XAI),
    _make_simple_extractor("api.grok.xai.com", Provider.XAI),
    _make_simple_extractor("api.groq.com", Provider.GROQ),
    _make_simple_extractor("api.mistral.ai", Provider.MISTRAL),
    _make_simple_extractor("api.together.xyz", Provider.TOGETHER),
    _make_simple_extractor("api.fireworks.ai", Provider.FIREWORKS),
    _make_simple_extractor("api.perplexity.ai", Provider.PERPLEXITY),
    _make_simple_extractor("api.cerebras.ai", Provider.CEREBRAS)
]


def find_extractor(url: str) -> ProviderExtractor | None:
    for ext in _extractors:
        if ext.match(url):
            return ext
    return None


def register_extractor(extractor: ProviderExtractor) -> None:
    _extractors.append(extractor)
