"""Provider extractors — detect AI providers by URL and extract usage from responses."""
from __future__ import annotations
import math
from urllib.parse import urlparse
from typing import Any, Protocol
from costkey.types import Provider, NormalizedUsage


class ProviderExtractor(Protocol):
    provider: Provider | str
    def match(self, url: str) -> bool: ...
    def extract_usage(self, body: Any) -> NormalizedUsage | None: ...
    def extract_model(self, request_body: Any, response_body: Any) -> str | None: ...


def _as_int(val: Any) -> int | None:
    if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
        return int(val)
    return None


def _as_number(val: Any) -> int | float | None:
    """Return finite JSON numbers while rejecting booleans and strings."""
    if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
        return val
    return None


def _native_units(usage: dict[str, Any], *, exclude: set[str] | None = None) -> dict[str, int | float | None]:
    """Keep numeric provider units that are not part of the token fields.

    Native units are intentionally opaque and remain keyed by the provider's
    vocabulary.  The accounting layer can display seconds/characters/images
    without pretending they are interchangeable with tokens.
    """
    excluded = exclude or set()
    result: dict[str, int | float | None] = {}
    for key, value in usage.items():
        if key in excluded or _as_number(value) is None:
            continue
        lowered = key.lower()
        if "token" in lowered and not any(term in lowered for term in ("audio", "image", "text")):
            continue
        if any(term in lowered for term in ("cost", "price", "reasoning")):
            continue
        if any(term in lowered for term in ("latency",)):
            continue
        # Preserve native audio/session durations and other modality units;
        # only generic transport timing belongs outside usage.
        if any(term in lowered for term in ("duration", "time")) and not any(
            term in lowered for term in ("audio", "character", "image", "request", "second", "minute", "byte")
        ):
            continue
        result[key] = value
    return result


def _extract_openai_usage(body: Any) -> NormalizedUsage | None:
    """Standard OpenAI-compatible usage extraction — works for many providers."""
    if not isinstance(body, dict):
        return None
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    input_t = _as_int(usage.get("prompt_tokens"))
    if input_t is None:
        input_t = _as_int(usage.get("input_tokens"))
    output_t = _as_int(usage.get("completion_tokens"))
    if output_t is None:
        output_t = _as_int(usage.get("output_tokens"))
    total_t = _as_int(usage.get("total_tokens"))
    if total_t is None and input_t is not None and output_t is not None:
        total_t = input_t + output_t
    input_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
    input_details = input_details if isinstance(input_details, dict) else {}
    details = usage.get("completion_tokens_details") or usage.get("output_tokens_details")
    details = details if isinstance(details, dict) else {}
    reasoning_t = _as_int(usage.get("reasoning_tokens"))
    if reasoning_t is None:
        reasoning_t = _as_int(details.get("reasoning_tokens"))
    image_t = _as_int(usage.get("image_output_tokens"))
    if image_t is None:
        image_t = _as_int(usage.get("image_tokens"))
    if image_t is None:
        image_t = _as_int(usage.get("native_tokens_completion_images"))
    if image_t is None:
        image_t = _as_int(details.get("image_tokens"))
    cost_usd = _as_number(usage.get("cost"))
    if cost_usd is None:
        cost_usd = _as_number(usage.get("cost_usd"))
    return NormalizedUsage(
        input_tokens=input_t,
        output_tokens=output_t,
        total_tokens=total_t,
        reasoning_tokens=reasoning_t,
        cache_read_tokens=_as_int(usage.get("cached_tokens"))
        if _as_int(usage.get("cached_tokens")) is not None
        else _as_int(input_details.get("cached_tokens")),
        cache_creation_tokens=_as_int(usage.get("cache_write_tokens"))
        if _as_int(usage.get("cache_write_tokens")) is not None
        else _as_int(input_details.get("cache_creation_tokens")),
        native_units=_native_units(usage, exclude={
            "prompt_tokens", "input_tokens", "completion_tokens", "output_tokens",
            "total_tokens", "reasoning_tokens", "cached_tokens", "cache_write_tokens",
            "image_output_tokens", "image_tokens", "native_tokens_completion_images",
        }),
        cost_usd=float(cost_usd) if cost_usd is not None else None,
        image_output_tokens=image_t,
    )


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
                detail_reasoning = _as_int(details.get("reasoning_tokens"))
                if detail_reasoning is not None:
                    usage.reasoning_tokens = detail_reasoning
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
            cache_creation_tokens=_as_int(usage.get("cache_creation_input_tokens")),
            native_units=_native_units(usage, exclude={
                "input_tokens", "output_tokens", "cache_read_input_tokens",
                "cache_creation_input_tokens",
            }))
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
        if total_t is None and input_t is not None and output_t is not None:
            total_t = input_t + output_t
        return NormalizedUsage(input_tokens=input_t, output_tokens=output_t, total_tokens=total_t,
            reasoning_tokens=_as_int(meta.get("thoughtsTokenCount")),
            cache_read_tokens=_as_int(meta.get("cachedContentTokenCount")),
            native_units=_native_units(meta, exclude={
                "promptTokenCount", "candidatesTokenCount", "totalTokenCount",
                "thoughtsTokenCount", "cachedContentTokenCount",
            }))
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
                    detail_reasoning = _as_int(details.get("reasoning_tokens"))
                    if detail_reasoning is not None:
                        usage.reasoning_tokens = detail_reasoning
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
            input_t = _as_int(usage.get("inputTokens"))
            output_t = _as_int(usage.get("outputTokens"))
            return NormalizedUsage(
                input_tokens=input_t,
                output_tokens=output_t,
                total_tokens=_as_int(usage.get("totalTokens"))
                if _as_int(usage.get("totalTokens")) is not None
                else (input_t + output_t if input_t is not None and output_t is not None else None),
                native_units=_native_units(usage, exclude={"inputTokens", "outputTokens", "totalTokens"}))
        metrics = body.get("amazon-bedrock-invocationMetrics")
        if isinstance(metrics, dict):
            return NormalizedUsage(
                input_tokens=_as_int(metrics.get("inputTokenCount")),
                output_tokens=_as_int(metrics.get("outputTokenCount")),
                total_tokens=(
                    _as_int(metrics.get("inputTokenCount")) + _as_int(metrics.get("outputTokenCount"))
                    if _as_int(metrics.get("inputTokenCount")) is not None
                    and _as_int(metrics.get("outputTokenCount")) is not None
                    else None
                ),
                native_units=_native_units(metrics, exclude={"inputTokenCount", "outputTokenCount"}))
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
                        output_tokens=_as_int(tokens.get("output_tokens")),
                        total_tokens=(
                            _as_int(tokens.get("input_tokens")) + _as_int(tokens.get("output_tokens"))
                            if _as_int(tokens.get("input_tokens")) is not None
                            and _as_int(tokens.get("output_tokens")) is not None
                            else None
                        ),
                        native_units=_native_units(tokens, exclude={"input_tokens", "output_tokens"}))
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
    _make_simple_extractor("api.cerebras.ai", Provider.CEREBRAS),
    # AI proxies/gateways
    _make_simple_extractor("api.portkey.ai", Provider.OPENAI),
    _make_simple_extractor("oai.helicone.ai", Provider.OPENAI),
    _make_simple_extractor("anthropic.helicone.ai", Provider.ANTHROPIC),
    _make_simple_extractor("gateway.helicone.ai", Provider.OPENAI),
]
_runtime_extractors: list[ProviderExtractor] = []


def find_extractor(url: str) -> ProviderExtractor | None:
    for ext in _extractors:
        if ext.match(url):
            return ext
    return None


def register_extractor(extractor: ProviderExtractor) -> None:
    _extractors.append(extractor)


def _lookup_path(value: Any, path: str | None) -> Any:
    if not path:
        return None
    current = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


class _RuntimeProviderExtractor:
    def __init__(self, config: dict[str, Any]) -> None:
        self._hostname = str(config.get("hostname") or "").lower()
        self._hostname = self._hostname.removeprefix("https://").removeprefix("http://").split("/", 1)[0]
        self._input_path = config.get("usage_input_path", config.get("usageInputPath"))
        self._output_path = config.get("usage_output_path", config.get("usageOutputPath"))
        self._model_path = config.get("model_path", config.get("modelPath"))
        self.provider = str(config.get("id") or config.get("name") or self._hostname or "custom")

    def match(self, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        if self._hostname.startswith("*."):
            return host.endswith(self._hostname[1:])
        return bool(self._hostname) and host == self._hostname

    def extract_usage(self, body: Any) -> NormalizedUsage | None:
        input_t = _as_int(_lookup_path(body, self._input_path))
        output_t = _as_int(_lookup_path(body, self._output_path))
        if input_t is None and output_t is None:
            return None
        return NormalizedUsage(
            input_tokens=input_t,
            output_tokens=output_t,
            total_tokens=input_t + output_t if input_t is not None and output_t is not None else None,
        )

    def extract_model(self, request_body: Any, response_body: Any) -> str | None:
        return (
            _lookup_path(response_body, self._model_path)
            or _lookup_path(request_body, self._model_path)
            or _extract_model(request_body, response_body)
        )


def apply_runtime_providers(configs: Any) -> None:
    """Replace DSN-configured custom providers without touching built-ins."""
    global _runtime_extractors
    for extractor in _runtime_extractors:
        try:
            _extractors.remove(extractor)
        except ValueError:
            pass
    _runtime_extractors = []
    if not isinstance(configs, list):
        return
    for config in configs:
        if not isinstance(config, dict) or not config.get("hostname"):
            continue
        extractor = _RuntimeProviderExtractor(config)
        _runtime_extractors.append(extractor)
        # Runtime entries follow built-ins so a configured host cannot replace
        # a known provider; their exact host match still beats no built-in.
        _extractors.append(extractor)


def clear_runtime_providers() -> None:
    """Drop DSN-scoped provider entries when a client shuts down."""
    apply_runtime_providers([])
