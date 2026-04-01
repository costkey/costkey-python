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


class OpenAIExtractor:
    provider = Provider.OPENAI

    def match(self, url: str) -> bool:
        host = urlparse(url).hostname or ""
        return host == "api.openai.com" or host.endswith(".openai.azure.com")

    def extract_usage(self, body: Any) -> NormalizedUsage | None:
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

        details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
        reasoning = _as_int(details.get("reasoning_tokens")) if isinstance(details, dict) else None

        return NormalizedUsage(
            input_tokens=input_t, output_tokens=output_t, total_tokens=total_t,
            reasoning_tokens=reasoning,
        )

    def extract_model(self, request_body: Any, response_body: Any) -> str | None:
        if isinstance(response_body, dict) and isinstance(response_body.get("model"), str):
            return response_body["model"]
        if isinstance(request_body, dict) and isinstance(request_body.get("model"), str):
            return request_body["model"]
        return None


class AnthropicExtractor:
    provider = Provider.ANTHROPIC

    def match(self, url: str) -> bool:
        host = urlparse(url).hostname or ""
        return host == "api.anthropic.com"

    def extract_usage(self, body: Any) -> NormalizedUsage | None:
        if not isinstance(body, dict):
            return None
        usage = body.get("usage")
        if not isinstance(usage, dict):
            return None

        input_t = _as_int(usage.get("input_tokens"))
        output_t = _as_int(usage.get("output_tokens"))
        total_t = (input_t or 0) + (output_t or 0) if input_t is not None or output_t is not None else None

        return NormalizedUsage(
            input_tokens=input_t, output_tokens=output_t, total_tokens=total_t,
            cache_read_tokens=_as_int(usage.get("cache_read_input_tokens")),
            cache_creation_tokens=_as_int(usage.get("cache_creation_input_tokens")),
        )

    def extract_model(self, request_body: Any, response_body: Any) -> str | None:
        if isinstance(response_body, dict) and isinstance(response_body.get("model"), str):
            return response_body["model"]
        if isinstance(request_body, dict) and isinstance(request_body.get("model"), str):
            return request_body["model"]
        return None


class GoogleExtractor:
    provider = Provider.GOOGLE

    def match(self, url: str) -> bool:
        host = urlparse(url).hostname or ""
        return host == "generativelanguage.googleapis.com" or host.endswith("-aiplatform.googleapis.com")

    def extract_usage(self, body: Any) -> NormalizedUsage | None:
        if not isinstance(body, dict):
            return None
        meta = body.get("usageMetadata")
        if not isinstance(meta, dict):
            return None

        input_t = _as_int(meta.get("promptTokenCount"))
        output_t = _as_int(meta.get("candidatesTokenCount"))
        total_t = _as_int(meta.get("totalTokenCount"))
        if total_t is None and input_t is not None and output_t is not None:
            total_t = input_t + output_t

        return NormalizedUsage(
            input_tokens=input_t, output_tokens=output_t, total_tokens=total_t,
            reasoning_tokens=_as_int(meta.get("thoughtsTokenCount")),
            cache_read_tokens=_as_int(meta.get("cachedContentTokenCount")),
        )

    def extract_model(self, request_body: Any, response_body: Any) -> str | None:
        if isinstance(response_body, dict) and isinstance(response_body.get("modelVersion"), str):
            return response_body["modelVersion"]
        return None


# Registry
_extractors: list[ProviderExtractor] = [OpenAIExtractor(), AnthropicExtractor(), GoogleExtractor()]


def find_extractor(url: str) -> ProviderExtractor | None:
    for ext in _extractors:
        if ext.match(url):
            return ext
    return None


def register_extractor(extractor: ProviderExtractor) -> None:
    _extractors.append(extractor)
