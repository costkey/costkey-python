"""Provider extractors — detect AI providers by URL and extract usage from responses."""
from __future__ import annotations
import dataclasses
import logging
import math
import re
from urllib.parse import urlparse, unquote
from typing import Any, Protocol
from costkey.types import Provider, NormalizedUsage

logger = logging.getLogger("costkey")


class ProviderExtractor(Protocol):
    provider: Provider | str
    def match(self, url: str) -> bool: ...
    def extract_usage(self, body: Any) -> NormalizedUsage | None: ...
    def extract_model(self, request_body: Any, response_body: Any) -> str | None: ...
    # Optional: extractors whose usage lives in the URL, request or headers
    # (per-character TTS, per-query search) also define
    #   extract_call_usage(url, request_body, response_body, headers)
    #   extract_call_model(url, request_body, response_body)


def _as_int(val: Any) -> int | None:
    if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
        return int(val)
    return None


def _as_number(val: Any) -> int | float | None:
    """Return finite JSON numbers while rejecting booleans and strings."""
    if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
        return val
    return None


def _first_int(source: Any, *keys: str) -> int | None:
    if not isinstance(source, dict):
        return None
    for key in keys:
        value = _as_int(source.get(key))
        if value is not None:
            return value
    return None


def _header(headers: Any, name: str) -> Any:
    if headers is None:
        return None
    try:
        value = headers.get(name)
        # botocore hands back a plain dict with lower-cased names.
        return value if value is not None else headers.get(name.lower())
    except Exception:
        return None


def _header_number(headers: Any, *names: str) -> int | float | None:
    for name in names:
        value = _header(headers, name)
        if isinstance(value, bytes):
            value = value.decode("latin-1")
        if isinstance(value, str):
            try:
                value = float(value.strip())
            except ValueError:
                continue
            if math.isfinite(value) and value >= 0:
                return int(value) if value.is_integer() else value
            continue
        number = _as_number(value)
        if number is not None and number >= 0:
            return number
    return None


def _path_segments(url: str) -> list[str]:
    return [segment for segment in (urlparse(url).path or "").split("/") if segment]


def merge_usage(current: NormalizedUsage | None, nxt: NormalizedUsage) -> NormalizedUsage:
    """Combine two reports of one call; every field keeps its largest value."""
    if current is None:
        return dataclasses.replace(nxt, native_units=dict(nxt.native_units or {}))

    def pick(a: Any, b: Any) -> Any:
        if a is None:
            return b
        if b is None:
            return a
        return max(a, b)

    merged = dataclasses.replace(current)
    for name in ("input_tokens", "output_tokens", "reasoning_tokens", "cache_read_tokens",
                 "cache_creation_tokens", "cost_usd", "image_output_tokens"):
        setattr(merged, name, pick(getattr(current, name, None), getattr(nxt, name, None)))
    total = pick(current.total_tokens, nxt.total_tokens)
    if merged.input_tokens is not None and merged.output_tokens is not None:
        total = max(total or 0, merged.input_tokens + merged.output_tokens)
    merged.total_tokens = total
    units = dict(current.native_units or {})
    for key, value in (nxt.native_units or {}).items():
        prior = units.get(key)
        units[key] = value if _as_number(prior) is None or _as_number(value) is None else max(prior, value)
    merged.native_units = units
    return merged


def call_usage(extractor: Any, url: str, request_body: Any, response_body: Any,
               headers: Any = None, status_code: int | None = None) -> NormalizedUsage | None:
    """Usage for one call, from the response or, where the provider bills on
    the request (characters, queries), from the URL, request and headers."""
    try:
        hook = getattr(extractor, "extract_call_usage", None)
        if callable(hook):
            # Request-derived units describe work the provider did not do on
            # a failed call.
            if status_code is not None and status_code >= 400:
                return None
            return hook(url, request_body, response_body, headers)
        return extractor.extract_usage(response_body) if response_body else None
    except Exception:
        logger.debug("[costkey] usage extraction failed", exc_info=True)
        return None


def call_model(extractor: Any, url: str, request_body: Any, response_body: Any) -> str | None:
    try:
        hook = getattr(extractor, "extract_call_model", None)
        if callable(hook):
            return hook(url, request_body, response_body)
        return extractor.extract_model(request_body, response_body)
    except Exception:
        logger.debug("[costkey] model extraction failed", exc_info=True)
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
        return (
            host == "api.openai.com"
            or host.endswith(".openai.azure.com")
            or host.endswith(".services.ai.azure.com")
            or host.endswith(".cognitiveservices.azure.com")
        )
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


def _grounding_units(bodies: Any) -> dict[str, int]:
    """Search and Maps grounding are billed per query/request, not per token.

    Gemini 3 bills each executed search query; earlier models bill each
    web-grounded prompt. Retrieval (Vertex AI Search) grounding is neither.
    """
    queries: set[str] = set()
    web = maps = False
    for body in bodies:
        candidates = body.get("candidates") if isinstance(body, dict) else None
        for candidate in candidates if isinstance(candidates, list) else ():
            metadata = candidate.get("groundingMetadata") if isinstance(candidate, dict) else None
            if not isinstance(metadata, dict):
                continue
            searched = metadata.get("webSearchQueries")
            if isinstance(searched, list) and searched:
                web = True
                queries.update(query for query in searched if isinstance(query, str) and query)
            chunks = metadata.get("groundingChunks")
            for chunk in chunks if isinstance(chunks, list) else ():
                if isinstance(chunk, dict):
                    web = web or "web" in chunk
                    maps = maps or "maps" in chunk
            maps = maps or bool(metadata.get("googleMapsWidgetContextToken"))
    units: dict[str, int] = {}
    if queries:
        units["searchQueries"] = len(queries)
    if web:
        units["groundingRequests"] = 1
    if maps:
        units["mapsGroundingRequests"] = 1
    return units


_GOOGLE_MODEL_PATH = re.compile(r"/models/([^/:]+)(?::|$)")


class GoogleExtractor:
    provider = Provider.GOOGLE
    def match(self, url: str) -> bool:
        host = urlparse(url).hostname or ""
        return (
            host == "generativelanguage.googleapis.com"
            or host == "aiplatform.googleapis.com"
            or host.endswith("-aiplatform.googleapis.com")
        )
    def extract_usage(self, body: Any) -> NormalizedUsage | None:
        if not isinstance(body, dict): return None
        meta = body.get("usageMetadata")
        if not isinstance(meta, dict): return None
        input_t = _as_int(meta.get("promptTokenCount"))
        # Grounding and tool calls add prompt tokens billed as input.
        tool_t = _as_int(meta.get("toolUsePromptTokenCount"))
        if tool_t:
            input_t = (input_t or 0) + tool_t
        output_t = _as_int(meta.get("candidatesTokenCount"))
        total_t = _as_int(meta.get("totalTokenCount"))
        if total_t is None and input_t is not None and output_t is not None:
            total_t = input_t + output_t
        native = _native_units(meta, exclude={
            "promptTokenCount", "candidatesTokenCount", "totalTokenCount",
            "thoughtsTokenCount", "cachedContentTokenCount", "toolUsePromptTokenCount",
        }) or {}
        native.update(_grounding_units([body]))
        return NormalizedUsage(input_tokens=input_t, output_tokens=output_t, total_tokens=total_t,
            reasoning_tokens=_as_int(meta.get("thoughtsTokenCount")),
            cache_read_tokens=_as_int(meta.get("cachedContentTokenCount")),
            native_units=native or None)
    def extract_stream_units(self, bodies: list[Any]) -> dict[str, int]:
        # A stream's distinct queries are the union across its chunks.
        return _grounding_units(bodies)
    def extract_model(self, req: Any, resp: Any) -> str | None:
        if isinstance(resp, dict) and isinstance(resp.get("modelVersion"), str):
            return resp["modelVersion"]
        return None
    def extract_call_model(self, url: str, req: Any, resp: Any) -> str | None:
        model = self.extract_model(req, resp)
        if model:
            return model
        match = _GOOGLE_MODEL_PATH.search(urlparse(url).path or "")
        return unquote(match.group(1)) if match else None


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


_BEDROCK_MODEL_PATH = re.compile(r"/model/([^/]+)/")


def _token_usage(input_t: int | None, output_t: int | None, *, total_t: int | None = None,
                 cache_read: int | None = None, cache_write: int | None = None,
                 native_units: dict[str, Any] | None = None) -> NormalizedUsage | None:
    if input_t is None and output_t is None and cache_read is None and cache_write is None and not native_units:
        return None
    if total_t is None and input_t is not None and output_t is not None:
        total_t = input_t + output_t
    return NormalizedUsage(input_tokens=input_t, output_tokens=output_t, total_tokens=total_t,
                           cache_read_tokens=cache_read, cache_creation_tokens=cache_write,
                           native_units=native_units or {})


class BedrockExtractor:
    provider = Provider.BEDROCK
    def match(self, url: str) -> bool:
        host = urlparse(url).hostname or ""
        return "bedrock-runtime" in host and host.endswith(".amazonaws.com")
    def extract_usage(self, body: Any) -> NormalizedUsage | None:
        if not isinstance(body, dict): return None
        found: NormalizedUsage | None = None
        usage = body.get("usage")
        if isinstance(usage, dict):
            # Converse reports camelCase; Anthropic InvokeModel bodies and
            # stream events report snake_case.
            reported = _token_usage(
                _first_int(usage, "inputTokens", "input_tokens"),
                _first_int(usage, "outputTokens", "output_tokens"),
                total_t=_first_int(usage, "totalTokens"),
                cache_read=_first_int(usage, "cacheReadInputTokens", "cache_read_input_tokens"),
                cache_write=_first_int(usage, "cacheWriteInputTokens", "cache_creation_input_tokens"),
                native_units=_native_units(usage, exclude={"inputTokens", "outputTokens", "totalTokens"}),
            )
            if reported:
                found = merge_usage(found, reported)
        metrics = body.get("amazon-bedrock-invocationMetrics")
        if isinstance(metrics, dict):
            reported = _token_usage(
                _first_int(metrics, "inputTokenCount"),
                _first_int(metrics, "outputTokenCount"),
                cache_read=_first_int(metrics, "cacheReadInputTokenCount"),
                cache_write=_first_int(metrics, "cacheWriteInputTokenCount"),
                native_units=_native_units(metrics, exclude={"inputTokenCount", "outputTokenCount"}),
            )
            if reported:
                found = merge_usage(found, reported)
        if found is None:
            # InvokeModel bodies for Llama and Titan text models.
            results = body.get("results")
            first = results[0] if isinstance(results, list) and results else None
            output_t = _first_int(body, "generation_token_count")
            found = _token_usage(
                _first_int(body, "prompt_token_count", "inputTextTokenCount"),
                output_t if output_t is not None else _first_int(first, "tokenCount"),
            )
        images = body.get("images")
        if isinstance(images, list) and images:
            found = found or NormalizedUsage()
            found.native_units = {**(found.native_units or {}), "images": len(images)}
        return found
    def extract_model(self, req: Any, resp: Any) -> str | None:
        if isinstance(req, dict) and isinstance(req.get("modelId"), str):
            return req["modelId"]
        return None
    def extract_call_model(self, url: str, req: Any, resp: Any) -> str | None:
        model = self.extract_model(req, resp)
        if model:
            return model
        match = _BEDROCK_MODEL_PATH.search(urlparse(url).path or "")
        return unquote(match.group(1)) if match else None
    def extract_call_usage(self, url: str, req: Any, resp: Any, headers: Any) -> NormalizedUsage | None:
        usage = self.extract_usage(resp)
        # InvokeModel reports the billed counts for every model family in
        # response headers; they win over whatever the body carried.
        counted = {
            "input_tokens": _header_number(headers, "x-amzn-bedrock-input-token-count"),
            "output_tokens": _header_number(headers, "x-amzn-bedrock-output-token-count"),
            "cache_read_tokens": _header_number(headers, "x-amzn-bedrock-cache-read-input-token-count"),
            "cache_creation_tokens": _header_number(headers, "x-amzn-bedrock-cache-write-input-token-count"),
        }
        if any(value is not None for value in counted.values()):
            usage = usage or NormalizedUsage()
            for name, value in counted.items():
                if value is not None:
                    setattr(usage, name, int(value))
            if usage.input_tokens is not None and usage.output_tokens is not None:
                usage.total_tokens = usage.input_tokens + usage.output_tokens
        return usage


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


# ── Non-token providers: speech, search and scraping ──

def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _request_text_length(req: Any, *keys: str) -> int | None:
    if not isinstance(req, dict):
        return None
    for key in keys:
        if isinstance(req.get(key), str):
            return len(req[key])
    return None


_ELEVENLABS_TTS_PATH = re.compile(r"^/v1/text-to-speech/[^/]+(?:/stream)?(?:/with-timestamps)?/?$")
_ELEVENLABS_STT_PATH = re.compile(r"^/v1/speech-to-text/?$")


def _transcript_seconds(resp: Any) -> int | float | None:
    if not isinstance(resp, dict):
        return None
    for key in ("audio_duration_secs", "audio_duration_seconds", "audio_duration", "duration_secs", "duration"):
        value = _as_number(resp.get(key))
        if value is not None and value >= 0:
            return value
    # Without a reported duration, the last word's end time bounds the audio.
    ends = []
    transcripts = resp.get("transcripts") if isinstance(resp.get("transcripts"), list) else [resp]
    for transcript in transcripts:
        words = transcript.get("words") if isinstance(transcript, dict) else None
        for word in words if isinstance(words, list) else ():
            end = _as_number(word.get("end")) if isinstance(word, dict) else None
            if end is not None:
                ends.append(end)
    return max(ends) if ends else None


class ElevenLabsExtractor:
    provider = Provider.ELEVENLABS
    def match(self, url: str) -> bool:
        return _host(url).endswith(".elevenlabs.io")
    def extract_usage(self, body: Any) -> NormalizedUsage | None:
        return None
    def extract_model(self, req: Any, resp: Any) -> str | None:
        if isinstance(req, dict) and isinstance(req.get("model_id"), str):
            return req["model_id"]
        return None
    def extract_call_model(self, url: str, req: Any, resp: Any) -> str | None:
        path = urlparse(url).path or ""
        if _ELEVENLABS_TTS_PATH.match(path):
            return self.extract_model(req, resp) or "eleven_multilingual_v2"
        if _ELEVENLABS_STT_PATH.match(path):
            return self.extract_model(req, resp) or "scribe_v1"
        return self.extract_model(req, resp)
    def extract_call_usage(self, url: str, req: Any, resp: Any, headers: Any) -> NormalizedUsage | None:
        path = urlparse(url).path or ""
        if _ELEVENLABS_TTS_PATH.match(path):
            characters = _header_number(headers, "character-cost", "x-character-count")
            if characters is None:
                characters = _request_text_length(req, "text")
            return NormalizedUsage(native_units={"characters": characters}) if characters is not None else None
        if _ELEVENLABS_STT_PATH.match(path):
            seconds = _transcript_seconds(resp)
            return NormalizedUsage(native_units={"audioSeconds": seconds}) if seconds is not None else None
        return None


class RumikExtractor:
    provider = Provider.RUMIK
    def match(self, url: str) -> bool:
        host = _host(url)
        return host == "rumik.ai" or host.endswith(".rumik.ai")
    def extract_usage(self, body: Any) -> NormalizedUsage | None:
        return None
    def extract_model(self, req: Any, resp: Any) -> str | None:
        if isinstance(req, dict) and isinstance(req.get("model"), str):
            return req["model"]
        return None
    def extract_call_model(self, url: str, req: Any, resp: Any) -> str | None:
        return self.extract_model(req, resp) or "mulberry"
    def extract_call_usage(self, url: str, req: Any, resp: Any, headers: Any) -> NormalizedUsage | None:
        characters = _request_text_length(req, "text", "input")
        return NormalizedUsage(native_units={"characters": characters}) if characters is not None else None


_EXA_SEARCH_TYPES = {
    "instant": "search-instant",
    "deep": "search-deep",
    "deep-lite": "search-deep",
    "deep-reasoning": "search-deep-reasoning",
}


class ExaExtractor:
    provider = Provider.EXA
    def match(self, url: str) -> bool:
        return _host(url) == "api.exa.ai"
    def extract_usage(self, body: Any) -> NormalizedUsage | None:
        cost = body.get("costDollars") if isinstance(body, dict) else None
        total = _as_number(cost.get("total")) if isinstance(cost, dict) else None
        return NormalizedUsage(cost_usd=float(total)) if total is not None else None
    def extract_model(self, req: Any, resp: Any) -> str | None:
        return None
    def extract_call_model(self, url: str, req: Any, resp: Any) -> str | None:
        segments = _path_segments(url)
        endpoint = segments[-1] if segments else None
        if endpoint == "search":
            search_type = req.get("type") if isinstance(req, dict) else None
            return _EXA_SEARCH_TYPES.get(search_type, "search") if isinstance(search_type, str) else "search"
        if endpoint == "findSimilar":
            return "find-similar"
        return endpoint
    def extract_call_usage(self, url: str, req: Any, resp: Any, headers: Any) -> NormalizedUsage | None:
        model = self.extract_call_model(url, req, resp)
        units: dict[str, Any] = {}
        if model and (model.startswith("search") or model == "find-similar"):
            results = _as_int(req.get("numResults")) if isinstance(req, dict) else None
            units = {"searchQueries": 1, "searchResults": results if results is not None and results >= 0 else 10}
        elif model == "contents" and isinstance(resp, dict) and isinstance(resp.get("results"), list):
            units = {"pages": len(resp["results"])}
        reported = self.extract_usage(resp)
        if not units and reported is None:
            return None
        return NormalizedUsage(native_units=units, cost_usd=reported.cost_usd if reported else None)


def _firecrawl_credits(body: Any) -> int | float | None:
    if not isinstance(body, dict):
        return None
    data = body.get("data") if isinstance(body.get("data"), dict) else {}
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    for value in (body.get("creditsUsed"), metadata.get("creditsUsed"), data.get("creditsUsed")):
        number = _as_number(value)
        if number is not None and number >= 0:
            return number
    return None


class FirecrawlExtractor:
    provider = Provider.FIRECRAWL
    def match(self, url: str) -> bool:
        return _host(url) == "api.firecrawl.dev"
    @staticmethod
    def _endpoint(url: str) -> tuple[str | None, bool]:
        """The endpoint after the version segment, and whether the call looks
        up an existing job (``/v2/crawl/{id}``) rather than starting work."""
        segments = _path_segments(url)
        if segments and re.fullmatch(r"v\d+", segments[0]):
            segments = segments[1:]
        if not segments:
            return None, False
        if segments[0] == "batch" and len(segments) > 1:
            return f"batch-{segments[1]}", len(segments) > 2
        return segments[0], len(segments) > 1
    def extract_usage(self, body: Any) -> NormalizedUsage | None:
        credits = _firecrawl_credits(body)
        return NormalizedUsage(native_units={"credits": credits}) if credits is not None else None
    def extract_model(self, req: Any, resp: Any) -> str | None:
        return None
    def extract_call_model(self, url: str, req: Any, resp: Any) -> str | None:
        return self._endpoint(url)[0]
    def extract_call_usage(self, url: str, req: Any, resp: Any, headers: Any) -> NormalizedUsage | None:
        endpoint, lookup = self._endpoint(url)
        if lookup:
            # Job status polls repeat the job's running credit total; counting
            # each poll would charge the same work many times.
            return None
        units: dict[str, Any] = {}
        credits = _firecrawl_credits(resp)
        if credits is None and endpoint in ("scrape", "map"):
            credits = 1
        if credits is not None:
            units["credits"] = credits
        if endpoint == "search":
            limit = _as_int(req.get("limit")) if isinstance(req, dict) else None
            units["searchQueries"] = 1
            units["searchResults"] = limit if limit is not None and limit >= 0 else 5
        return NormalizedUsage(native_units=units) if units else None


class TypesafeExtractor:
    provider = Provider.TYPESAFE
    def match(self, url: str) -> bool:
        return _host(url) == "api.typesafe.ai"
    def extract_usage(self, body: Any) -> NormalizedUsage | None:
        return _extract_openai_usage(body)
    def extract_model(self, req: Any, resp: Any) -> str | None:
        for body in (req, resp):
            if isinstance(body, dict) and isinstance(body.get("model"), str) and body["model"]:
                return body["model"]
        return None
    def extract_call_model(self, url: str, req: Any, resp: Any) -> str | None:
        return self.extract_model(req, resp) or "jev-latest"


# Registry
_extractors: list[ProviderExtractor] = [
    OpenAIExtractor(),
    AnthropicExtractor(),
    GoogleExtractor(),
    DeepSeekExtractor(),
    BedrockExtractor(),
    CohereExtractor(),
    ElevenLabsExtractor(),
    RumikExtractor(),
    ExaExtractor(),
    FirecrawlExtractor(),
    TypesafeExtractor(),
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
