"""The hosts CostKey captures.

This is the only provider knowledge in the SDK: whether a URL belongs to a
host whose calls are captured, and the label of the host list it matched.
Everything else (provider, model, usage, request ids) is derived on the
server from the raw telemetry.
"""
from __future__ import annotations

import json

import threading
from typing import Any
from urllib.parse import urlsplit

from costkey.types import Provider

# Exact hostnames → host-list label.
_EXACT_HOSTS: dict[str, str] = {
    "api.openai.com": Provider.OPENAI.value,
    "api.anthropic.com": Provider.ANTHROPIC.value,
    "generativelanguage.googleapis.com": Provider.GOOGLE.value,
    "aiplatform.googleapis.com": Provider.GOOGLE.value,
    "api.deepseek.com": Provider.DEEPSEEK.value,
    "api.cohere.com": Provider.COHERE.value,
    "api.exa.ai": Provider.EXA.value,
    "api.firecrawl.dev": Provider.FIRECRAWL.value,
    "api.typesafe.ai": Provider.TYPESAFE.value,
    "openrouter.ai": Provider.OPENROUTER.value,
    "api.x.ai": Provider.XAI.value,
    "api.grok.xai.com": Provider.XAI.value,
    "api.groq.com": Provider.GROQ.value,
    "api.mistral.ai": Provider.MISTRAL.value,
    "api.together.xyz": Provider.TOGETHER.value,
    "api.fireworks.ai": Provider.FIREWORKS.value,
    "api.perplexity.ai": Provider.PERPLEXITY.value,
    "api.cerebras.ai": Provider.CEREBRAS.value,
    "api.deepinfra.com": Provider.DEEPINFRA.value,
    "integrate.api.nvidia.com": Provider.NVIDIA.value,
    "api.sambanova.ai": Provider.SAMBANOVA.value,
    "api.novita.ai": Provider.NOVITA.value,
    "api.hyperbolic.xyz": Provider.HYPERBOLIC.value,
    "api.studio.nebius.com": Provider.NEBIUS.value,
    "api.studio.nebius.ai": Provider.NEBIUS.value,
    "api.tokenfactory.nebius.com": Provider.NEBIUS.value,
    "dashscope.aliyuncs.com": Provider.ALIBABA.value,
    "dashscope-intl.aliyuncs.com": Provider.ALIBABA.value,
    "dashscope-us.aliyuncs.com": Provider.ALIBABA.value,
    "api.moonshot.ai": Provider.MOONSHOT.value,
    "api.moonshot.cn": Provider.MOONSHOT.value,
    "api.z.ai": Provider.ZAI.value,
    "open.bigmodel.cn": Provider.ZAI.value,
    "api.minimax.io": Provider.MINIMAX.value,
    "api.minimaxi.com": Provider.MINIMAX.value,
    "api.minimax.chat": Provider.MINIMAX.value,
    "router.huggingface.co": Provider.HUGGINGFACE.value,
    "api-inference.huggingface.co": Provider.HUGGINGFACE.value,
    "inference.baseten.co": Provider.BASETEN.value,
    "api.lambda.ai": Provider.LAMBDA.value,
    "api.featherless.ai": Provider.FEATHERLESS.value,
    "api.friendli.ai": Provider.FRIENDLI.value,
    "api.ai21.com": Provider.AI21.value,
    "api.upstage.ai": Provider.UPSTAGE.value,
    "api.sarvam.ai": Provider.SARVAM.value,
    "ai-gateway.vercel.sh": Provider.VERCEL.value,
    "api.deepgram.com": Provider.DEEPGRAM.value,
    "api.assemblyai.com": Provider.ASSEMBLYAI.value,
    "api.eu.assemblyai.com": Provider.ASSEMBLYAI.value,
    "api.cartesia.ai": Provider.CARTESIA.value,
    "api.voyageai.com": Provider.VOYAGE.value,
    "api.jina.ai": Provider.JINA.value,
    "api.tavily.com": Provider.TAVILY.value,
    "rumik.ai": Provider.RUMIK.value,
    # AI gateways: labelled by the gateway, never relabelled to the upstream.
    "api.portkey.ai": "portkey",
    "oai.helicone.ai": "helicone",
    "anthropic.helicone.ai": "helicone",
    "gateway.helicone.ai": "helicone",
}

# Hostname suffixes → host-list label.
_SUFFIX_HOSTS: tuple[tuple[str, str], ...] = (
    (".openai.azure.com", Provider.OPENAI.value),
    (".services.ai.azure.com", Provider.OPENAI.value),
    (".cognitiveservices.azure.com", Provider.OPENAI.value),
    ("-aiplatform.googleapis.com", Provider.GOOGLE.value),
    (".elevenlabs.io", Provider.ELEVENLABS.value),
    (".rumik.ai", Provider.RUMIK.value),
)

_OLLAMA_HOSTS = frozenset({"localhost", "127.0.0.1"})

# Extra hostnames from the DSN-authenticated project config. These only widen
# what is captured; how their bodies are read is server configuration.
_runtime_lock = threading.Lock()
_runtime_hosts: dict[str, str] = {}


def match_provider(url: str) -> str | None:
    """The host-list label for a captured URL, or ``None`` when not captured."""
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
    except Exception:
        return None
    if not host:
        return None
    if host == "api.openai.com" and (parsed.path or "").startswith("/v1/traces/ingest"):
        # The OpenAI Agents SDK's own trace exporter, not a model call.
        return None
    label = _EXACT_HOSTS.get(host)
    if label:
        return label
    for suffix, suffix_label in _SUFFIX_HOSTS:
        if host.endswith(suffix):
            return suffix_label
    if host == "api.cloudflare.com" and "/ai/" in (parsed.path or ""):
        return Provider.CLOUDFLARE.value
    if "bedrock-runtime" in host and host.endswith(".amazonaws.com"):
        return Provider.BEDROCK.value
    if host in _OLLAMA_HOSTS:
        try:
            if parsed.port == 11434:
                return Provider.OLLAMA.value
        except ValueError:
            return None
    with _runtime_lock:
        for pattern, runtime_label in _runtime_hosts.items():
            if pattern.startswith("*.") and host.endswith(pattern[1:]):
                return runtime_label
            if host == pattern:
                return runtime_label
    return None


def apply_runtime_hosts(configs: Any) -> None:
    """Replace the project's configured custom hosts (capture only)."""
    hosts: dict[str, str] = {}
    for config in configs if isinstance(configs, list) else ():
        if not isinstance(config, dict):
            continue
        hostname = str(config.get("hostname") or "").lower()
        hostname = hostname.removeprefix("https://").removeprefix("http://").split("/", 1)[0]
        if hostname:
            hosts[hostname] = str(config.get("id") or config.get("name") or hostname)
    with _runtime_lock:
        _runtime_hosts.clear()
        _runtime_hosts.update(hosts)


def clear_runtime_hosts() -> None:
    apply_runtime_hosts([])


__all__ = ["match_provider", "apply_runtime_hosts", "clear_runtime_hosts"]


# Hosts that only report usage on a streamed chat completion when the request
# asks for it (stream_options.include_usage). "always" hosts get the flag on
# any streamed request; "chat" hosts only on chat-shaped ones. Adding the flag
# makes the provider report usage; it is not processing. Mirrors the TS SDK.
_STREAM_USAGE_ALWAYS = frozenset({"openrouter.ai", "api.x.ai", "api.grok.xai.com"})
_STREAM_USAGE_CHAT = frozenset({
    "api.openai.com", "ai-gateway.vercel.sh", "router.huggingface.co", "api-inference.huggingface.co",
    "api.portkey.ai", "oai.helicone.ai", "gateway.helicone.ai", "api.deepinfra.com",
    "integrate.api.nvidia.com", "api.sambanova.ai", "api.novita.ai", "api.hyperbolic.xyz",
    "api.studio.nebius.com", "api.studio.nebius.ai", "api.tokenfactory.nebius.com",
    "inference.baseten.co", "api.lambda.ai", "api.featherless.ai", "api.friendli.ai",
    "dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com", "dashscope-us.aliyuncs.com",
})
_STREAM_USAGE_CHAT_SUFFIXES = (".openai.azure.com", ".services.ai.azure.com", ".cognitiveservices.azure.com")


def stream_usage_mode(url: str) -> str | None:
    """"always", "chat" or None: whether a streamed request should ask for usage."""
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
    except Exception:
        return None
    if host in _STREAM_USAGE_ALWAYS:
        return "always"
    if host in _STREAM_USAGE_CHAT or host.endswith(_STREAM_USAGE_CHAT_SUFFIXES):
        return "chat"
    if host in _OLLAMA_HOSTS:
        try:
            return "chat" if parsed.port == 11434 else None
        except ValueError:
            return None
    return None


def inject_stream_usage(url: str, body: Any) -> Any:
    """The request body with stream_options.include_usage set, or ``body`` unchanged."""
    mode = stream_usage_mode(url)
    if not mode or not isinstance(body, dict) or not body.get("stream"):
        return body
    if mode == "chat" and "messages" not in body and "prompt" not in body:
        return body
    existing = body.get("stream_options")
    options = dict(existing) if isinstance(existing, dict) else {}
    if options.get("include_usage") is True:
        return body
    options["include_usage"] = True
    return {**body, "stream_options": options}


def inject_stream_usage_bytes(url: str, content: Any) -> Any:
    """Same as :func:`inject_stream_usage` for a JSON byte or str body."""
    if not stream_usage_mode(url) or not isinstance(content, (bytes, bytearray, str)) or not content:
        return content
    try:
        body = json.loads(content)
    except Exception:
        return content
    updated = inject_stream_usage(url, body)
    if updated is body:
        return content
    encoded = json.dumps(updated, separators=(",", ":"))
    return encoded if isinstance(content, str) else encoded.encode("utf-8")
