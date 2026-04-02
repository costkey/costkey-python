# costkey

> AI cost observability. Track every LLM call's cost, tokens, and latency with one line of code.

## Install

```bash
pip install costkey
```

## Quick Start

```python
import costkey

costkey.init(dsn="https://ck_your_key@app.costkey.dev/your-project")

# That's it. Every AI call is now tracked automatically.
```

**No wrapping. No per-client setup. No manual tagging.** CostKey patches `httpx` and `requests` and auto-detects AI provider calls.

## What You Get (Zero Config)

- **Cost tracking** — server-side cost computation using live pricing (50+ models)
- **Stack trace attribution** — see which business logic function made each AI call
- **Streaming metrics** — TTFT (time to first token), TPS (tokens/sec), chunk count
- **Request tracing** — group AI calls per request with `start_trace()`
- **Feature detection** — call chains analyzed to detect logical "features" in your code
- **Body capture** — input prompts + output completions captured and scrubbed
- **Credential scrubbing** — API keys, JWTs, and secrets auto-redacted

## How It Works

CostKey patches `httpx.Client.send` and `requests.Session.send`. When your code calls any AI provider:

1. **Detects** the provider from the URL (15 providers supported)
2. **Extracts** token usage from the response (streaming + non-streaming)
3. **Captures** a stack trace — filters library frames, shows your business logic
4. **Measures** streaming timing (TTFT, TPS) by wrapping response iteration
5. **Ships** the event async to your CostKey dashboard — server calculates cost

Non-AI HTTP calls pass through untouched with zero overhead.

## Streaming Support

Streaming responses are fully supported. The SDK wraps `iter_bytes()`, `iter_lines()`, `iter_text()`, and `read()` on httpx responses to capture:

- **TTFT** — time from request to first token
- **TPS** — output tokens per second
- **Usage** — extracted from the final SSE chunk
- **Duration** — total stream time

Works with Anthropic, OpenAI, Google, and all SSE-based providers.

## Sourcemaps (JavaScript/TypeScript)

For minified JS/TS builds, upload sourcemaps so stack traces show original source:

```python
costkey.init(
    dsn="https://ck_your_key@app.costkey.dev/your-project",
    release="v1.2.3"  # matches uploaded sourcemaps
)
```

## Tracing

```python
with costkey.start_trace(name="POST /api/search"):
    intent = classify_intent(query)
    results = search(query, intent)
    summary = summarize_results(results)

# Dashboard shows one trace with all calls and total cost
```

## Manual Context

```python
with costkey.with_context(task="summarize", team="search"):
    response = client.messages.create(...)
```

## Supported Providers

| Provider | Hostname | Streaming |
|---|---|---|
| OpenAI | `api.openai.com` | Yes |
| Anthropic | `api.anthropic.com` | Yes |
| Google Gemini | `generativelanguage.googleapis.com` | Yes |
| Azure OpenAI | `*.openai.azure.com` | Yes |
| Google Vertex AI | `*-aiplatform.googleapis.com` | Yes |
| Groq | `api.groq.com` | Yes |
| xAI (Grok) | `api.x.ai` | Yes |
| Mistral | `api.mistral.ai` | Yes |
| DeepSeek | `api.deepseek.com` | Yes |
| Cohere | `api.cohere.com` | Yes |
| Together AI | `api.together.xyz` | Yes |
| Fireworks | `api.fireworks.ai` | Yes |
| Perplexity | `api.perplexity.ai` | Yes |
| Cerebras | `api.cerebras.ai` | Yes |
| OpenRouter | `openrouter.ai` | Yes |
| AWS Bedrock | `bedrock-runtime.*.amazonaws.com` | No |

Custom providers: `costkey.register_extractor(extractor)`

## Privacy & Security

- **Never captures API keys** — request headers are never read
- **Auto-scrubs credentials** from captured bodies (OpenAI keys, JWTs, tokens, etc.)
- **`before_send` hook** for custom PII scrubbing:

```python
def scrub(event):
    event.request_body = None  # strip prompts
    return event

costkey.init(dsn="...", before_send=scrub)
```

## API

| Function | Description |
|---|---|
| `costkey.init(dsn, **opts)` | Initialize. Call once at startup. |
| `costkey.with_context(**kwargs)` | Context manager for custom tags. |
| `costkey.start_trace(name)` | Context manager for request tracing. |
| `costkey.shutdown()` | Flush and restore original HTTP clients. |
| `costkey.flush()` | Force-flush pending events. |
| `costkey.register_extractor(ext)` | Add custom AI provider. |

### Options

| Option | Default | Description |
|---|---|---|
| `dsn` | required | `https://ck_key@app.costkey.dev/project` |
| `capture_body` | `True` | Capture request/response bodies |
| `release` | `None` | Release version for sourcemaps |
| `before_send` | `None` | Hook to modify/filter events |
| `max_batch_size` | `50` | Events per batch |
| `flush_interval` | `5.0` | Seconds between flushes |
| `debug` | `False` | Enable debug logging |

## License

MIT
