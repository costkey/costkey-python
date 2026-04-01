# costkey

> AI cost observability. Track every LLM call's cost, tokens, and latency with one line of code.

## Install

```bash
pip install costkey
```

## Quick Start

```python
import costkey

costkey.init(dsn="https://ck_your_key@costkey.dev/your-project")

# That's it. Every AI call is now tracked automatically.
# Works with OpenAI, Anthropic, Google Gemini, Azure OpenAI.
```

**No wrapping. No per-client setup. No manual tagging.** CostKey patches `httpx` and `requests` and auto-detects AI provider calls.

## What You Get (Zero Config)

All of these work automatically after `init()`:

- **Cost tracking** — per-call cost computed from built-in pricing (30+ models)
- **Stack trace attribution** — see which function, file, and line made each AI call
- **Request tracing** — group AI calls per request with `start_trace()`
- **Feature detection** — call chains are analyzed to detect logical "features" in your code
- **Credential scrubbing** — API keys, JWTs, and secrets are auto-redacted from captured bodies

## How It Works

CostKey patches `httpx.Client.send` and `requests.Session.send`. When your code calls any AI provider:

1. **Detects** the provider from the URL (OpenAI, Anthropic, Google, Azure)
2. **Extracts** token usage from the response
3. **Captures** a stack trace for automatic code attribution
4. **Computes** cost using built-in model pricing
5. **Ships** the event async to your CostKey dashboard

Non-AI HTTP calls pass through untouched with zero overhead.

## Tracing

```python
# Group all AI calls in a request into one trace
with costkey.start_trace(name="POST /api/search"):
    intent = classify_intent(query)        # AI call 1
    results = search(query, intent)
    summary = summarize_results(results)   # AI call 2
    reranked = rerank_results(results)     # AI call 3

# Dashboard shows one trace with all 3 calls and total cost
```

## Manual Context

```python
with costkey.with_context(task="summarize", team="search"):
    response = openai.chat.completions.create(...)
```

## Privacy & Security

- **Never captures API keys** — request headers are never read
- **Auto-scrubs credentials** from captured bodies (OpenAI keys, JWTs, tokens, etc.)
- **`before_send` hook** for custom PII scrubbing:

```python
def scrub(event):
    event.request_body = None  # strip prompts entirely
    return event

costkey.init(dsn="...", before_send=scrub)
```

## Supported Providers

| Provider | Auto-detected | Patches |
|---|---|---|
| OpenAI | `api.openai.com` | httpx, requests |
| Anthropic | `api.anthropic.com` | httpx, requests |
| Google Gemini | `generativelanguage.googleapis.com` | httpx, requests |
| Azure OpenAI | `*.openai.azure.com` | httpx, requests |
| Google Vertex AI | `*-aiplatform.googleapis.com` | httpx, requests |

## API

### `costkey.init(dsn, **options)` — Initialize. Call once at startup.
### `costkey.with_context(**kwargs)` — Context manager for custom tags.
### `costkey.start_trace(name)` — Context manager for request tracing.
### `costkey.shutdown()` — Flush and restore original HTTP clients.
### `costkey.register_extractor(extractor)` — Add custom AI provider.
### `costkey.register_pricing(model, ...)` — Add custom model pricing.

## Also available for TypeScript

```bash
npm install costkey
```

```typescript
import { CostKey } from 'costkey'
CostKey.init({ dsn: 'https://ck_...@costkey.dev/proj' })
```

## License

MIT
