# costkey

> Sentry for AI costs. Track every LLM call's cost, tokens, and latency with one line of code.

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

## How It Works

CostKey patches `httpx` and `requests` — the HTTP clients that every AI SDK uses under the hood. When your code calls any AI provider, CostKey automatically:

1. **Detects** the AI provider from the URL
2. **Extracts** token usage from the response
3. **Captures** a stack trace (which function, which file, which line)
4. **Computes** cost using built-in pricing for 30+ models
5. **Ships** the event to your CostKey dashboard (async, non-blocking)

## Tracing

```python
with costkey.start_trace(name="POST /api/search"):
    intent = classify_intent(query)
    results = search(query)
    summary = summarize(results)
# All 3 AI calls grouped under one trace
```

## Manual Context

```python
with costkey.with_context(task="summarize", team="search"):
    response = openai.chat.completions.create(...)
```

## Privacy

- Never captures API keys — request headers are never read
- Auto-scrubs credentials from request/response bodies
- `before_send` hook for custom PII scrubbing

## License

MIT
