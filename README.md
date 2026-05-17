# CostKey for Python

[![PyPI version](https://img.shields.io/pypi/v/costkey.svg)](https://pypi.org/project/costkey/)
[![CI](https://github.com/costkey/costkey-python/actions/workflows/ci.yml/badge.svg)](https://github.com/costkey/costkey-python/actions/workflows/ci.yml)
[![MIT](https://img.shields.io/badge/license-MIT-black.svg)](LICENSE)

AI cost observability for Python. CostKey tracks every LLM call's cost, tokens, latency, prompts, tool calls, and code location without proxying your traffic.

```bash
pipx run costkey setup
```

Your provider dashboard says you spent $2,000. CostKey tells you `generate_summary()` in `app/search.py:47` spent $1,200 of that.

## Why CostKey

Most AI observability tools make you route traffic through a proxy or wrap every provider client. CostKey instruments `httpx` and `requests` in-process, detects AI provider requests, captures the useful metadata, and ships events asynchronously to the CostKey dashboard.

- **No proxy** - zero added network hop and no provider migration.
- **No wrappers** - keep using OpenAI, Anthropic, Gemini, LangChain, LiteLLM, or your own gateway.
- **Code attribution** - see function, file, line, release, and stack trace for expensive calls.
- **Prompt and tool visibility** - inspect system prompts, messages, tool definitions, tool calls, tool results, citations, and provider metadata.
- **Trace grouping** - group calls from the same user action, request, or agent run.
- **Call graph analysis** - the SDK scans your project structure so the dashboard highlights business logic, not thin wrappers.
- **Privacy controls** - credentials are scrubbed and `before_send` can redact or drop events.

## Quick Start

Run setup in your app:

```bash
pipx run costkey setup
```

The CLI opens browser auth, creates a project, writes `COSTKEY_DSN` to `.env`, detects FastAPI, Django, Flask, and plain Python apps, and writes `.costkey/setup.md` with the exact entrypoint instructions.

Manual setup is still tiny:

```bash
pip install costkey
```

```py
import os
import costkey

costkey.init(dsn=os.environ["COSTKEY_DSN"])
```

Initialize CostKey before your first AI call. After that, provider calls made through `httpx` or `requests` are tracked automatically.

## No-Code Bootstrap

For server processes where you do not want to touch application code, `costkey setup` writes `.costkey/sitecustomize.py`.

```bash
PYTHONPATH="$PWD/.costkey:$PYTHONPATH" uvicorn app:app
```

The bootstrap reads:

- `COSTKEY_DSN`
- `COSTKEY_RELEASE`
- `COSTKEY_CAPTURE_BODY=false`
- `COSTKEY_DEBUG=true`

## What Shows Up

Every captured event can include:

| Data | Example |
| --- | --- |
| Cost | `$0.0184` for a single model call |
| Usage | input, output, cache, reasoning, total tokens |
| Latency | total duration, TTFT, tokens/sec for streams |
| Code location | `app/search.py:47 generate_summary()` |
| Trace | `POST /api/search`, `agent:research-summary` |
| Prompt metadata | system/developer prompt, user messages, response text |
| Tool metadata | tool definitions, tool choice, tool calls, tool results |
| Provider metadata | citations, web search usage, raw provider fields |
| Release | deploy SHA or version for cost-per-deploy |

## Frameworks

`costkey setup` detects and guides setup for:

- FastAPI
- Django
- Flask
- plain Python scripts and services

CostKey works with provider clients that use `httpx` or `requests` under the hood.

## Traces and Context

Group all calls from one user action:

```py
with costkey.start_trace(name="POST /api/search"):
    intent = classify_intent(query)
    results = search_products(query, intent)
    summary = summarize_results(results)
```

Add business context for filtering and rollups:

```py
with costkey.with_context(feature="search", team="growth", customer_id="cus_123"):
    client.messages.create(
        model="claude-sonnet-4-5-20250514",
        messages=messages,
        max_tokens=1024,
    )
```

## Streaming

Streaming responses are fully supported. The SDK wraps `httpx` response iteration to capture:

- TTFT, time to first token
- output tokens per second
- stream duration
- final usage chunks when providers emit them

Non-AI HTTP calls pass through untouched.

## Privacy

CostKey never reads request headers, so API keys are not captured. Request and response bodies are captured by default because they power debugging and prompt/tool inspection.

Redact before anything leaves your process:

```py
def scrub(event):
    if event.request_body and "customer_email" in event.request_body:
        event.request_body["customer_email"] = "[redacted]"
    return event

costkey.init(dsn=os.environ["COSTKEY_DSN"], before_send=scrub)
```

Disable body capture entirely:

```py
costkey.init(
    dsn=os.environ["COSTKEY_DSN"],
    capture_body=False,
)
```

## Providers

Built-in detection covers OpenAI, Anthropic, Google Gemini, Google Vertex AI, Azure OpenAI, Groq, xAI, Mistral, DeepSeek, Cohere, Together, Fireworks, Perplexity, Cerebras, OpenRouter, and AWS Bedrock.

Add your own provider or internal gateway:

```py
from costkey.providers import ProviderExtractor, NormalizedUsage

class InternalGatewayExtractor:
    provider = "internal"

    def match(self, url: str) -> bool:
        return "llm.internal.example.com" in url

    def extract_usage(self, body) -> NormalizedUsage | None:
        usage = body.get("usage") if isinstance(body, dict) else None
        if not usage:
            return None
        return NormalizedUsage(
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

costkey.register_extractor(InternalGatewayExtractor())
```

## API

| API | Purpose |
| --- | --- |
| `costkey.init(dsn, **options)` | Initialize once at process startup |
| `costkey.shutdown()` | Flush and restore patched clients |
| `costkey.flush()` | Flush buffered events |
| `costkey.with_context(**context)` | Attach metadata to nested AI calls |
| `costkey.start_trace(name=None, trace_id=None)` | Group calls into one trace |
| `costkey.register_extractor(extractor)` | Add custom provider detection |
| `costkey.register_pricing(model, pricing)` | Add custom model pricing |

## Development

```bash
python -m pip install -e . pytest build twine
python -m pytest
python -m build
python -m twine check dist/*
```

## Release

Publishing is done from GitHub Releases only.

1. Update `pyproject.toml` and `costkey/__init__.py` versions.
2. Merge to `main`.
3. Create a GitHub Release tagged `vX.Y.Z`.
4. GitHub Actions runs tests, builds, verifies the tag matches the package version, and publishes to PyPI.

PyPI must have trusted publishing configured for `costkey/costkey-python` and the `pypi` GitHub environment.

## Links

- Docs: https://costkey.dev/docs
- Dashboard: https://app.costkey.dev
- PyPI: https://pypi.org/project/costkey/
- TypeScript SDK: https://github.com/costkey/costkey

If CostKey helps you understand your AI bill, starring the repo helps more developers find it.

## License

MIT
