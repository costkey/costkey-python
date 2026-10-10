# CostKey for Python

[![PyPI version](https://img.shields.io/pypi/v/costkey.svg)](https://pypi.org/project/costkey/)
[![CI](https://github.com/costkey/costkey-python/actions/workflows/ci.yml/badge.svg)](https://github.com/costkey/costkey-python/actions/workflows/ci.yml)
[![MIT](https://img.shields.io/badge/license-MIT-black.svg)](LICENSE)

AI cost observability for Python. CostKey tracks supported provider calls'
cost, tokens, latency, prompts, tool calls, and code location without adding a
provider-routing hop. Endpoint coverage is explicit; framework-native
components use `record_usage` when HTTP capture does not apply.

```bash
pipx run costkey setup
```

Your provider dashboard says you spent $2,000. CostKey tells you `generate_summary()` in `app/search.py:47` spent $1,200 of that.

## Why CostKey

Most AI observability tools make you route traffic through a proxy or wrap every provider client. CostKey instruments supported `httpx` and `requests` calls in-process, detects configured AI provider endpoints, captures useful metadata, and ships events asynchronously to the CostKey dashboard.

- **No extra provider-routing hop** - provider traffic stays on its existing path; CostKey adds local interception and asynchronous telemetry. Explicit enforcement performs a separate admission decision.
- **No provider-client wrappers** - keep using OpenAI, Anthropic, Gemini, LangChain, LiteLLM, or your own gateway.
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

Initialize CostKey before your first AI call. After that, supported provider
calls made through `httpx` or `requests` are tracked automatically.

## No-Code Bootstrap

For server processes where you do not want to touch application code, `costkey setup` writes `.costkey/sitecustomize.py`.

```bash
PYTHONPATH="$PWD/.costkey:$PYTHONPATH" uvicorn app:app
```

The bootstrap reads:

- `COSTKEY_DSN`
- `COSTKEY_RELEASE`
- `COSTKEY_CAPTURE_BODY=false` (unset leaves the server project setting in control)
- `COSTKEY_DEBUG=true`

## What Shows Up

Captured provider events can include:

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

## Agents

Record an agent run as a tree of spans. Model calls are attributed to the
innermost span, and the run page shows cost and latency at each level.

```py
with costkey.agent("dash", user_id=user.id) as span:
    span.set_attribute("model", "gpt-5.1")
    with costkey.tool("get_holdings"):
        holdings = get_holdings(user.id)
```

Framework hooks:

```py
# OpenAI Agents SDK
from costkey.integrations.openai_agents import CostKeyRunHooks
result = await Runner.run(agent, user_input, hooks=CostKeyRunHooks())

# browser-use
from costkey.integrations.browser_use import step_hooks
on_step_start, on_step_end = step_hooks()
with costkey.agent("browser-agent"):
    await browser_agent.run(on_step_start=on_step_start, on_step_end=on_step_end)
```

See [Agents and tools](https://costkey.dev/docs/agents).

## Grouped work and explicit usage

Use one stable group name for repeatable work; each invocation gets a fresh
execution ID and lifecycle receipt. If the runner knows item counts, pass
`expected_items`, `completed_items`, and `failed_items`; they are carried on
both lifecycle records:

```py
with costkey.start_execution(
    name="nightly-document-import",
    kind="batch",
    expected_items=100,
    completed_items=98,
    failed_items=2,
):
    with costkey.start_trace(name="document-42"):
        import_document(42)
```

Adapters for components that are not observed through supported HTTP clients
report a caller-owned identity and monotonic revision. Cumulative snapshots
replace the prior snapshot; they are not summed:

```py
costkey.record_usage(
    event_id="voice-session-42:stt:r7",
    component_id="voice-session-42:stt",
    provider="deepgram",
    model="nova-3",
    usage={"audioSeconds": 12.4},
    aggregation="cumulative",
    revision=7,
)
```

For a verified LiveKit `session_usage_updated` callback, use
`costkey.livekit.record_livekit_usage(update, session_id=..., revision=...)`.
This reports the observed STT/LLM/TTS components under one session; it does
not intercept arbitrary voice traffic, and omitted components remain partial.

```py
from costkey.livekit import record_livekit_usage

@session.on("session_usage_updated")
def on_usage(ev):
    # ev.usage.model_usage is a cumulative list of typed provider/model rows.
    record_livekit_usage(
        ev,
        session_id=session.room.name,
        revision=next_revision(),  # supplied by the application adapter
        expected_components=("stt", "llm", "tts"),
    )
```

The adapter preserves Python's native duration unit (seconds), token fields,
characters and provider/model identity. It does not treat a final session
report as a second charge for the same revision.

LiveKit's usage classes do not include provider prices. Supply
`declared_amounts`/`declared_pricing_ids` when the application has an explicit
rate basis; these amounts are marked estimated and are never inferred from
duration or token counts.

When HTTP auto-capture already records the underlying LLM requests, use
`exclude_components={"llm"}` so the session aggregate does not double-count
them; session totals and individual request IDs are different granularities.
If the same LLM provider request is also auto-captured, make the session
snapshot the source of charge for that scoped request with
`with costkey.with_context(costkeyCapture="manual"):` so it is not counted at
both HTTP and session granularity. Only use this when the integration has
verified that the session snapshot covers that request.

To cross a trusted service or queue boundary, explicitly attach the
allowlisted carrier returned by `costkey.export_context()` (or its JSON header
form from `costkey.export_context_headers()`) and import it with
`costkey.with_context(...)`. Context is never automatically added to unrelated
outbound URLs.

## Streaming

Streaming responses are fully supported. The SDK wraps `httpx` response iteration to capture:

- TTFT, time to first token
- output tokens per second
- stream duration
- final usage chunks when providers emit them

Non-AI HTTP calls pass through untouched.

## Privacy

CostKey does not capture provider authorization headers. It reads only the
explicit `x-costkey-context`/legacy trace headers used for a trusted handoff,
then removes those headers before the provider request. Body and prompt
capture starts metadata-only until the project runtime config confirms text
capture; an explicit `capture_body=False` always wins. The DSN config is
refreshed periodically (customize with `config_refresh_interval`) so a remote
revocation returns future calls to metadata-only capture.

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
| `costkey.agent / step / tool(name, **options)` | Agent, step or tool span (context manager or decorator) |
| `costkey.span(name, kind=..., **options)` | A span of any kind |
| `costkey.start_trace(name=None, trace_id=None)` | Group calls into one trace |
| `costkey.start_execution(name, kind="batch", execution_id=None)` | Group repeatable work with a fresh invocation ID |
| `costkey.record_usage(...)` | Report deduplicated external/native usage with delta or cumulative semantics |
| `costkey.export_context()` | Export an explicit allowlisted carrier for trusted handoff |
| `costkey.export_context_headers()` | Encode that carrier for an explicitly trusted request |
| `costkey.get_runtime_config_status()` | Inspect metadata-safe runtime capture config state |
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
