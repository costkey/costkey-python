# CostKey for Python

[![PyPI version](https://img.shields.io/pypi/v/costkey.svg)](https://pypi.org/project/costkey/)
[![CI](https://github.com/costkey/costkey-python/actions/workflows/ci.yml/badge.svg)](https://github.com/costkey/costkey-python/actions/workflows/ci.yml)
[![MIT](https://img.shields.io/badge/license-MIT-black.svg)](LICENSE)

AI cost observability for Python. The SDK captures raw telemetry from AI
provider calls and agent frameworks, in process and without a provider-routing
hop, and sends it to CostKey. The CostKey server does everything else: provider
and model detection, usage extraction, pricing, deduplication and scoring.

```bash
pipx run costkey setup
```

Your provider dashboard says you spent $2,000. CostKey tells you `generate_summary()` in `app/search.py:47` spent $1,200 of that.

## Why CostKey

Most AI observability tools make you route traffic through a proxy or wrap every provider client. CostKey instruments `httpx`, `requests`, `aiohttp` and boto3/botocore Bedrock calls in-process, recognizes AI provider hosts, and ships what it saw asynchronously to the CostKey server.

- **No extra provider-routing hop** - provider traffic stays on its existing path; CostKey adds local interception and asynchronous telemetry. Explicit enforcement performs a separate admission decision.
- **No provider-client wrappers** - keep using OpenAI, Anthropic, Gemini, LangChain, LiteLLM, or your own gateway.
- **Telemetry only** - the SDK never prices, counts tokens or characters, guesses models, merges stream usage or classifies errors. It sends a content-free skeleton of each request and response; the server computes the rest, so pricing and provider fixes ship without an SDK upgrade.
- **Code attribution** - every call carries its raw stack, so the dashboard shows `generate_summary()` in `app/search.py:47`.
- **Agent frameworks** - spans for OpenAI Agents, LangChain/LangGraph, CrewAI, LlamaIndex, Pydantic AI, Google ADK, Claude Agent SDK, smolagents, AutoGen, AG2, DSPy, Strands and any OpenTelemetry tracer.
- **Privacy controls** - prompt and response text never leave the process unless the project enables body capture; credentials are scrubbed from URLs and headers, and `before_send` can redact or drop events.

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
calls made through `httpx`, `requests`, `aiohttp` (google-genai uses it when
installed) or boto3 `bedrock-runtime` clients are tracked automatically.

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

Every captured provider call carries `telemetry` (raw telemetry contract v1):

| Field | Contents |
| --- | --- |
| `request` | method, URL (credential query parameters removed), allowlisted headers (`content-type`, `anthropic-version`, `anthropic-beta`, `openai-beta`, `x-goog-api-client`, `x-stainless-lang`), body skeleton, byte count |
| `response` | status, headers (cookies, auth and secret token headers removed), body skeleton, byte count |
| `stream` | `sse` / `ndjson` / `json-array` / `aws-eventstream` chunk skeletons (first 200, or first 20 + last 50), real `chunkCount`, `firstChunkMs`, `lastChunkMs` |
| `error` | the raw exception class name (and errno name), never the message |
| `startedAt`, `durationMs` | timing |

A skeleton keeps the shape of a body and its numbers, and replaces text with
its length: `{"role": "user", "content": "hello"}` becomes
`{"role": {"$s": 4}, "content": {"$s": 5}}`. Strings are kept only under keys
such as `model`, `id`, `type`, `status`, `finish_reason` and `service_tier`
(200 characters at most). Arrays over 32 items keep 8 and a `{"$more": n}`
count (embedding vectors never ship), binary bodies become `{"$bytes": n}`,
uploads become `{"$multipart": [...]}`, depth is capped at 12 and each
skeleton at 64 KB. Calls also carry trace, span, context, release and the
call stack. The server derives provider, model, tokens, cost, latency and
TTFT from this.

## Frameworks

`costkey setup` detects and guides setup for FastAPI, Django, Flask and plain
Python scripts and services.

CostKey captures provider clients that use `httpx`, `requests` or `aiohttp`
under the hood, and boto3/botocore `bedrock-runtime` clients. Agent framework
integrations are listed under [Agents](#agents).

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

### Framework integrations

Each integration is an optional extra (`pip install "costkey[langchain]"`)
imported only from `costkey.integrations.<name>`; `import costkey` never
imports a framework. Integrations create spans with the framework's raw
attributes under `attributes.<framework>` (a skeleton unless body capture is
on). They never emit call events for in-process model calls, which the HTTP
capture already records; the Claude Agent SDK is the exception, because its
model calls run in a separate Claude Code process.

| Framework | Extra | What becomes a span |
| --- | --- | --- |
| OpenAI Agents SDK | `openai-agents` | traces, agents, turns, tools, handoffs, guardrails |
| LangChain / LangGraph | `langchain` / `langgraph` | chains, graph nodes, tools, retrievers, agents |
| CrewAI | `crewai` | crews, flows, agents, tasks, tools, memory and knowledge queries |
| LlamaIndex | `llama-index` | agents, workflows and steps, retrievers, tools, query and chat engines |
| Pydantic AI | `pydantic-ai` | agent runs, tool executions |
| Google ADK | `google-adk` | runner invocations, agents, tools |
| Claude Agent SDK | `claude-agent-sdk` | sessions, subagents, tool uses, plus usage call events |
| smolagents | `smolagents` | agent runs, action and planning steps |
| AutoGen AgentChat | `autogen` | `invoke_agent`, `execute_tool`, `create_agent` (via OpenTelemetry) |
| AG2 | `ag2` | agent turns, tool executions |
| DSPy | `dspy` | modules, ReAct agents, tools, evaluations, compiles |
| Strands Agents | `strands` | invocations, tools, multi-agent nodes |
| Any OpenTelemetry tracer | `opentelemetry` | every span, with raw OTel attributes |

```py
# OpenAI Agents SDK: a tracing processor covers every run
from costkey.integrations.openai_agents import instrument
instrument()
result = await Runner.run(agent, "What's the weather in SF?")
# or, with tracing disabled: Runner.run(agent, input, hooks=CostKeyRunHooks())
```

```py
# LangChain / LangGraph
from costkey.integrations.langchain import CostKeyCallbackHandler
result = graph.invoke(inputs, config={"callbacks": [CostKeyCallbackHandler()]})
```

```py
# CrewAI
from costkey.integrations.crewai import CostKeyCrewAIListener
listener = CostKeyCrewAIListener()  # registers on crewai_event_bus
crew.kickoff()
```

```py
# LlamaIndex
from costkey.integrations.llama_index import instrument
instrument()  # adds a span handler to the root dispatcher
response = index.as_query_engine().query("What changed?")
```

```py
# Pydantic AI
from costkey.integrations.pydantic_ai import CostKeyCapability
agent = Agent("openai:gpt-5", capabilities=[CostKeyCapability()])
result = await agent.run("Summarize the incident")
```

```py
# Google ADK
from costkey.integrations.google_adk import CostKeyPlugin
runner = InMemoryRunner(agent=root_agent, app_name="app", plugins=[CostKeyPlugin()])
```

```py
# Claude Agent SDK
from costkey.integrations.claude_agent_sdk import instrument
claude = instrument()
options = claude.with_hooks(ClaudeAgentOptions(model="claude-sonnet-4-5"))
async for message in query(prompt="Fix the failing test", options=options):
    ...
```

```py
# smolagents
from costkey.integrations.smolagents import instrument
agent = instrument(CodeAgent(tools=[...], model=model))
agent.run("task")
```

```py
# AutoGen AgentChat (traces through OpenTelemetry)
from costkey.integrations.autogen import instrument
instrument()
result = await AssistantAgent("assistant", model_client=client).run(task="...")
```

```py
# AG2
from costkey.integrations.ag2 import costkey_middleware
agent = Agent("researcher", config=config, middleware=[costkey_middleware("researcher")])
```

```py
# DSPy
from costkey.integrations.dspy import CostKeyCallback
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), callbacks=[CostKeyCallback()])
```

```py
# Strands Agents
from costkey.integrations.strands import CostKeyHooks
agent = Agent(model=model, tools=[...], hooks=[CostKeyHooks()])
```

```py
# Any OpenTelemetry tracer (OpenLLMetry, OpenInference, your own spans)
from costkey.integrations.opentelemetry import instrument
instrument()  # adds CostKeySpanProcessor to the global SDK TracerProvider
```

```py
# browser-use
from costkey.integrations.browser_use import step_hooks
on_step_start, on_step_end = step_hooks()
with costkey.agent("browser-agent"):
    await browser_agent.run(on_step_start=on_step_start, on_step_end=on_step_end)
```

Notes:

- **OpenAI Agents SDK.** Use the tracing processor *or* the hooks, not both.
  The SDK runs hooks in child tasks, so with hooks the span tree is recorded
  but model calls attach to the span around `Runner.run`. The processor needs
  the Agents SDK's tracing enabled. Its own trace exporter
  (`/v1/traces/ingest`) is never captured as a model call.
- **Claude Agent SDK.** Each message that carries usage (`usage` or
  `modelUsage` on the message, `message`, `event` or `event.message`) is sent
  as a call event with `telemetry.source = "claude-agent-sdk"` and the raw
  message skeleton in `telemetry.response.body`. `instrument()` reads the raw
  CLI messages by wrapping the SDK's internal `Query.receive_messages`.
- **DSPy 3.4** sends model requests through its vendored `lm15` HTTP client
  (raw sockets). `CostKeyCallback` also patches that transport so those calls
  produce the same raw telemetry.
- **smolagents** reports steps only after they finish, so steps are sent as
  completed spans with the step's own timing; model calls sit under the agent
  span and the server places them in steps by time.
- **CrewAI** handlers run on a thread pool; span ids derive from CrewAI's
  event ids, so the tree is right even when handlers run out of order.

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

Components that are not observed through supported HTTP clients can report
usage explicitly. The SDK sends exactly what you pass: it fills in no
defaults (charge id, currency, amount kind) and derives nothing. Deduplication,
revision fences and pricing happen on the server, which requires an
`event_id`, a `component_id` and a non-negative integer `revision`.
Cumulative snapshots replace the prior snapshot on the server; they are not
summed:

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

For LiveKit Agents, send each `session_usage_updated` snapshot as it is. The
SDK sends the raw update as a skeleton (numbers, `provider`, `model` and
`type` kept); the server classifies the STT/LLM/TTS rows and sums them.

```py
from costkey.livekit import record_livekit_usage

@session.on("session_usage_updated")
def on_usage(ev):
    record_livekit_usage(
        ev,
        session_id=session.room.name,
        revision=next_revision(),  # monotonic, supplied by the application
        exclude_components=("llm",),  # optional: forwarded to the server
    )
```

`expected_components`, `exclude_components`, `declared_amounts`,
`declared_pricing_ids` and `declared_billing_units` are forwarded verbatim in
`telemetry.livekit` for the server to apply. If the same LLM request is also
auto-captured and the session snapshot should own its charge, scope it with
`with costkey.with_context(costkeyCapture="manual"):`.

To cross a trusted service or queue boundary, explicitly attach the
allowlisted carrier returned by `costkey.export_context()` (or its JSON header
form from `costkey.export_context_headers()`) and import it with
`costkey.with_context(...)`. Context is never automatically added to unrelated
outbound URLs.

## Streaming

The SDK records the bytes your code reads from `httpx`, `requests` and
`aiohttp` streams (and the events of boto3 Bedrock streams) without buffering
ahead of you. Each stream is sent as chunk skeletons (`sse`, `ndjson`,
`json-array` or decoded `aws-eventstream` frames) with the chunk count and the
times of the first and last network reads; the server derives TTFT, tokens per
second and usage. SSE `event:` lines and the `[DONE]` sentinel are framing and
are not sent.

A stream your code stops reading early is reported when the response is
closed or released. Non-AI HTTP calls pass through untouched.

## Privacy

CostKey does not capture provider authorization headers. It reads only the
explicit `x-costkey-context`/legacy trace headers used for a trusted handoff,
then removes those headers before the provider request. Body and prompt
capture starts metadata-only until the project runtime config confirms text
capture; an explicit `capture_body=False` always wins. The DSN config is
refreshed periodically (customize with `config_refresh_interval`) so a remote
revocation returns future calls to metadata-only capture. In metadata-only
mode a call ships only its telemetry skeleton: text becomes its length, and
framework span attributes and OpenTelemetry attributes are skeletons too.
With text capture on, prompts, responses, span input and output, and framework
and OpenTelemetry attributes are sent whole, never truncated (secret-looking
values are still redacted). Deliveries are split so one request stays near
8 MB; a larger event goes alone.

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

Calls to these hosts are captured:

- **Model APIs:** OpenAI (including audio), Azure OpenAI, Anthropic, Google Gemini, Google Vertex AI, AWS Bedrock (HTTP and boto3), xAI, Mistral, DeepSeek, Cohere, AI21, Upstage, Sarvam, TypeSafe and Rumik.
- **OpenAI-compatible hosts:** Groq, Together, Fireworks, Perplexity, Cerebras, OpenRouter, DeepInfra, NVIDIA NIM, SambaNova, Novita, Hyperbolic, Nebius, Alibaba Cloud Model Studio (DashScope), Moonshot (Kimi), Z.ai (GLM), MiniMax, Hugging Face Inference Providers, Baseten, Lambda, Featherless, Friendli, Cloudflare Workers AI, Vercel AI Gateway, and a local Ollama server (port 11434). Gateways (Portkey, Helicone) are captured under their own label.
- **Speech:** ElevenLabs, Deepgram, AssemblyAI, Cartesia and OpenAI speech/transcription.
- **Embeddings, rerank and search:** Voyage AI, Jina AI, Cohere, Exa, Tavily and Firecrawl.

Which provider, model, tokens, native units (characters, audio seconds,
search queries, credits) and cost a call represents is decided on the server
from its raw telemetry. Custom hosts and internal gateways are configured in
the CostKey project; the SDK picks the hostnames up from the DSN runtime
config and captures them like any other provider.

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
| `costkey.record_usage(...)` | Report external usage exactly as passed (delta or cumulative) |
| `costkey.record_livekit_usage(update, ...)` | Send a raw LiveKit usage snapshot |
| `costkey.export_context()` | Export an explicit allowlisted carrier for trusted handoff |
| `costkey.export_context_headers()` | Encode that carrier for an explicitly trusted request |
| `costkey.get_runtime_config_status()` | Inspect metadata-safe runtime capture config state |

## Upgrading from 0.6

0.7.0 makes the SDK telemetry-only. Breaking changes:

- `costkey.score()` is removed. Quality checks run on the server.
- `costkey.register_pricing()` and `costkey.pricing` are removed. Pricing is server-side.
- `costkey.register_extractor()`, the provider extractor classes, `NormalizedUsage` and `StreamTiming` are removed. Custom providers are server configuration.
- Call events no longer carry SDK-computed `model`, `usage`, `costUsd`, `streamTiming`, `requestModel`, `providerRequestId`, `chargeId`, `systemPrompt` or `metadata`; they carry `telemetry` instead.
- Call events are now also sent when the provider call raises (`telemetry.error`, `response` null).
- `record_usage` no longer deduplicates, fences revisions, rejects negative or non-numeric units, or fills in `chargeId`, `currency`, `amountKind`, `unitKind` or `costUsd`.
- `record_livekit_usage` sends one raw event per snapshot instead of per-component usage events.
- Stack frames are sent unfiltered; the call graph upload sends raw functions, callees and call expressions instead of scores.
- Span and lifecycle errors send the raw exception class name (`errorName`, plus `errorType`); `CancelledError` is no longer mapped to `cancelled` in the SDK.
- Admission (`/admit`) receives the request skeleton as `request` instead of `model` and `usageBounds`; settlement always reports `outcome: "unknown"` with the call's `eventId`.
- Framework integration span attributes move under `attributes.<framework>`.

## Development

```bash
python -m pip install -e ".[test]" build twine
python -m pytest
# Framework integration tests skip frameworks that are not installed. CrewAI
# and DSPy pin openai<3, which conflicts with openai-agents, so run their
# tests from a second environment with those two installed.
python -m build
python -m twine check dist/*
```

## Release

Publishing is done from GitHub Releases only.

1. Update the versions in `pyproject.toml` and `costkey/version.py`.
2. Merge to `main`.
3. Create a GitHub Release tagged `vX.Y.Z`.
4. GitHub Actions runs tests, builds, verifies the tag matches the package version, and publishes to PyPI.

PyPI must have trusted publishing configured for `costkey/costkey-python` and the `pypi` GitHub environment.

## Links

- Docs: https://costkey.dev/docs
- Self-hosting: https://costkey.dev/docs/self-hosting/
- PyPI: https://pypi.org/project/costkey/
- TypeScript SDK: https://github.com/costkey/costkey

If CostKey helps you understand your AI bill, starring the repo helps more developers find it.

## License

MIT
