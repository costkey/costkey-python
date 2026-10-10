"""Agent spans, integrations and the 0.6 call-envelope fixes."""
from __future__ import annotations

import asyncio
import datetime
import json
import time

import httpx
import pytest

import costkey
import costkey.patch as patch_module
from costkey.integrations.browser_use import step_hooks
from costkey.integrations.openai_agents import CostKeyRunHooks
from costkey.providers import AnthropicExtractor, GoogleExtractor, OpenAIExtractor
from costkey.transport import Transport
from test_devx import capture_events


def _openai_handler(request: httpx.Request) -> httpx.Response:
    time.sleep(0.04)
    return httpx.Response(200, json={
        "id": "resp_1", "model": "gpt-6-sol",
        "usage": {"input_tokens": 12, "output_tokens": 4},
    })


@pytest.fixture()
def patched():
    with capture_events() as transport:
        patch_module._patch_httpx()
        patch_module._state.patched = True
        try:
            yield transport
        finally:
            patch_module.unpatch()


def _post(http: httpx.Client) -> None:
    http.post("https://acme.openai.azure.com/openai/v1/responses", json={"model": "gpt-6-sol", "input": "hi"})


def test_nested_spans_attribute_calls_to_the_innermost_span(patched):
    with httpx.Client(transport=httpx.MockTransport(_openai_handler)) as http:
        with costkey.agent("dash", user_id="u_1") as agent_span:
            agent_span.set_attribute("model", "gpt-6-sol")
            _post(http)
            with costkey.step("turn 1"):
                with costkey.tool("search_funds") as tool_span:
                    _post(http)

    spans = [e for e in patched.events if e.event_type == "span"]
    calls = [e for e in patched.events if e.event_type == "call"]
    assert len(spans) == 6, "every span is delivered at start and at end"
    ended = [e for e in spans if e.span_status != "running"]
    assert [(e.span_kind, e.span_name, e.span_status) for e in ended] == [
        ("tool", "search_funds", "ok"), ("step", "turn 1", "ok"), ("agent", "dash", "ok"),
    ]
    tool_end, step_end, agent_end = ended
    assert step_end.parent_span_id == agent_span.id
    assert tool_end.parent_span_id == step_end.id
    assert agent_end.parent_span_id is None
    assert agent_end.attributes["model"] == "gpt-6-sol"

    # Outside a run, the agent span started one.
    assert len({e.trace_id for e in spans} | {c.context["traceId"] for c in calls}) == 1
    assert calls[0].span_id == agent_span.id
    assert calls[0].context["agent"] == "dash"
    assert calls[0].context["userId"] == "u_1"
    assert calls[1].span_id == tool_span.id
    assert calls[1].context["tool"] == "search_funds"
    assert patch_module.get_context() == {}, "context is restored after the run"

    wire = Transport(endpoint="https://example.invalid", auth_key="k", max_batch_size=10, flush_interval=1, debug=False)._serialize(tool_end)
    assert wire["eventType"] == "span"
    assert wire["kind"] == "tool"
    assert wire["name"] == "search_funds"
    assert wire["parentSpanId"] == step_end.id
    assert Transport(endpoint="https://example.invalid", auth_key="k", max_batch_size=10, flush_interval=1, debug=False)._serialize(calls[1])["spanId"] == tool_span.id


def test_calls_are_stamped_at_start_and_attempt_labels_never_become_ids(patched):
    with httpx.Client(transport=httpx.MockTransport(_openai_handler)) as http:
        before = datetime.datetime.now(datetime.timezone.utc)
        with costkey.with_context(attempt_id="attempt-2", parent_event_id="attempt-1"):
            _post(http)
            _post(http)
    calls = [e for e in patched.events if e.event_type == "call"]
    assert len(calls) == 2
    assert calls[0].id != calls[1].id
    assert {c.attempt_id for c in calls} == {"attempt-2"}
    assert {c.parent_event_id for c in calls} == {"attempt-1"}
    started = datetime.datetime.fromisoformat(calls[0].timestamp.replace("Z", "+00:00"))
    assert (started - before).total_seconds() < 0.03
    assert calls[0].duration_ms >= 35


def test_decorated_async_spans_record_errors_and_outputs(patched):
    @costkey.tool("lookup")
    async def lookup(code: str) -> dict:
        return {"fund": code}

    @costkey.agent("planner")
    async def plan() -> None:
        await lookup("INF1")
        raise RuntimeError("gave up")

    with pytest.raises(RuntimeError):
        asyncio.run(plan())
    ended = {e.span_name: e for e in patched.events if e.event_type == "span" and e.span_status != "running"}
    assert ended["lookup"].span_status == "ok"
    assert ended["lookup"].attributes["output"] == json.dumps({"fund": "INF1"})
    assert ended["planner"].span_status == "error"
    assert ended["planner"].error_type == "RuntimeError"


def test_span_content_is_dropped_when_capture_is_off(patched):
    patch_module._state.capture_body = False
    with costkey.tool("lookup", input={"q": "secret"}) as span:
        span.set_output("secret result")
    for event in (e for e in patched.events if e.event_type == "span"):
        assert "input" not in (event.attributes or {})
        assert "output" not in (event.attributes or {})


def test_openai_agents_hooks_record_agents_tools_and_handoffs(patched):
    class Agent:
        def __init__(self, name):
            self.name = name
            self.model = "gpt-6-sol"

    class Tool:
        name = "get_holdings"

    async def run() -> None:
        hooks = CostKeyRunHooks()
        ctx = object()
        dash, browser, tool = Agent("dash"), Agent("browser"), Tool()
        await hooks.on_agent_start(ctx, dash)
        await hooks.on_tool_start(ctx, dash, tool)
        assert patch_module.get_context()["tool"] == "get_holdings"
        await hooks.on_tool_end(ctx, dash, tool, "3 funds")
        await hooks.on_handoff(ctx, dash, browser)
        await hooks.on_agent_end(ctx, dash, "done")

    asyncio.run(run())
    ended = [(e.span_kind, e.span_name) for e in patched.events if e.event_type == "span" and e.span_status != "running"]
    assert ended == [("tool", "get_holdings"), ("handoff", "dash → browser"), ("agent", "dash")]
    agent_end = [e for e in patched.events if e.span_kind == "agent" and e.span_status == "ok"][0]
    assert agent_end.attributes["model"] == "gpt-6-sol"


def test_browser_use_step_hooks_number_each_step(patched):
    class State:
        n_steps = 0

    class BrowserAgent:
        state = State()

    async def run() -> None:
        on_start, on_end = step_hooks()
        agent = BrowserAgent()
        with costkey.agent("browser-agent"):
            for number in (1, 2):
                agent.state.n_steps = number
                await on_start(agent)
                await on_end(agent)

    asyncio.run(run())
    steps = [e for e in patched.events if e.span_kind == "step" and e.span_status == "ok"]
    assert [e.attributes["step"] for e in steps] == [1, 2]
    agent = [e for e in patched.events if e.span_kind == "agent" and e.span_status == "running"][0]
    assert all(step.parent_span_id == agent.id for step in steps)


def test_streaming_usage_is_merged_across_provider_event_shapes():
    def sse(*events):
        return "".join(f"data: {json.dumps(event)}\n\n" for event in events)

    responses = sse(
        {"type": "response.created", "response": {"model": "gpt-6-sol"}},
        {"type": "response.completed", "response": {"usage": {
            "input_tokens": 120, "output_tokens": 30,
            "input_tokens_details": {"cached_tokens": 100},
            "output_tokens_details": {"reasoning_tokens": 12},
        }}},
    )
    usage = patch_module._extract_sse_usage(responses, OpenAIExtractor())
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (120, 30, 150)

    anthropic = sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 900, "output_tokens": 1}}},
        {"type": "message_delta", "usage": {"output_tokens": 75}},
    )
    usage = patch_module._extract_sse_usage(anthropic, AnthropicExtractor())
    assert (usage.input_tokens, usage.output_tokens) == (900, 75)

    gemini = sse(
        {"usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 5}},
        {"usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 40, "thoughtsTokenCount": 20}},
    )
    usage = patch_module._extract_sse_usage(gemini, GoogleExtractor())
    assert usage.input_tokens == 50 and usage.output_tokens >= 40


def test_streaming_is_detected_from_the_url_and_bounds_cover_input_and_output():
    assert patch_module._is_streaming_request({}, "https://aiplatform.googleapis.com/v1/models/g:streamGenerateContent?alt=sse")
    assert patch_module._is_streaming_request({}, "https://bedrock-runtime.us-east-1.amazonaws.com/model/x/converse-stream")
    assert not patch_module._is_streaming_request({}, "https://api.openai.com/v1/responses")
    body = {"model": "gpt-6-sol", "input": "hello", "max_output_tokens": 512}
    raw = json.dumps(body).encode()
    assert patch_module._estimate_request_usage(body, raw) == {"maxInputTokens": len(raw), "maxOutputTokens": 512}
