"""Non-call payloads stay raw: call graph, explicit usage, LiveKit, errors."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass

import pytest

import costkey
from costkey.callgraph import build_call_graph, build_payload, discover_files
from costkey.transport import Transport

from conftest import capturing


def _wire(event):
    return Transport("https://x.test/api/v1/events", "k", 50, 5, False)._serialize(event)


# ── Call graph (contract detail 6) ──

def test_call_graph_is_raw_functions_callees_and_expressions(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "search.py").write_text(
        "import openai\n"
        "client = openai.OpenAI()\n"
        "def summarize(text):\n"
        "    return client.chat.completions.create(model='m', messages=[])\n"
        "class Service:\n"
        "    def run(self, q):\n"
        "        helper(q)\n"
        "        return summarize(q)\n"
        "def helper(q):\n"
        "    return openai.OpenAI().responses.create(input=q)\n"
    )
    (tmp_path / "app" / "api.py").write_text(
        "from app.search import Service\n"
        "def handler(request):\n"
        "    return Service().run(request['q'])\n"
    )
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.py").write_text("def nope(): pass\n")
    graph = build_call_graph(discover_files(str(tmp_path)), str(tmp_path))
    assert graph == {
        "app/api.py:handler": {"line": 2, "callees": ["app/search.py:Service.run"], "callExpressions": ["Service().run", "Service"]},
        "app/search.py:summarize": {"line": 3, "callees": [], "callExpressions": ["client.chat.completions.create"]},
        "app/search.py:Service.run": {"line": 6, "callees": ["app/search.py:helper", "app/search.py:summarize"],
                                      "callExpressions": ["helper", "summarize"]},
        "app/search.py:helper": {"line": 9, "callees": [], "callExpressions": ["openai.OpenAI().responses.create", "openai.OpenAI"]},
    }
    payload = build_payload(graph)
    assert set(payload) == {"sdkVersion", "functions"}
    flat = json.dumps(payload)
    for removed in ("score", "fan_in", "calls_ai", "ai_distance"):
        assert removed not in flat


def test_call_graph_payload_is_size_capped():
    graph = {f"f{i:05}.py:fn": {"line": 1, "callees": [], "callExpressions": ["x" * 180] * 64} for i in range(200)}
    payload = build_payload(graph)
    assert payload["truncated"] is True
    assert len(json.dumps(payload).encode()) <= 1024 * 1024
    assert "f00000.py:fn" in payload["functions"]


# ── record_usage: what the caller passes, nothing derived ──

def test_record_usage_sends_exactly_what_was_passed():
    with capturing() as sink:
        costkey.record_usage(
            event_id="u1", component_id="search", revision=0,
            usage={"queries": 2, "results": float("nan")}, amount=0.004,
        )
    wire = _wire(sink.events[0])
    assert wire["eventType"] == "usage"
    assert wire["usage"] == {"queries": 2, "results": None}
    assert wire["amount"] == 0.004
    assert wire["aggregation"] == "delta"
    assert wire["revision"] == 0
    # No derived defaults: charge id, currency, amountKind and costUsd are the server's.
    for key in ("chargeId", "currency", "amountKind", "unitKind"):
        assert key not in wire
    assert wire["costUsd"] is None
    assert wire["provider"] == "unknown"


def test_record_usage_keeps_repeats_and_conflicts_for_the_server():
    with capturing() as sink:
        kwargs = dict(component_id="c", revision=1, aggregation="cumulative", usage={"seconds": 1})
        costkey.record_usage(event_id="a", **kwargs)
        costkey.record_usage(event_id="b", **{**kwargs, "usage": {"seconds": 2}})
        costkey.record_usage(event_id="c", **{**kwargs, "revision": 0})
        costkey.record_usage(event_id="d", component_id="c", revision=2, usage={"seconds": -3})
    assert [event.id for event in sink.events] == ["a", "b", "c", "d"]
    assert sink.events[3].native_usage == {"seconds": -3}


def test_record_usage_still_requires_the_fields_the_server_rejects_without():
    with pytest.raises(ValueError):
        costkey.record_usage(component_id="c", revision=0)
    with pytest.raises(ValueError):
        costkey.record_usage(event_id="e", revision=0)
    with pytest.raises(ValueError):
        costkey.record_usage(event_id="e", component_id="c")
    with pytest.raises(ValueError):
        costkey.record_usage(event_id="e", component_id="c", revision=0, aggregation="sum")


def test_record_usage_passes_through_amount_objects_and_currency():
    with capturing() as sink:
        costkey.record_usage(event_id="e", component_id="c", revision=0, provider="deepgram",
                             amount={"value": 12, "currency": "inr"}, currency="inr", unit="seconds")
    wire = _wire(sink.events[0])
    assert wire["amount"] == {"value": 12, "currency": "inr"}
    assert wire["currency"] == "inr"
    assert wire["unit"] == "seconds"
    assert wire["provider"] == "deepgram"


# ── LiveKit: raw rows ──

@dataclass
class LLMModelUsage:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    session_duration: float


@dataclass
class TTSModelUsage:
    provider: str
    model: str
    characters_count: int
    audio_duration: float


def test_livekit_sends_the_raw_update_as_a_skeleton():
    update = {
        "type": "session_usage_updated",
        "created_at": 1760000000.5,
        "usage": {"model_usage": [
            LLMModelUsage("openai", "gpt-4o-mini", 120, 30, 12.5),
            TTSModelUsage("elevenlabs", "eleven_turbo_v2", 90, 4.2),
        ]},
    }
    with capturing() as sink:
        ids = costkey.record_livekit_usage(update, session_id="room-1", revision=3,
                                           exclude_components=["llm"], conversation_id="conv")
    assert ids == ["livekit:room-1:r3"]
    event = sink.events[0]
    assert event.event_type == "usage"
    assert event.source == "livekit"
    assert event.aggregation == "cumulative" and event.revision == 3
    assert event.context["voiceSessionId"] == "room-1"
    assert event.telemetry == {
        "v": 1,
        "source": "livekit",
        "livekit": {
            "sessionId": "room-1",
            "revision": 3,
            "update": {
                "type": "session_usage_updated",
                "created_at": 1760000000.5,
                "usage": {"model_usage": [
                    {"provider": "openai", "model": "gpt-4o-mini", "input_tokens": 120, "output_tokens": 30,
                     "session_duration": 12.5},
                    {"provider": "elevenlabs", "model": "eleven_turbo_v2", "characters_count": 90,
                     "audio_duration": 4.2},
                ]},
            },
            "excludeComponents": ["llm"],
        },
    }
    wire = _wire(event)
    assert "billingUnits" not in wire and "unit" not in wire and "amount" not in wire


def test_livekit_requires_session_and_revision():
    with pytest.raises(ValueError):
        costkey.record_livekit_usage({}, revision=1)
    with pytest.raises(ValueError):
        costkey.record_livekit_usage({}, session_id="s")


# ── Errors are raw class names ──

def test_trace_and_span_errors_send_the_raw_name():
    with capturing() as sink:
        with pytest.raises(KeyError):
            with costkey.start_trace("t"):
                with costkey.span("work"):
                    raise KeyError("x")
    lifecycle = [e for e in sink.events if e.event_type == "lifecycle"]
    assert lifecycle[-1].lifecycle == "failed"
    assert lifecycle[-1].error_type == "KeyError"
    span_end = sink.ended_spans()[0]
    assert span_end.span_status == "error"
    wire = _wire(span_end)
    assert wire["errorName"] == "KeyError" and wire["errorType"] == "KeyError"


def test_cancellation_is_not_classified_in_the_sdk():
    async def cancelled():
        with costkey.start_trace("t"):
            async with costkey.agent("a"):
                raise asyncio.CancelledError()

    with capturing() as sink:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(cancelled())
    lifecycle = [e for e in sink.events if e.event_type == "lifecycle"]
    assert lifecycle[-1].lifecycle == "failed"
    assert lifecycle[-1].error_type == "CancelledError"
    assert sink.ended_spans()[0].error_type == "CancelledError"


# ── Removed APIs and import hygiene ──

def test_removed_apis_are_gone():
    for name in ("score", "register_pricing", "register_extractor"):
        assert not hasattr(costkey, name)
    with pytest.raises(ImportError):
        __import__("costkey.pricing")
    assert costkey.__version__ == "0.7.0"


def test_importing_costkey_imports_no_framework():
    frameworks = [
        "agents", "langchain_core", "langgraph", "crewai", "llama_index", "pydantic_ai", "google.adk",
        "claude_agent_sdk", "smolagents", "autogen_agentchat", "autogen_core", "ag2", "dspy", "strands",
        "opentelemetry",
    ]
    code = (
        "import sys, costkey, costkey.integrations; "
        f"print([m for m in {frameworks!r} if m in sys.modules])"
    )
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": os.getcwd()}, check=True).stdout.strip()
    assert output == "[]"
