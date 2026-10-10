"""Contract checks for context, lifecycle, explicit usage, and admission."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

import costkey.client as client
import costkey.patch as patch_module
from costkey.livekit import record_livekit_usage
from costkey.policy import CostKeyPolicyRefused, PolicyOptions
from costkey.providers import OpenAIExtractor, apply_runtime_providers, find_extractor
from costkey.transport import Transport


class CaptureTransport:
    def __init__(self) -> None:
        self.events = []

    def enqueue(self, event):
        self.events.append(event)


@contextmanager
def capture_events():
    state = patch_module._state
    previous = {
        "transport": state.transport,
        "project_id": state.project_id,
        "before_send": state.before_send,
        "before_send_event": state.before_send_event,
        "debug": state.debug,
        "admission": state.admission,
        "admission_mode": state.admission_mode,
        "capture_body": state.capture_body,
        "local_capture_body": state.local_capture_body,
        "config_known": state.config_known,
        "config_status": state.config_status,
        "redaction_enabled": state.redaction_enabled,
    }
    transport = CaptureTransport()
    state.transport = transport
    state.project_id = "project-test"
    state.before_send = None
    state.before_send_event = None
    state.debug = False
    state.admission = None
    state.admission_mode = "monitor"
    state.capture_body = True
    state.local_capture_body = True
    state.config_known = True
    state.config_status = "available"
    state.redaction_enabled = True
    try:
        yield transport
    finally:
        for key, value in previous.items():
            setattr(state, key, value)
        with client._usage_lock:
            client._usage_updates.clear()
            client._usage_modes.clear()
            client._usage_event_ids.clear()


def test_trace_lifecycle_is_non_billable_and_context_is_restored():
    with capture_events() as transport:
        with client.start_trace(name="search"):
            current = patch_module.get_context()
            assert current["traceName"] == "search"
            assert current["traceId"]

        assert patch_module.get_context() == {}
        lifecycle = [event for event in transport.events if event.event_type == "lifecycle"]
        assert [event.lifecycle for event in lifecycle] == ["started", "completed"]
        assert [event.sequence for event in lifecycle] == [0, 1]
        assert all(event.entity_type == "trace" for event in lifecycle)

        wire = Transport("https://example.test", "key", 50, 5, False)._serialize(lifecycle[0])
        assert wire["eventType"] == "lifecycle"
        assert wire["entityType"] == "trace"
        assert wire["entityId"] == wire["traceId"]


def test_lifecycle_inherits_default_context():
    state = patch_module._state
    previous = state.default_context
    state.default_context = {"environment": "test"}
    try:
        with capture_events() as transport:
            with client.start_trace(name="default-context"):
                pass
            started = transport.events[0]
            assert started.context["environment"] == "test"
    finally:
        state.default_context = previous


def test_explicit_lifecycle_sequence_advances_future_automatic_sequence():
    with capture_events() as transport:
        patch_module.emit_lifecycle(
            entity_type="trace",
            entity_id="manual-sequence",
            lifecycle="started",
            context={},
            sequence=10,
        )
        patch_module.emit_lifecycle(
            entity_type="trace",
            entity_id="manual-sequence",
            lifecycle="completed",
            context={},
        )
    assert [event.sequence for event in transport.events] == [10, 11]


def test_lifecycle_sequence_scope_includes_project_identity():
    state = patch_module._state
    with capture_events() as first:
        patch_module.emit_lifecycle(
            entity_type="execution",
            entity_id="shared-id",
            lifecycle="started",
            context={},
        )
        state.project_id = "another-project"
        patch_module.emit_lifecycle(
            entity_type="execution",
            entity_id="shared-id",
            lifecycle="started",
            context={},
        )
    assert [event.sequence for event in first.events] == [0, 0]


def test_runtime_config_fetch_is_real_http_and_enables_only_remote_text():
    seen = []
    ready = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization")))
            payload = {
                "projectId": "project-test",
                "capture": {
                    "mode": "text",
                    "bodyCaptureEnabled": True,
                    "redactionEnabled": True,
                },
                "providers": [],
            }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            ready.set()

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state = patch_module._state
    previous = (state.project_id, state.local_capture_body, state.capture_body, state.config_status)
    state.project_id = "project-test"
    state.local_capture_body = True
    state.capture_body = False
    state.config_status = "unknown"
    try:
        port = server.server_address[1]
        patch_module.start_runtime_config_fetch("http://127.0.0.1:%s" % port, "config-key", "project-test")
        assert ready.wait(2)
        deadline = threading.Event()
        # The response handler sets ready before the worker applies its state;
        # a short bounded polling loop avoids sleeping for an unbounded time.
        for _ in range(100):
            if state.config_status == "available":
                break
            deadline.wait(0.01)
        assert state.config_status == "available"
        assert state.capture_body is True
    finally:
        state.project_id, state.local_capture_body, state.capture_body, state.config_status = previous
        server.shutdown()
        server.server_close()
    assert seen == [("/api/v1/projects/project-test/config", "Bearer config-key")]


def test_runtime_config_revocation_and_remote_metadata_only_keep_bodies_off():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"error":"Invalid auth key"}'
            self.send_response(401)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state = patch_module._state
    previous = (state.project_id, state.local_capture_body, state.capture_body, state.config_status)
    state.project_id = "project-test"
    state.local_capture_body = True
    state.capture_body = False
    state.config_status = "unknown"
    try:
        patch_module.start_runtime_config_fetch(
            "http://127.0.0.1:%s" % server.server_address[1], "revoked-key", "project-test"
        )
        for _ in range(100):
            if state.config_status == "revoked":
                break
            threading.Event().wait(0.01)
        assert state.config_status == "revoked"
        assert state.capture_body is False
    finally:
        state.project_id, state.local_capture_body, state.capture_body, state.config_status = previous
        server.shutdown()
        server.server_close()


def test_remote_metadata_mode_overrides_local_true():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = json.dumps({
                "projectId": "project-test",
                "capture": {"mode": "metadata", "bodyCaptureEnabled": False, "redactionEnabled": True},
                "providers": [],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state = patch_module._state
    previous = (state.project_id, state.local_capture_body, state.capture_body, state.config_status)
    state.project_id = "project-test"
    state.local_capture_body = True
    state.capture_body = False
    state.config_status = "unknown"
    try:
        patch_module.start_runtime_config_fetch(
            "http://127.0.0.1:%s" % server.server_address[1], "key", "project-test"
        )
        for _ in range(100):
            if state.config_status == "available":
                break
            threading.Event().wait(0.01)
        assert state.config_status == "available"
        assert state.capture_body is False
    finally:
        state.project_id, state.local_capture_body, state.capture_body, state.config_status = previous
        server.shutdown()
        server.server_close()


def test_explicit_local_capture_false_remains_strongest_after_config():
    state = patch_module._state
    previous = (state.project_id, state.local_capture_body, state.capture_body, state.config_status)
    state.project_id = "project-test"
    state.local_capture_body = False
    state.capture_body = False
    state.config_status = "unknown"
    try:
        assert patch_module.apply_runtime_config({
            "projectId": "project-test",
            "capture": {"mode": "text", "bodyCaptureEnabled": True, "redactionEnabled": True},
            "providers": [],
        }, project_id="project-test")
        assert patch_module.runtime_config_status() == "available"
        assert state.capture_body is False
    finally:
        state.project_id, state.local_capture_body, state.capture_body, state.config_status = previous


def test_init_uses_real_config_and_ingest_http_without_emitting_provider_secrets():
    received: list[tuple[str, dict | None, str | None]] = []
    ready = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = json.dumps({
                "projectId": "privacy-project",
                "capture": {
                    "mode": "metadata",
                    "bodyCaptureEnabled": False,
                    "redactionEnabled": True,
                },
                "providers": [],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            ready.set()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            received.append((self.path, body, self.headers.get("Authorization")))
            if self.path == "/provider-privacy":
                response = {
                    "id": "provider-privacy-1",
                    "model": "fixture-model",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
                status = 200
            else:
                response = {"accepted": len(body.get("events", []))}
                status = 202
            payload = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            return

    class PrivacyExtractor:
        provider = "fixture"

        def match(self, url):
            return urlparse(url).path == "/provider-privacy"

        def extract_usage(self, body):
            usage = body.get("usage") if isinstance(body, dict) else None
            if not isinstance(usage, dict):
                return None
            from costkey.types import NormalizedUsage
            return NormalizedUsage(
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                total_tokens=(usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0),
            )

        def extract_model(self, request_body, response_body):
            for body in (response_body, request_body):
                if isinstance(body, dict) and isinstance(body.get("model"), str):
                    return body["model"]
            return None

    from urllib.parse import urlparse
    from costkey.providers import register_extractor
    register_extractor(PrivacyExtractor())
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        import costkey
        costkey.init(
            dsn=f"http://dsn-key@127.0.0.1:{server.server_address[1]}/privacy-project",
            capture_body=True,
            max_batch_size=10,
            flush_interval=60,
            scan_callgraph=False,
        )
        assert ready.wait(2)
        with costkey.with_context(traceId="privacy-trace"):
            with httpx.Client() as http:
                http.post(
                    f"http://user:provider-password@127.0.0.1:{server.server_address[1]}/provider-privacy?key=provider-secret",
                    headers={"x-costkey-context": json.dumps({"traceId": "carrier-trace"})},
                    json={
                        "model": "fixture-model",
                        "messages": [{"role": "user", "content": "body-secret"}],
                    },
                )
        costkey.flush()
    finally:
        costkey.shutdown()
        server.shutdown()
        server.server_close()

    ingest = [body for path, body, _auth in received if path == "/api/v1/events"]
    assert ingest
    wire = ingest[-1]["events"][0]
    assert wire["requestBody"] is None
    assert wire["responseBody"] is None
    assert wire["metadata"] == {"raw": {"promptCount": 1, "promptRoles": ["user"]}}
    assert wire["url"] == f"http://127.0.0.1:{server.server_address[1]}/provider-privacy"
    assert "provider-secret" not in json.dumps(ingest)
    assert "provider-password" not in json.dumps(ingest)
    assert wire["context"]["traceId"] == "privacy-trace"


def test_metadata_safe_capture_gate_removes_body_and_prompt_fields_on_wire():
    state = patch_module._state
    with capture_events() as transport:
        state.capture_body = False
        request = {"model": "gpt-4o", "messages": [{"role": "user", "content": "private"}]}
        response = {"id": "provider-1", "model": "gpt-4o", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        patch_module._process(
            OpenAIExtractor(), "https://api.openai.com/v1/chat/completions?key=secret", "POST", 200,
            request, response, 1, False, None, None, {"traceId": "trace"}, event_id="call-1",
        )
        wire = Transport("https://example.test", "key", 50, 5, False)._serialize(transport.events[0])
    assert wire["requestBody"] is None
    assert wire["responseBody"] is None
    assert wire["metadata"] == {
        "raw": {"promptCount": 1, "promptRoles": ["user"]},
    }
    assert "private" not in json.dumps(wire)
    assert wire["systemPrompt"] is None
    assert "secret" not in wire["url"]


def test_carrier_allowlist_accepts_only_finite_scalars_and_headers_case_insensitively():
    with client.with_context(
        traceId="trace",
        feature={"prompt": "must-not-cross"},
        customerId=float("inf"),
    ):
        carrier = client.export_context()
        assert carrier == {"traceId": "trace"}

    assert client.import_context({"X-CostKey-Context": json.dumps({
        "traceId": "from-header",
        "feature": {"private": "value"},
    })}) == {"traceId": "from-header"}
    with client.with_context({"X-CostKey-Context": json.dumps({"traceId": "nested-header"})}):
        assert patch_module.get_context() == {"traceId": "nested-header"}


def test_nested_trace_is_fresh_but_imported_trace_joins():
    with capture_events():
        with client.start_trace(name="outer"):
            outer = patch_module.get_context()["traceId"]
            with client.start_trace(name="inner"):
                assert patch_module.get_context()["traceId"] != outer
        with client.with_context(traceId="upstream"):
            with client.start_trace(name="joined"):
                assert patch_module.get_context()["traceId"] == "upstream"


def test_trace_failure_emits_type_without_exception_message():
    with capture_events() as transport:
        with pytest.raises(RuntimeError, match="private details"):
            with client.start_trace(name="fails"):
                raise RuntimeError("private details")
    failed = [event for event in transport.events if event.lifecycle == "failed"][0]
    assert failed.error_type == "RuntimeError"
    assert not hasattr(failed, "error_message")


def test_execution_reuses_group_but_freshes_invocation():
    with capture_events() as transport:
        with client.start_execution("nightly", kind="batch"):
            first = patch_module.get_context()["executionId"]
        with client.start_execution("nightly", kind="batch"):
            second = patch_module.get_context()["executionId"]

        assert first != second
        starts = [
            event for event in transport.events
            if event.event_type == "lifecycle" and event.lifecycle == "started"
        ]
        assert [event.group_name for event in starts] == ["nightly", "nightly"]
        assert starts[0].entity_id != starts[1].entity_id


def test_execution_lifecycle_carries_item_counts_on_start_and_terminal():
    with capture_events() as transport:
        with client.start_execution(
            "nightly-counted",
            kind="batch",
            expected_items=10,
            completed_items=8,
            failed_items=2,
        ):
            pass
        lifecycle = [event for event in transport.events if event.entity_type == "execution"]
        assert [(event.expected_items, event.completed_items, event.failed_items) for event in lifecycle] == [
            (10, 8, 2),
            (10, 8, 2),
        ]
        wire = Transport("https://example.test", "key", 50, 5, False)._serialize(lifecycle[-1])
    assert wire["expectedItems"] == 10
    assert wire["completedItems"] == 8
    assert wire["failedItems"] == 2


def test_execution_lifecycle_rejects_negative_item_counts():
    with capture_events():
        with pytest.raises(ValueError, match="completed_items"):
            with client.start_execution("invalid-count", completed_items=-1):
                pass


def test_contextvar_isolation_across_concurrent_tasks():
    async def worker(label: str) -> str:
        with client.with_context(feature=label):
            await asyncio.sleep(0)
            return patch_module.get_context()["feature"]

    async def run():
        return await asyncio.gather(worker("a"), worker("b"))

    assert asyncio.run(run()) == ["a", "b"]
    assert patch_module.get_context() == {}


def test_async_provider_calls_capture_their_task_context():
    async def handler(request):
        await asyncio.sleep(0)
        return httpx.Response(200, json={
            "id": request.headers.get("x-request-id", "call"),
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        })

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            async def worker(label):
                with client.with_context(feature=label):
                    await http.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"x-request-id": label},
                        json={"model": "gpt-4o", "messages": []},
                    )
            await asyncio.gather(worker("a"), worker("b"))

    with capture_events() as transport:
        patch_module._patch_httpx()
        patch_module._state.patched = True
        try:
            asyncio.run(run())
        finally:
            patch_module.unpatch()
    assert {event.context["feature"] for event in transport.events} == {"a", "b"}


def test_slow_transport_flush_does_not_delay_provider_return(monkeypatch):
    import costkey.transport as transport_module

    slow_started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow_post(*_args, **_kwargs):
        slow_started.set()
        release.wait(2)
        finished.set()
        return httpx.Response(202)

    monkeypatch.setattr(transport_module.httpx, "post", slow_post)
    state = patch_module._state
    previous = {
        "transport": state.transport,
        "project_id": state.project_id,
        "capture_body": state.capture_body,
        "local_capture_body": state.local_capture_body,
        "config_known": state.config_known,
        "config_status": state.config_status,
        "before_send": state.before_send,
        "before_send_event": state.before_send_event,
        "admission": state.admission,
        "admission_mode": state.admission_mode,
    }
    telemetry = Transport("https://ingest.test/events", "key", 1, 60, False)
    state.transport = telemetry
    state.project_id = "slow-transport-project"
    state.capture_body = False
    state.local_capture_body = False
    state.config_known = True
    state.config_status = "available"
    state.before_send = None
    state.before_send_event = None
    state.admission = None
    state.admission_mode = "monitor"
    try:
        patch_module._patch_httpx()
        state.patched = True
        telemetry.start()
        started = time.perf_counter()
        with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
            200,
            json={
                "id": "slow-provider-request",
                "model": "fixture-model",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        ))) as http:
            response = http.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "fixture-model", "messages": []},
            )
        elapsed = time.perf_counter() - started
        assert response.status_code == 200
        assert elapsed < 0.25
        assert slow_started.wait(1)
    finally:
        release.set()
        assert finished.wait(2)
        telemetry.stop()
        if state.patched:
            patch_module.unpatch()
        for key, value in previous.items():
            setattr(state, key, value)


def test_explicit_carrier_is_imported_then_removed_before_provider_transport():
    seen_headers = []

    def handler(request):
        seen_headers.append(dict(request.headers))
        return httpx.Response(200, json={"model": "gpt-4o", "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    with capture_events() as transport:
        patch_module._patch_httpx()
        patch_module._state.patched = True
        try:
            with httpx.Client(transport=httpx.MockTransport(handler)) as http:
                http.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"x-costkey-context": json.dumps({"traceId": "imported", "privateLabel": "local-only"})},
                    json={"model": "gpt-4o", "messages": []},
                )
        finally:
            patch_module.unpatch()
    assert "x-costkey-context" not in seen_headers[0]
    assert transport.events[0].context["traceId"] == "imported"
    assert "privateLabel" not in transport.events[0].context


def test_record_usage_deduplicates_and_preserves_cumulative_snapshots():
    with capture_events() as transport:
        client.record_usage(
            event_id="voice-1-r1",
            component_id="voice:stt",
            charge_id="voice:stt",
            provider="deepgram",
            usage={"audioSeconds": 1.0},
            aggregation="cumulative",
            revision=1,
        )
        client.record_usage(
            event_id="voice-1-r2",
            component_id="voice:stt",
            charge_id="voice:stt",
            provider="deepgram",
            usage={"audioSeconds": 2.0},
            aggregation="cumulative",
            revision=2,
        )
        # Same charge/revision and payload is idempotent, even with a new ID.
        client.record_usage(
            event_id="voice-1-r2-retry",
            component_id="voice:stt",
            charge_id="voice:stt",
            provider="deepgram",
            usage={"audioSeconds": 2.0},
            aggregation="cumulative",
            revision=2,
        )
        # Older snapshots are ignored, not converted to a negative delta.
        client.record_usage(
            event_id="voice-1-old",
            component_id="voice:stt",
            charge_id="voice:stt",
            provider="deepgram",
            usage={"audioSeconds": 1.5},
            aggregation="cumulative",
            revision=0,
        )

        assert [event.id for event in transport.events] == ["voice-1-r1", "voice-1-r2"]
        serialized = Transport("https://example.test", "key", 50, 5, False)._serialize(transport.events[-1])
        assert serialized["eventType"] == "usage"
        assert serialized["componentId"] == "voice:stt"
        assert serialized["chargeId"] == "voice:stt"
        assert serialized["aggregation"] == "cumulative"
        assert serialized["revision"] == 2
        assert serialized["usage"] == {"audioSeconds": 2.0}
        assert "usageEvent" not in serialized


def test_usage_conflicting_revision_and_modes_are_rejected():
    with capture_events():
        kwargs = dict(
            component_id="component",
            charge_id="charge",
            provider="provider",
            usage={"seconds": 1},
            aggregation="delta",
            revision=1,
        )
        client.record_usage(event_id="one", **kwargs)
        with pytest.raises(ValueError, match="conflicting usage update"):
            client.record_usage(event_id="two", **{**kwargs, "usage": {"seconds": 2}})
        with pytest.raises(ValueError, match="cannot switch aggregation"):
            client.record_usage(event_id="three", **{**kwargs, "aggregation": "cumulative", "revision": 2})


def test_record_usage_normalizes_provider_and_cost_aliases():
    with capture_events() as transport:
        client.record_usage({
            "id": "cost-alias",
            "componentId": "meter",
            "chargeId": "meter-charge",
            "provider": None,
            "costUsd": 0.03,
            "currency": " usd ",
            "usage": {"requests": 1},
            "aggregation": "delta",
            "revision": 1,
        })
    assert transport.events[0].provider == "unknown"
    assert transport.events[0].amount == {"value": 0.03, "currency": "USD"}


def test_record_usage_keeps_currency_when_amount_is_unknown():
    with capture_events() as transport:
        client.record_usage(
            event_id="unknown-inr",
            component_id="meter",
            provider="meter",
            usage={"requests": None},
            currency="inr",
            revision=1,
            aggregation="delta",
        )
    wire = Transport("https://example.test", "key", 50, 5, False)._serialize(transport.events[0])
    assert wire.get("amount") is None
    assert wire["currency"] == "INR"


def test_record_usage_accepts_contract_shaped_mapping_and_id_alias():
    with capture_events() as transport:
        client.record_usage({
            "componentId": "tool:search",
            "chargeId": "charge:search",
            "provider": "custom",
            "usage": {"requests": 1},
            "amount": {"value": 0.25, "currency": "INR"},
            "aggregation": "delta",
            "revision": 4,
            "unitKind": "requests",
            "unitValue": 1,
            "metadata": {"source": "fixture", "api_key": "hidden"},
        }, id="usage-4")
    event = transport.events[0]
    assert event.id == "usage-4"
    assert event.amount == {"value": 0.25, "currency": "INR"}
    assert event.cost_usd is None
    assert event.unit_kind == "requests"
    assert event.unit_value == 1
    assert event.metadata == {"source": "fixture", "api_key": "[REDACTED]"}


def test_record_usage_rejects_non_numeric_or_negative_units():
    with capture_events():
        with pytest.raises(TypeError, match="usage.seconds"):
            client.record_usage(
                event_id="bad",
                component_id="component",
                provider="custom",
                usage={"seconds": -1},
                revision=1,
                aggregation="delta",
            )
        with pytest.raises(TypeError, match="usage.seconds"):
            client.record_usage(
                event_id="nan",
                component_id="component",
                provider="custom",
                usage={"seconds": float("nan")},
                revision=2,
                aggregation="delta",
            )


def test_export_context_is_allowlisted():
    with client.with_context(traceId="trace", executionId="exec", api_key="secret", prompt="private"):
        carrier = client.export_context()
    assert carrier == {"traceId": "trace", "executionId": "exec"}
    with pytest.raises(ValueError, match="not exportable"):
        client.export_context(keys=["api_key"])


def test_context_header_roundtrip_remains_explicit_and_allowlisted():
    with client.with_context(traceId="trace", internal="local"):
        headers = client.export_context_headers()
    assert set(headers) == {"x-costkey-context"}
    assert client.import_context(headers["x-costkey-context"]) == {"traceId": "trace"}


def test_livekit_model_usage_fixture_reports_components_once():
    class Usage:
        def __init__(self, provider, model, **values):
            self.provider = provider
            self.model = model
            self.__dict__.update(values)

    class SessionUsage:
        def __init__(self):
            self.model_usage = [
                Usage("deepgram", "nova-3", audio_duration=1.25, input_audio_tokens=20, output_text_tokens=4),
                Usage("openai", "gpt-4o", input_tokens=10, output_tokens=5),
                Usage("elevenlabs", "turbo", characters_count=30, audio_duration=2.0),
            ]

    class Update:
        session_id = "room-1"
        usage = SessionUsage()

    with capture_events() as transport:
        ids = record_livekit_usage(Update(), revision=7, expected_components=("stt", "llm", "tts"))
        assert len(ids) == 3
        assert {event.component_id for event in transport.events} == {
            "livekit:room-1:stt:deepgram:nova-3",
            "livekit:room-1:llm:openai:gpt-4o",
            "livekit:room-1:tts:elevenlabs:turbo",
        }
        assert all(event.aggregation == "cumulative" and event.revision == 7 for event in transport.events)
        assert all(event.metadata["livekit"]["revision"] == 7 for event in transport.events)
        assert {
            json.dumps(event.context["voiceCoverage"], sort_keys=True)
            for event in transport.events
        } == {
            json.dumps({
                "expectedComponents": ["llm", "stt", "tts"],
                "observedComponents": ["llm", "stt", "tts"],
                "missingComponents": [],
                "excludedComponents": [],
            }, sort_keys=True),
        }


def test_livekit_model_usage_dict_fixture_uses_declared_component_type():
    with capture_events() as transport:
        record_livekit_usage({
            "usage": {
                "model_usage": [
                    {"type": "stt_usage", "provider": "deepgram", "model": "nova", "audio_duration": 1.0},
                    {"type": "tts_usage", "provider": "elevenlabs", "model": "turbo", "characters_count": 12},
                ],
            },
        }, session_id="room-2", revision=1)
    assert [event.context["voiceSessionId"] for event in transport.events] == ["room-2", "room-2"]
    assert [event.component_id.split(":")[3] for event in transport.events] == ["deepgram", "elevenlabs"]


def test_livekit_billing_units_are_authoritative_and_drop_derived_total_tokens():
    with capture_events() as transport:
        record_livekit_usage({
            "model_usage": [{
                "type": "llm_usage",
                "provider": "openai",
                "model": "gpt-4o",
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
            }],
        }, session_id="room-billing", revision=1)
        event = transport.events[0]
        assert event.billing_units == {"input_tokens": 3, "output_tokens": 2}
        wire = Transport("https://example.test", "key", 50, 5, False)._serialize(event)
    assert wire["billingUnits"] == {"input_tokens": 3, "output_tokens": 2}


def test_livekit_mixed_native_billing_families_require_explicit_selection_without_amount():
    with capture_events() as transport:
        record_livekit_usage({
            "model_usage": [{
                "type": "llm_usage",
                "provider": "openai",
                "model": "gpt-4o",
                "input_tokens": 3,
                "audio_duration": 1.5,
            }],
        }, session_id="room-mixed", revision=1)
    event = transport.events[0]
    assert event.billing_units is None
    assert event.context["voiceBilling"]["status"] == "selection_required"


def test_livekit_reordered_rows_keep_provider_model_ids_and_full_coverage():
    first = {
        "model_usage": [
            {"type": "llm_usage", "provider": "provider-a", "model": "same-model", "input_tokens": 1},
            {"type": "llm_usage", "provider": "provider-b", "model": "same-model", "input_tokens": 2},
            {"type": "stt_usage", "provider": "deepgram", "model": "nova", "audio_duration": 1},
        ],
    }
    second = {"model_usage": list(reversed(first["model_usage"]))}
    with capture_events() as transport:
        record_livekit_usage(first, session_id="room-reordered", revision=1,
                             expected_components=("stt", "llm", "tts"))
        ids_first = {event.component_id for event in transport.events}
        coverage_first = {json.dumps(event.context["voiceCoverage"], sort_keys=True) for event in transport.events}
        transport.events.clear()
        # A reordered cumulative snapshot should address the same charges;
        # every emitted row should also expose the final missing-component set.
        record_livekit_usage(second, session_id="room-reordered", revision=2,
                             expected_components=("stt", "llm", "tts"))
        ids_second = {event.component_id for event in transport.events}
        coverage_second = {json.dumps(event.context["voiceCoverage"], sort_keys=True) for event in transport.events}
    assert ids_first == ids_second
    assert coverage_first == coverage_second == {
        json.dumps({
            "expectedComponents": ["llm", "stt", "tts"],
            "observedComponents": ["llm", "stt"],
            "missingComponents": ["tts"],
            "excludedComponents": [],
        }, sort_keys=True),
    }


def test_livekit_duplicate_provider_model_rows_use_stable_row_ids():
    update = {
        "model_usage": [
            {"type": "llm_usage", "id": "request-b", "provider": "openai", "model": "gpt", "input_tokens": 2},
            {"type": "llm_usage", "id": "request-a", "provider": "openai", "model": "gpt", "input_tokens": 1},
        ],
    }
    with capture_events() as transport:
        record_livekit_usage(update, session_id="room-ids", revision=1)
        first_ids = {event.component_id for event in transport.events}
        transport.events.clear()
        record_livekit_usage({"model_usage": list(reversed(update["model_usage"]))}, session_id="room-ids", revision=2)
        second_ids = {event.component_id for event in transport.events}
    assert first_ids == second_ids == {
        "livekit:room-ids:llm:openai:gpt:request-a",
        "livekit:room-ids:llm:openai:gpt:request-b",
    }


def test_livekit_can_exclude_aggregate_llm_when_http_is_source_of_charge():
    with capture_events() as transport:
        record_livekit_usage({
            "model_usage": [
                {"type": "stt_usage", "provider": "deepgram", "model": "nova", "audio_duration": 1.0},
                {"type": "llm_usage", "provider": "openai", "model": "gpt-4o", "input_tokens": 3, "output_tokens": 2},
                {"type": "tts_usage", "provider": "elevenlabs", "model": "turbo", "characters_count": 4},
            ],
        }, session_id="room-3", revision=1, expected_components=("stt", "llm", "tts"), exclude_components={"llm"})
    assert len(transport.events) == 2
    assert {event.component_id.split(":")[2] for event in transport.events} == {"stt", "tts"}
    assert all(event.context["voiceCoverage"]["excludedComponents"] == ["llm"] for event in transport.events)
    assert all("llm" not in event.context["voiceCoverage"]["expectedComponents"] for event in transport.events)


def test_enforced_admission_refusal_is_typed_and_prevents_provider_call():
    calls = []

    def refuse(payload):
        calls.append(payload)
        return {"allowed": False, "reason": "price_unavailable", "policyId": "p1"}

    state = patch_module._state
    previous = (state.admission, state.admission_mode)
    state.admission = refuse
    state.admission_mode = "enforce"
    try:
        patch_module._run_admission_sync({"id": "call", "projectId": "p"})
    except CostKeyPolicyRefused as exc:
        assert exc.reason == "price_unavailable"
        assert exc.policy_id == "p1"
    else:
        raise AssertionError("expected policy refusal")
    finally:
        state.admission, state.admission_mode = previous
    assert calls == [{"id": "call", "projectId": "p"}]


def test_policy_settlement_payload_releases_an_allowed_call():
    state = patch_module._state
    previous = (state.admission, state.admission_mode, state.project_id)
    settled = []

    class Admission:
        def __call__(self, payload):
            return {
                "decision": "allow",
                "reservationId": "reservation-1",
                "policies": [{"unit": "calls", "result": "allow"}],
            }

        def settle(self, payload):
            settled.append(payload)

    state.admission = Admission()
    state.admission_mode = "enforce"
    state.project_id = "project-test"
    try:
        decision = patch_module._run_admission_sync({"id": "call-1"})
        patch_module._settle_sync(decision, None, 200, "call-1")
    finally:
        state.admission, state.admission_mode, state.project_id = previous
    assert settled == [{
        "projectId": "project-test",
        "reservationId": "reservation-1",
        "settlementId": "call-1:settlement",
        "outcome": "success",
        "providerAttempted": True,
    }]


def test_real_http_provider_admit_and_settle_lifecycle():
    requests = []
    settled = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            requests.append((self.path, body, self.headers.get("Authorization")))
            if self.path.endswith("/policy/admit"):
                response = {
                    "decision": "allow",
                    "providerCallAllowed": True,
                    "reservationId": "reservation-http-1",
                    "policies": [{"unit": "calls", "result": "allow"}],
                }
            elif self.path.endswith("/policy/settle"):
                settled.append(body)
                response = {"state": "settled"}
            elif self.path == "/provider":
                response = {
                    "id": "provider-http-1",
                    "model": "fixture-model",
                    "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                }
            else:
                self.send_response(404)
                self.end_headers()
                return
            payload = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            return

    class FixtureExtractor:
        provider = "fixture"

        def match(self, url):
            return url.endswith("/provider")

        def extract_usage(self, body):
            from costkey.types import NormalizedUsage
            usage = body.get("usage") if isinstance(body, dict) else None
            if not isinstance(usage, dict):
                return None
            return NormalizedUsage(
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                total_tokens=usage.get("total_tokens"),
            )

        def extract_model(self, request_body, response_body):
            for body in (response_body, request_body):
                if isinstance(body, dict) and isinstance(body.get("model"), str):
                    return body["model"]
            return None

    from costkey.providers import register_extractor
    register_extractor(FixtureExtractor())
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state = patch_module._state
    previous = {
        "transport": state.transport,
        "project_id": state.project_id,
        "capture_body": state.capture_body,
        "local_capture_body": state.local_capture_body,
        "config_known": state.config_known,
        "config_status": state.config_status,
        "before_send": state.before_send,
        "before_send_event": state.before_send_event,
        "admission": state.admission,
        "admission_mode": state.admission_mode,
    }
    sink = CaptureTransport()
    state.transport = sink
    state.project_id = "project-http"
    state.capture_body = False
    state.local_capture_body = False
    state.config_known = True
    state.config_status = "local_disabled"
    state.before_send = None
    state.before_send_event = None
    state.admission_mode = "enforce"
    try:
        from costkey.policy import PolicyAdmissionOptions
        base = "http://127.0.0.1:%s" % server.server_address[1]
        state.admission = PolicyAdmissionOptions(
            f"{base}/api/v1/sdk/policy/admit",
            "dsn-key",
            mode="enforce",
            integration={"supported": True, "admissionSupported": True, "verified": True},
        )
        patch_module._patch_httpx()
        state.patched = True
        with httpx.Client() as http:
            response = http.post(f"{base}/provider", json={"model": "fixture-model"})
            assert response.status_code == 200
    finally:
        if state.patched:
            patch_module.unpatch()
        for key, value in previous.items():
            setattr(state, key, value)
        server.shutdown()
        server.server_close()

    assert [path for path, _, _ in requests] == [
        "/api/v1/sdk/policy/admit",
        "/provider",
        "/api/v1/sdk/policy/settle",
    ]
    admission_body = requests[0][1]
    assert admission_body["admissionId"] == admission_body["requestId"]
    assert admission_body["integration"]["verified"] is True
    assert settled[0]["outcome"] == "success"
    assert settled[0]["actualUsage"]["totalTokens"] == 5


def test_init_policy_uses_canonical_dsn_routes_and_settles_real_provider_call():
    received = []
    settled = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = json.dumps({
                "projectId": "policy-project",
                "capture": {"mode": "metadata", "bodyCaptureEnabled": False, "redactionEnabled": True},
                "providers": [],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            received.append((self.path, body, self.headers.get("Authorization")))
            if self.path.endswith("/policy/admit"):
                response = {
                    "decision": "allow",
                    "providerCallAllowed": True,
                    "reservationId": "init-reservation",
                    "policies": [{"unit": "calls", "result": "allow"}],
                }
            elif self.path.endswith("/policy/settle"):
                settled.append(body)
                response = {"state": "settled"}
            elif self.path.split("?", 1)[0] == "/provider-policy":
                response = {
                    "id": "provider-policy-request",
                    "model": "fixture-model",
                    "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
                }
            else:
                response = {"accepted": len(body.get("events", []))}
            payload = json.dumps(response).encode()
            self.send_response(200 if self.path != "/api/v1/events" else 202)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            return

    class PolicyExtractor:
        provider = "fixture"

        def match(self, url):
            return urlparse(url).path == "/provider-policy"

        def extract_usage(self, body):
            from costkey.types import NormalizedUsage
            usage = body.get("usage") if isinstance(body, dict) else None
            if not isinstance(usage, dict):
                return None
            return NormalizedUsage(
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                total_tokens=usage.get("total_tokens"),
            )

        def extract_model(self, request_body, response_body):
            for body in (response_body, request_body):
                if isinstance(body, dict) and isinstance(body.get("model"), str):
                    return body["model"]
            return None

    from urllib.parse import urlparse
    from costkey.providers import register_extractor
    register_extractor(PolicyExtractor())
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    import costkey
    port = server.server_address[1]
    try:
        costkey.init(
            dsn=f"http://dsn-policy-key@127.0.0.1:{port}/policy-project",
            capture_body=False,
            max_batch_size=10,
            flush_interval=60,
            scan_callgraph=False,
            policy={
                "mode": "enforce",
                "integration": {"supported": True, "admissionSupported": True, "verified": True},
            },
            config_refresh_interval=None,
        )
        with httpx.Client() as http:
            response = http.post(
                f"http://provider-user:provider-pass@127.0.0.1:{port}/provider-policy?key=provider-secret",
                json={"model": "fixture-model"},
            )
            assert response.status_code == 200
        costkey.flush()
    finally:
        costkey.shutdown()
        server.shutdown()
        server.server_close()

    paths = [path for path, _body, _auth in received]
    assert "/api/v1/sdk/policy/admit" in paths
    assert "/api/v1/sdk/policy/settle" in paths
    admission = next(body for path, body, _auth in received if path.endswith("/policy/admit"))
    assert admission["admissionId"] == admission["requestId"]
    assert admission["integration"]["verified"] is True
    assert "requestBody" not in admission
    assert settled and settled[0]["actualUsage"]["totalTokens"] == 5
    ingest = next(body for path, body, _auth in received if path == "/api/v1/events")
    event = ingest["events"][0]
    assert event["eventType"] == "call"
    assert event["requestBody"] is None
    assert event["url"] == f"http://127.0.0.1:{port}/provider-policy"
    assert "provider-secret" not in json.dumps(ingest)
    assert "provider-pass" not in json.dumps(ingest)


def test_provider_native_units_and_reported_cost_are_preserved():
    usage = OpenAIExtractor().extract_usage({
        "model": "gpt-4o",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost": 0.004,
            "input_audio_tokens": 12,
            "audio_duration": 1.5,
        },
    })
    assert usage is not None
    assert usage.cost_usd == 0.004
    assert usage.native_units == {"input_audio_tokens": 12, "audio_duration": 1.5}


def test_provider_url_credentials_and_query_keys_are_not_emitted():
    from costkey.patch import _safe_url

    value = _safe_url("https://user:pass@api.example.test/v1/generate?key=secret&api-version=1&X-Api-Key=another&sig=token")
    assert value == "https://api.example.test/v1/generate"
    assert "secret" not in value
    assert "pass" not in value


def test_transport_sanitizes_legacy_event_url_at_outbound_boundary():
    from costkey.types import CostKeyEvent

    event = CostKeyEvent(
        id="legacy-url",
        project_id="project-test",
        url="https://user:password@provider.example/v1?key=sentinel#fragment",
    )
    serialized = Transport("https://example.test", "key", 50, 5, False)._serialize(event)
    assert serialized["url"] == "https://provider.example/v1"
    assert "sentinel" not in json.dumps(serialized)


def test_runtime_provider_capabilities_are_applied_from_dsn_config():
    apply_runtime_providers([{
        "id": "custom-router",
        "hostname": "llm.internal.test",
        "usage_input_path": "usage.input_tokens",
        "usage_output_path": "usage.output_tokens",
        "model_path": "model",
    }])
    try:
        extractor = find_extractor("https://llm.internal.test/v1/chat")
        assert extractor is not None
        usage = extractor.extract_usage({"model": "custom", "usage": {"input_tokens": 4, "output_tokens": 3}})
        assert usage is not None
        assert usage.total_tokens == 7
        assert extractor.extract_model({"model": "requested"}, {"model": "served"}) == "served"
    finally:
        apply_runtime_providers([])


def test_status_only_admission_failure_is_unavailable(monkeypatch):
    class Response:
        status_code = 403

        def json(self):
            return {"error": "gateway"}

    monkeypatch.setattr("costkey.policy.httpx.post", lambda *args, **kwargs: Response())
    from costkey.policy import PolicyAdmissionOptions

    result = PolicyAdmissionOptions("https://policy.test/admit", "secret")({"id": "1"})
    assert result["allowed"] is None
    assert result["reason"] == "decision_unavailable"


def test_real_policy_client_uses_admit_and_settle_routes(monkeypatch):
    seen = []

    class Response:
        status_code = 200

        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    def post(url, **kwargs):
        seen.append((url, kwargs["json"]))
        if url.endswith("/settle"):
            return Response({"settled": True})
        return Response({
            "decision": "allow",
            "reservationId": "reservation-1",
            "policies": [{"unit": "calls", "result": "allow"}],
        })

    monkeypatch.setattr("costkey.policy.httpx.post", post)
    from costkey.policy import PolicyAdmissionOptions

    client_policy = PolicyAdmissionOptions(
        "https://app.test/api/v1/sdk/policy/admit",
        "secret",
        mode="enforce",
        integration={"supported": True, "admissionSupported": True, "verified": True},
    )
    decision = client_policy({"id": "a1", "projectId": "p", "provider": "openai"})
    client_policy.settle({
        "projectId": "p",
        "reservationId": decision["reservationId"],
        "settlementId": "s1",
        "outcome": "success",
        "providerAttempted": True,
    })
    assert seen[0][0].endswith("/policy/admit")
    assert seen[0][1]["admissionId"] == "a1"
    assert seen[1][0].endswith("/policy/settle")


def test_monitor_mode_never_blocks_on_typed_refusal():
    state = patch_module._state
    previous = (state.admission, state.admission_mode)
    state.admission = lambda payload: {"allowed": False, "reason": "budget_exhausted"}
    state.admission_mode = "monitor"
    try:
        patch_module._run_admission_sync({"id": "call"})
    finally:
        state.admission, state.admission_mode = previous


def test_policy_options_object_matches_mapping_configuration():
    admission = lambda payload: {"allowed": True}
    resolved, mode = client._resolve_policy(
        policy=PolicyOptions(mode="enforce", admission=admission),
        policy_admission=None,
        policy_endpoint=None,
        policy_mode="monitor",
        policy_timeout=2,
        endpoint="https://app.test/api/v1/events",
        auth_key="key",
    )
    assert resolved is admission
    assert mode == "enforce"
