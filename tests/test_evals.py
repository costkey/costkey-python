"""run_eval: read a dataset with a token, run the task per item in its own trace, score it."""
from __future__ import annotations

import asyncio
import base64
import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import pytest

import costkey
import costkey.client as client_module
from costkey.evals import EvalItem, EvalVerdict, computer_action, default_eval_score
from costkey.transport import Transport
from conftest import capturing

SHA = "a" * 64
PNG = b"\x89PNG-frame"


def click(x: int, y: int) -> Dict[str, Any]:
    """An Anthropic response whose computer tool clicks at (x, y)."""
    return {"content": [{"type": "tool_use", "name": "computer", "input": {"action": "left_click", "coordinate": [x, y]}}]}


ROWS = [
    {"id": "item_good", "label": "Good", "input": {"messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": f"[image data omitted: image/png; ~10 bytes; frame {SHA}]"}},
    ]}]}, "output": click(100, 200), "model": "claude-sonnet-4-5", "provider": "anthropic", "traceId": "tr_1", "sourceEventId": "evt_1"},
    {"id": "item_bad", "label": "Bad", "input": {"image_url": f"[image data omitted: image/png; ~10 bytes; frame {'b' * 64}]"},
     "output": click(500, 500), "traceId": "tr_2", "sourceEventId": "evt_2"},
    {"id": "item_text", "label": "Good", "expected": "order 4471", "input": {"messages": []}, "output": {"content": [{"type": "text", "text": "hi"}]}, "sourceEventId": "evt_3"},
    {"id": "item_none", "label": "", "input": {}, "output": {}, "sourceEventId": "evt_4"},
]


class DatasetApi:
    def __init__(self) -> None:
        self.auth: List[str] = []
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802
                api.auth.append(self.headers.get("Authorization", ""))
                if self.path.endswith("/projects/project-test/datasets"):
                    self._send(200, json.dumps({"datasets": [{"id": "ds_1", "name": "Checkout clicks", "revision": 3}]}).encode())
                elif self.path.endswith("/datasets/ds_1/export"):
                    self._send(200, "\n".join(json.dumps(row) for row in ROWS).encode(), "application/x-ndjson")
                elif self.path.endswith(f"/frames/{SHA}"):
                    self._send(200, PNG, "image/png")
                else:
                    self._send(404, b"{}")

            def _send(self, status: int, body: bytes, kind: str = "application/json") -> None:
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}/api/v1"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def api(monkeypatch):
    server = DatasetApi()
    monkeypatch.setattr(client_module, "_initialized", True)
    monkeypatch.setattr(client_module, "_api_base", server.base)
    monkeypatch.setenv("COSTKEY_API_TOKEN", "ck_pat_test")
    yield server
    server.stop()


def candidate(item: EvalItem) -> Dict[str, Any]:
    """Clicks where the good item did; repeats the bad item's click; answers the text item."""
    if item.id == "item_text":
        return {"content": [{"type": "text", "text": "Your order 4471 ships today."}]}
    if item.id == "item_good":
        return click(110, 205)
    return item.output


def check_summary(summary: costkey.EvalSummary, sink: Any) -> None:
    assert summary.dataset == {"id": "ds_1", "name": "Checkout clicks", "revision": 3}
    assert (summary.total, summary.scored, summary.passed, summary.failed, summary.errors) == (4, 3, 2, 1, 0)
    by_item = {result.item_id: result for result in summary.results}
    assert by_item["item_good"].passed is True and "Matched the good action" in by_item["item_good"].reason
    assert by_item["item_bad"].passed is False and "Repeated the bad action" in by_item["item_bad"].reason
    assert by_item["item_text"].passed is True
    assert by_item["item_none"].passed is None
    # Every item ran in a trace of its own, inside one evaluation execution.
    assert len({result.trace_id for result in summary.results}) == 4
    executions = [event for event in sink.events if event.event_type == "lifecycle" and event.entity_type == "execution"]
    assert {event.lifecycle for event in executions} == {"started", "completed"}
    assert executions[0].group_kind == "evaluation" and executions[0].expected_items == 4
    assert executions[0].context["datasetRevision"] == "ds_1@3"
    assert all(event.execution_id == summary.execution_id for event in executions)
    traces = [event for event in sink.events if event.event_type == "lifecycle" and event.entity_type == "trace" and event.lifecycle == "started"]
    assert {event.context["caseId"] for event in traces} == {"item_good", "item_bad", "item_text", "item_none"}
    assert {event.context["phase"] for event in traces} == {"candidate"}
    # One score per scored item, on that item's trace.
    scores = [event for event in sink.events if event.event_type == "score"]
    assert sorted((event.case_id, event.score_label) for event in scores) == [("item_bad", "fail"), ("item_good", "pass"), ("item_text", "pass")]
    assert {event.trace_id for event in scores} == {by_item[event.case_id].trace_id for event in scores}
    assert all(event.score_name == "eval: Checkout clicks" and event.context["executionId"] == summary.execution_id for event in scores)


def test_run_eval_scores_every_item_in_its_own_trace(api):
    seen: List[EvalItem] = []

    def task(item: EvalItem) -> Dict[str, Any]:
        seen.append(item)
        return candidate(item)

    with capturing() as sink:
        summary = costkey.run_eval("checkout clicks", task, configuration={"model": "claude-haiku-4-5"}, concurrency=3)
    check_summary(summary, sink)
    assert set(api.auth) == {"Bearer ck_pat_test"}
    good = next(item for item in seen if item.id == "item_good")
    # The screenshot is back in the request body; the one never uploaded stays a placeholder.
    assert good.input["messages"][0]["content"][0]["source"]["data"] == base64.b64encode(PNG).decode()
    assert good.incomplete is False and good.trace_id == "tr_1" and good.model == "claude-sonnet-4-5"
    bad = next(item for item in seen if item.id == "item_bad")
    assert bad.incomplete is True and bad.input["image_url"].startswith("[image data omitted:")


def test_run_eval_async_runs_async_tasks(api):
    async def task(item: EvalItem) -> Dict[str, Any]:
        await asyncio.sleep(0.01)
        return candidate(item)

    async def score(item: EvalItem, output: Any) -> EvalVerdict:
        return default_eval_score(item, output)

    with capturing() as sink:
        summary = asyncio.run(costkey.run_eval_async("ds_1", task, score=score, concurrency=2))
    check_summary(summary, sink)


def test_a_failing_task_is_a_failed_result_not_a_crash(api):
    def task(item: EvalItem) -> Any:
        raise TimeoutError("model took too long")

    with capturing():
        summary = costkey.run_eval("Checkout clicks", task)
    assert summary.errors == 4 and summary.failed == 4
    assert summary.results[0].error == "TimeoutError: model took too long"
    assert summary.results[0].reason == "The task failed (TimeoutError)"


def test_an_async_task_needs_run_eval_async(api):
    async def task(item: EvalItem) -> Any:
        return {}

    with capturing():
        summary = costkey.run_eval("Checkout clicks", task)
    assert summary.errors == 4
    assert "run_eval_async" in (summary.results[0].error or "")


def test_run_eval_needs_init_a_token_and_a_real_dataset(api, monkeypatch):
    with capturing():
        with pytest.raises(ValueError, match="No dataset named"):
            costkey.run_eval("nope", candidate)
        monkeypatch.delenv("COSTKEY_API_TOKEN")
        with pytest.raises(RuntimeError, match="personal access token"):
            costkey.run_eval("Checkout clicks", candidate)
        monkeypatch.setattr(client_module, "_initialized", False)
        with pytest.raises(RuntimeError, match="costkey.init"):
            costkey.run_eval("Checkout clicks", candidate, api_token="ck_pat_test")


@dataclass
class ToolUse:
    type: str
    input: Dict[str, Any]


@dataclass
class Message:
    content: List[Any]


def test_the_default_scorer_reads_sdk_objects_and_openai_computer_calls():
    assert computer_action(Message(content=[ToolUse(type="tool_use", input={"action": "left_click", "coordinate": [10, 20]})])) == {"type": "left_click", "x": 10.0, "y": 20.0}
    openai = {"output": [{"type": "computer_call", "action": {"type": "click", "x": 300, "y": 40}}]}
    item = EvalItem(id="i", label="Bad", note=None, expected={"action": "click", "coordinate": [300, 45]}, input={}, output={}, model=None, provider=None, trace_id=None, source_event_id="e")
    assert default_eval_score(item, openai).passed is True
    assert default_eval_score(item, {"output": [{"type": "computer_call", "action": {"type": "click", "x": 900, "y": 40}}]}).passed is False
    assert default_eval_score(item, {"output": []}).reason == "Expected click at 300, 45, got no action"


def test_score_events_serialize_for_the_server():
    from costkey.types import CostKeyEvent

    transport = Transport(endpoint="https://costkey.example.com/api/v1/events", auth_key="k", max_batch_size=10, flush_interval=60, debug=False)
    wire = transport._serialize(CostKeyEvent(id="eval_x_1", event_type="score", source="code", project_id="p", trace_id="tr", score_name="eval: Checkout clicks", score_value=1, score_label="pass", score_comment="Matched"))
    assert (wire["eventType"], wire["name"], wire["value"], wire["label"], wire["comment"], wire["traceId"], wire["source"]) == (
        "score", "eval: Checkout clicks", 1, "pass", "Matched", "tr", "code",
    )
