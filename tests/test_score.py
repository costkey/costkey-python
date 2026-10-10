"""costkey.score(): judge/human/code scores on runs."""
from __future__ import annotations

import pytest

import costkey
import costkey.patch as patch_module
from costkey.transport import Transport
from test_devx import capture_events


def _scores(transport):
    return [e for e in transport.events if e.event_type == "score"]


def test_score_attaches_to_the_current_span_and_trace():
    with capture_events() as transport:
        with costkey.agent("support") as span:
            costkey.score("faithfulness", 0.82, label="pass", source="judge", comment="grounded")
        (score,) = _scores(transport)
        assert score.score_name == "faithfulness"
        assert score.score_value == 0.82
        assert score.score_label == "pass"
        assert score.source == "judge"
        assert score.span_id == span.id
        assert score.trace_id == span.trace_id
        assert score.score_comment == "grounded"
        assert score.id.startswith("sc_")


def test_score_for_another_run_does_not_inherit_the_span():
    with capture_events() as transport:
        with costkey.agent("support"):
            costkey.score("accuracy", 1, trace_id="tr_other", id="sc_fixed", source="human")
        (score,) = _scores(transport)
        assert (score.id, score.trace_id, score.span_id) == ("sc_fixed", "tr_other", None)


def test_comment_is_content_and_follows_body_capture():
    with capture_events() as transport:
        patch_module._state.capture_body = False
        costkey.score("tone", 3, trace_id="tr_1", comment="private reasoning")
        (score,) = _scores(transport)
        assert score.score_comment is None


def test_comment_is_bounded():
    with capture_events() as transport:
        costkey.score("tone", 3, trace_id="tr_1", comment="c" * 1500)
        assert len(_scores(transport)[0].score_comment) == 1000


def test_score_serializes_to_the_wire_shape():
    with capture_events() as transport:
        costkey.score("faithfulness", 0.4, label="fail", trace_id="tr_1", span_id="sp_1", source="code", timestamp="2026-10-01T00:00:00.000Z")
        wire = Transport._serialize(None, _scores(transport)[0])  # type: ignore[arg-type]
    assert wire["eventType"] == "score"
    assert wire["source"] == "code"
    assert (wire["name"], wire["value"], wire["label"]) == ("faithfulness", 0.4, "fail")
    assert (wire["traceId"], wire["spanId"]) == ("tr_1", "sp_1")
    assert wire["timestamp"] == "2026-10-01T00:00:00.000Z"
    assert "comment" not in wire


@pytest.mark.parametrize("kwargs", [
    {"name": "", "value": 1, "trace_id": "tr"},
    {"name": "x" * 121, "value": 1, "trace_id": "tr"},
    {"name": "x", "value": float("nan"), "trace_id": "tr"},
    {"name": "x", "value": "1", "trace_id": "tr"},
    {"name": "x", "value": True, "trace_id": "tr"},
    {"name": "x", "value": 1, "trace_id": "tr", "label": "l" * 61},
    {"name": "x", "value": 1, "trace_id": "tr", "source": "robot"},
    {"name": "x", "value": 1},
])
def test_invalid_scores_raise(kwargs):
    with capture_events():
        name = kwargs.pop("name")
        value = kwargs.pop("value")
        with pytest.raises(ValueError):
            costkey.score(name, value, **kwargs)


def test_before_send_event_can_drop_scores():
    with capture_events() as transport:
        patch_module._state.before_send_event = lambda e: None if e.event_type == "score" and e.score_name == "drop" else e
        costkey.score("drop", 0, trace_id="tr_1")
        costkey.score("keep", 1, trace_id="tr_1")
        assert [e.score_name for e in _scores(transport)] == ["keep"]


def test_score_before_init_is_a_no_op():
    state = patch_module._state
    previous = (state.transport, state.project_id)
    state.transport, state.project_id = None, None
    try:
        costkey.score("x", 1, trace_id="tr_1")
    finally:
        state.transport, state.project_id = previous
