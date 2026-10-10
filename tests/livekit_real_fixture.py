"""Run with an isolated venv containing livekit-agents==1.8.1.

Example:
  PYTHONPATH=. /tmp/costkey-livekit.Rn9Gqh/venv/bin/python tests/livekit_real_fixture.py
"""
from __future__ import annotations

from livekit.agents import SessionUsageUpdatedEvent
from livekit.agents.metrics import AgentSessionUsage, LLMModelUsage, STTModelUsage, TTSModelUsage

import costkey.patch as patch_module
from costkey.livekit import record_livekit_usage


class Sink:
    def __init__(self) -> None:
        self.events = []

    def enqueue(self, event):
        self.events.append(event)


def main() -> None:
    state = patch_module._state
    previous = (state.transport, state.project_id, state.before_send_event)
    sink = Sink()
    state.transport = sink
    state.project_id = "fixture-project"
    state.before_send_event = None
    try:
        update = SessionUsageUpdatedEvent(usage=AgentSessionUsage(model_usage=[
            STTModelUsage(provider="deepgram", model="nova-3", audio_duration=1.25),
            LLMModelUsage(provider="openai", model="gpt-4o", input_tokens=20, output_tokens=5),
            TTSModelUsage(provider="elevenlabs", model="turbo", characters_count=30, audio_duration=2.0),
        ]))
        ids = record_livekit_usage(
            update,
            session_id="room-1",
            revision=7,
            expected_components=("stt", "llm", "tts"),
            # LiveKit's native usage classes do not include provider prices.
            # These explicit fixture amounts exercise known-cost accounting
            # without contacting a provider or pretending to reconcile an
            # invoice.
            declared_amounts={
                "stt:deepgram:nova-3": {"value": 0.01, "currency": "USD"},
                "llm:openai:gpt-4o": {"value": 0.02, "currency": "USD"},
                "tts:elevenlabs:turbo": {"value": 0.03, "currency": "USD"},
            },
            declared_pricing_ids={
                "stt:deepgram:nova-3": "fixture:deepgram",
                "llm:openai:gpt-4o": "fixture:openai",
                "tts:elevenlabs:turbo": "fixture:elevenlabs",
            },
        )
        assert len(ids) == 3
        assert {event.component_id for event in sink.events} == {
            "livekit:room-1:stt:deepgram:nova-3",
            "livekit:room-1:llm:openai:gpt-4o",
            "livekit:room-1:tts:elevenlabs:turbo",
        }
        assert sink.events[0].native_usage["audio_duration"] == 1.25
        assert sink.events[0].billing_units == {
            "input_tokens": 0,
            "output_tokens": 0,
            "audio_seconds": 1.25,
        }
        assert sink.events[0].metadata["livekit"]["type"] == "stt_usage"
        assert sink.events[0].timestamp.endswith("Z")
        assert {event.amount["value"] for event in sink.events} == {0.01, 0.02, 0.03}
        assert all(event.amount_kind == "estimated" for event in sink.events)
        assert {event.pricing_id for event in sink.events} == {
            "fixture:deepgram", "fixture:openai", "fixture:elevenlabs",
        }
        assert all(event.aggregation == "cumulative" and event.revision == 7 for event in sink.events)
        reordered = SessionUsageUpdatedEvent(usage=AgentSessionUsage(
            model_usage=list(reversed(update.usage.model_usage)),
        ))
        reordered_ids = record_livekit_usage(
            reordered,
            session_id="room-1",
            revision=8,
            expected_components=("stt", "llm", "tts"),
            declared_amounts={
                "stt:deepgram:nova-3": {"value": 0.01, "currency": "USD"},
                "llm:openai:gpt-4o": {"value": 0.02, "currency": "USD"},
                "tts:elevenlabs:turbo": {"value": 0.03, "currency": "USD"},
            },
        )
        assert set(reordered_ids) == {
            "livekit:room-1:stt:deepgram:nova-3:r8",
            "livekit:room-1:llm:openai:gpt-4o:r8",
            "livekit:room-1:tts:elevenlabs:turbo:r8",
        }
        excluded_ids = record_livekit_usage(
            update,
            session_id="room-2",
            revision=1,
            expected_components=("stt", "llm", "tts"),
            exclude_components={"llm"},
        )
        assert len(excluded_ids) == 2
        assert all(":llm:" not in event.component_id for event in sink.events[6:])
        print("livekit real fixture: 3 typed components + scoped LLM exclusion, native units, cumulative revisions: PASS")
    finally:
        state.transport, state.project_id, state.before_send_event = previous


if __name__ == "__main__":
    main()
