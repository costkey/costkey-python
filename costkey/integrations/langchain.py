"""LangChain and LangGraph integration (``pip install "costkey[langchain]"``).

Pass the handler as a callback::

    from costkey.integrations.langchain import CostKeyCallbackHandler

    handler = CostKeyCallbackHandler()
    result = graph.invoke(inputs, config={"callbacks": [handler]})

Chains become ``chain`` spans, LangGraph nodes ``step`` spans, the root run
a ``workflow`` span, tools ``tool`` spans, retrievers ``retrieval`` spans and
``AgentExecutor`` runs ``agent`` spans. Set ``metadata={"costkey_kind": "agent"}``
on a runnable (or pass ``agent_names``) to label your own agents. Runs tagged
``langsmith:hidden`` (LangGraph's internal writers and branches) get no span.

Chat model and LLM runs get no span either: the HTTP capture records each
model call and attributes it to the chain, node or tool it ran in. The
handler runs inline (``run_inline = True``) so that attribution also holds
in async graphs.
"""
from __future__ import annotations

from typing import Any, Iterable
from uuid import UUID

from costkey.integrations._tracker import SpanTracker

try:  # pragma: no cover - exercised only when the framework is installed
    from langchain_core.callbacks import BaseCallbackHandler as _HandlerBase
except Exception:  # The framework is optional; the handler is duck-typed.
    _HandlerBase = object  # type: ignore[assignment,misc]

FRAMEWORK = "langchain"
_HIDDEN_TAG = "langsmith:hidden"


def _run_name(serialized: Any, kwargs: dict[str, Any], fallback: str) -> str:
    name = kwargs.get("name")
    if isinstance(name, str) and name:
        return name
    if isinstance(serialized, dict):
        value = serialized.get("name")
        if isinstance(value, str) and value:
            return value
        ids = serialized.get("id")
        if isinstance(ids, list) and ids and isinstance(ids[-1], str):
            return ids[-1]
    return fallback


def _metadata_attributes(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    attributes: dict[str, Any] = {}
    for key in ("langgraph_node", "langgraph_step", "langgraph_checkpoint_ns", "ls_provider", "ls_model_name"):
        value = metadata.get(key)
        if value is None or isinstance(value, (str, int, float, bool)):
            if value is not None:
                attributes[key] = value
    return attributes


class CostKeyCallbackHandler(_HandlerBase):  # type: ignore[misc,valid-type]
    """A LangChain callback handler that records runs as CostKey spans."""

    raise_error = False
    run_inline = True

    def __init__(self, *, agent_names: Iterable[str] = ("AgentExecutor",)) -> None:
        super().__init__()
        self._tracker = SpanTracker(FRAMEWORK)
        self._agent_names = frozenset(agent_names)

    # ── helpers ──

    def _start(self, run_id: UUID, parent_run_id: UUID | None, name: str, kind: str,
               *, tags: Any = None, metadata: Any = None, input: Any = None,
               extra: dict[str, Any] | None = None) -> None:
        if isinstance(tags, list) and _HIDDEN_TAG in tags:
            self._tracker.skip(run_id, parent_run_id)
            return
        attributes = {
            "run_id": str(run_id),
            **_metadata_attributes(metadata),
            **(extra or {}),
        }
        self._tracker.start(run_id, name, kind, parent=parent_run_id, attributes=attributes, input=input)

    def _end(self, run_id: UUID, *, output: Any = None, error: Any = None) -> None:
        if self._tracker.is_open(run_id):
            self._tracker.end(run_id, output=output, error=error)
        else:
            self._tracker.forget(run_id)

    def _chain_kind(self, name: str, parent_run_id: UUID | None, metadata: Any) -> str:
        meta = metadata if isinstance(metadata, dict) else {}
        override = meta.get("costkey_kind")
        if isinstance(override, str) and override:
            return override
        if name in self._agent_names:
            return "agent"
        if meta.get("langgraph_node") is not None and meta.get("langgraph_node") == name:
            return "step"
        if parent_run_id is None:
            return "workflow"
        return "chain"

    # ── chains, graphs and graph nodes ──

    def on_chain_start(self, serialized: dict[str, Any], inputs: Any, *, run_id: UUID,
                       parent_run_id: UUID | None = None, tags: list[str] | None = None,
                       metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        name = _run_name(serialized, kwargs, "chain")
        self._start(run_id, parent_run_id, name, self._chain_kind(name, parent_run_id, metadata),
                    tags=tags, metadata=metadata, input=inputs)

    def on_chain_end(self, outputs: Any, *, run_id: UUID, parent_run_id: UUID | None = None,
                     **kwargs: Any) -> None:
        self._end(run_id, output=outputs)

    def on_chain_error(self, error: BaseException, *, run_id: UUID,
                       parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        self._end(run_id, error=error)

    # ── tools ──

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id: UUID,
                      parent_run_id: UUID | None = None, tags: list[str] | None = None,
                      metadata: dict[str, Any] | None = None, inputs: dict[str, Any] | None = None,
                      **kwargs: Any) -> None:
        self._start(run_id, parent_run_id, _run_name(serialized, kwargs, "tool"), "tool",
                    tags=tags, metadata=metadata, input=inputs if inputs is not None else input_str)

    def on_tool_end(self, output: Any, *, run_id: UUID, parent_run_id: UUID | None = None,
                    **kwargs: Any) -> None:
        self._end(run_id, output=output)

    def on_tool_error(self, error: BaseException, *, run_id: UUID,
                      parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        self._end(run_id, error=error)

    # ── retrievers ──

    def on_retriever_start(self, serialized: dict[str, Any], query: str, *, run_id: UUID,
                           parent_run_id: UUID | None = None, tags: list[str] | None = None,
                           metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._start(run_id, parent_run_id, _run_name(serialized, kwargs, "retriever"), "retrieval",
                    tags=tags, metadata=metadata, input=query)

    def on_retriever_end(self, documents: Any, *, run_id: UUID,
                         parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        count = len(documents) if isinstance(documents, (list, tuple)) else None
        handle = self._tracker.handle(run_id)
        if handle is not None and count is not None:
            handle.set_attribute("documents", count)
        self._end(run_id)

    def on_retriever_error(self, error: BaseException, *, run_id: UUID,
                           parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        self._end(run_id, error=error)

    # ── model calls: no span, the HTTP capture records them ──

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID,
                     parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        self._tracker.skip(run_id, parent_run_id)

    def on_chat_model_start(self, serialized: dict[str, Any], messages: Any, *, run_id: UUID,
                            parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        self._tracker.skip(run_id, parent_run_id)

    def on_llm_end(self, response: Any, *, run_id: UUID, parent_run_id: UUID | None = None,
                   **kwargs: Any) -> None:
        self._tracker.forget(run_id)

    def on_llm_error(self, error: BaseException, *, run_id: UUID,
                     parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        self._tracker.forget(run_id)


__all__ = ["CostKeyCallbackHandler"]
