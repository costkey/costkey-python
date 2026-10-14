"""LlamaIndex integration (``pip install "costkey[llama-index]"``).

Register the span handler on the root instrumentation dispatcher::

    from costkey.integrations.llama_index import instrument
    instrument()  # get_dispatcher().add_span_handler(CostKeySpanHandler())

Agents (``FunctionAgent``, ``ReActAgent``, ``AgentWorkflow``) become ``agent``
spans, other workflows ``workflow`` spans, workflow steps ``step`` spans,
retrievers and node postprocessors ``retrieval`` spans, tools ``tool`` spans,
and query engines, chat engines and response synthesizers ``chain`` spans.
LLM and embedding spans are not recorded: the HTTP capture records those
calls and attributes them to the enclosing span. Other internal spans are
folded into their parent.
"""
from __future__ import annotations

import threading
from typing import Any, Optional

from costkey.integrations._tracker import SpanTracker, scalar_attributes

from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from pydantic import PrivateAttr

FRAMEWORK = "llama_index"


def _classes() -> dict[str, tuple[type, ...]]:
    """The llama-index base classes we map, resolved lazily and tolerantly."""
    found: dict[str, list[type]] = {key: [] for key in ("agent", "workflow", "retrieval", "tool", "chain", "model")}
    candidates = (
        ("agent", "llama_index.core.agent.workflow.base_agent", "BaseWorkflowAgent"),
        ("agent", "llama_index.core.agent.workflow.multi_agent_workflow", "AgentWorkflow"),
        ("workflow", "workflows", "Workflow"),
        ("workflow", "llama_index.core.workflow", "Workflow"),
        ("retrieval", "llama_index.core.base.base_retriever", "BaseRetriever"),
        ("retrieval", "llama_index.core.postprocessor.types", "BaseNodePostprocessor"),
        ("tool", "llama_index.core.tools.types", "BaseTool"),
        ("chain", "llama_index.core.base.base_query_engine", "BaseQueryEngine"),
        ("chain", "llama_index.core.chat_engine.types", "BaseChatEngine"),
        ("chain", "llama_index.core.response_synthesizers.base", "BaseSynthesizer"),
        ("model", "llama_index.core.base.llms.base", "BaseLLM"),
        ("model", "llama_index.core.base.embeddings.base", "BaseEmbedding"),
        ("model", "llama_index.core.base.embeddings.base_sparse", "BaseSparseEmbedding"),
    )
    import importlib

    for kind, module_name, attr in candidates:
        try:
            cls = getattr(importlib.import_module(module_name), attr)
        except Exception:
            continue
        if isinstance(cls, type):
            found[kind].append(cls)
    return {kind: tuple(classes) for kind, classes in found.items()}


_class_cache: dict[str, tuple[type, ...]] | None = None
_class_lock = threading.Lock()


def _kind(instance: Any, span_name: str) -> str | None:
    global _class_cache
    with _class_lock:
        if _class_cache is None:
            _class_cache = _classes()
    classes = _class_cache
    if instance is not None:
        for kind in ("model", "agent", "workflow", "retrieval", "tool", "chain"):
            if classes[kind] and isinstance(instance, classes[kind]):
                return None if kind == "model" else kind
        return None
    # Workflow steps are plain functions named "<Workflow>.<step>".
    return "step" if "." in span_name else None


def _span_name(id_: str) -> str:
    # Span ids are "<Class>.<method>-<uuid4>" or "<qualname>-<uuid4>".
    return id_[:-37] if len(id_) > 37 and id_[-37] == "-" else id_


class CostKeySpanHandler(BaseSpanHandler[Any]):
    """A LlamaIndex span handler that records CostKey spans."""

    _tracker: SpanTracker = PrivateAttr(default_factory=lambda: SpanTracker(FRAMEWORK))

    @classmethod
    def class_name(cls) -> str:
        return "CostKeySpanHandler"

    def new_span(self, id_: str, bound_args: Any, instance: Optional[Any] = None,
                 parent_span_id: Optional[str] = None, tags: Optional[dict[str, Any]] = None,
                 **kwargs: Any) -> Optional[Any]:
        name = _span_name(id_)
        # Private methods (``_query``, ``_retrieve``) fold into the public
        # call that wraps them.
        kind = None if name.rsplit(".", 1)[-1].startswith("_") else _kind(instance, name)
        if kind is None:
            self._tracker.skip(id_, parent_span_id)
            return id_
        attributes = {
            "span_id": id_,
            "class": type(instance).__name__ if instance is not None else None,
            **scalar_attributes(tags),
        }
        self._tracker.start(id_, name, kind, parent=parent_span_id, attributes=attributes)
        return id_

    def prepare_to_exit_span(self, id_: str, bound_args: Any, instance: Optional[Any] = None,
                             result: Optional[Any] = None, **kwargs: Any) -> Optional[Any]:
        if self._tracker.is_open(id_):
            self._tracker.end(id_, output=result if isinstance(result, (str, int, float, bool)) else None)
        else:
            self._tracker.forget(id_)
        return id_

    def prepare_to_drop_span(self, id_: str, bound_args: Any, instance: Optional[Any] = None,
                             err: Optional[BaseException] = None, **kwargs: Any) -> Optional[Any]:
        if self._tracker.is_open(id_):
            self._tracker.end(id_, error=err if err is not None else "SpanDropped")
        else:
            self._tracker.forget(id_)
        return id_


_installed: CostKeySpanHandler | None = None
_install_lock = threading.Lock()


def instrument() -> CostKeySpanHandler:
    """Add one CostKeySpanHandler to LlamaIndex's root dispatcher."""
    global _installed
    from llama_index.core.instrumentation import get_dispatcher

    with _install_lock:
        if _installed is None:
            _installed = CostKeySpanHandler()
            get_dispatcher().add_span_handler(_installed)
        return _installed


__all__ = ["CostKeySpanHandler", "instrument"]
