"""Framework integrations, run against the real frameworks with mocked HTTP.

Each test drives the framework's own run loop. Model calls go through the
patched HTTP clients (MockTransport underneath), so the tests also check that
an integration never emits call events of its own (no double counting) and
that calls are attributed to the innermost framework span.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from conftest import capturing

MOCK = httpx.MockTransport(lambda request: httpx.Response(200, json={"id": "x", "results": []}))


def tree(sink):
    """(kind:name, status, parent kind:name) per ended span, and kind:name per call."""
    names = sink.span_names()
    spans = [(names[e.id], e.span_status, names.get(e.parent_span_id)) for e in sink.ended_spans()]
    calls = [(e.provider, names.get(e.span_id)) for e in sink.calls()]
    return spans, calls


def one_trace(sink):
    traces = {e.trace_id for e in sink.events if e.event_type in ("span", "call")}
    assert len(traces) == 1, traces
    return traces.pop()


# ── OpenAI Agents SDK ──

def _responses_handler():
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        base = {"id": f"resp_{state['n']}", "object": "response", "created_at": 1, "model": "gpt-4o",
                "status": "completed", "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                          "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}
        if state["n"] == 1:
            base["output"] = [{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather",
                               "arguments": '{"city": "SF"}', "status": "completed"}]
        else:
            base["output"] = [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                               "content": [{"type": "output_text", "text": "Sunny", "annotations": []}]}]
        return httpx.Response(200, json=base)

    return handler


def _weather_agent():
    agents = pytest.importorskip("agents")
    from agents.models.openai_responses import OpenAIResponsesModel
    from openai import AsyncOpenAI

    @agents.function_tool
    def get_weather(city: str) -> str:
        """Weather for a city."""
        httpx.Client(transport=MOCK).post("https://api.tavily.com/search", json={"query": city})
        return "sunny in " + city

    client = AsyncOpenAI(api_key="sk-test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(_responses_handler())))
    return agents.Agent(name="weather", instructions="be brief", tools=[get_weather],
                        model=OpenAIResponsesModel(model="gpt-4o", openai_client=client))


def test_openai_agents_tracing_processor():
    agents = pytest.importorskip("agents")
    from costkey.integrations.openai_agents import CostKeyTracingProcessor

    agent = _weather_agent()
    agents.set_trace_processors([CostKeyTracingProcessor()])
    try:
        with capturing() as sink:
            result = asyncio.run(agents.Runner.run(agent, "weather in SF?"))
    finally:
        agents.set_trace_processors([])
    assert result.final_output == "Sunny"
    spans, calls = tree(sink)
    assert ("agent:weather", "ok", "workflow:Agent workflow") in spans
    assert ("tool:get_weather", "ok", "step:turn 1") in spans
    assert ("step:turn 1", "ok", "agent:weather") in spans
    # Two model calls (one per turn) captured once each, plus the tool's own call.
    assert calls == [("openai", "step:turn 1"), ("tavily", "tool:get_weather"), ("openai", "step:turn 2")]
    one_trace(sink)
    agent_span = next(e for e in sink.ended_spans() if e.span_kind == "agent")
    assert agent_span.attributes["framework"] == "openai_agents"
    assert agent_span.attributes["openai_agents"]["type"] == "agent"


def test_openai_agents_run_hooks():
    agents = pytest.importorskip("agents")
    from costkey.integrations.openai_agents import CostKeyRunHooks

    import costkey

    agent = _weather_agent()

    async def run():
        # Hooks run in child tasks, so calls attribute to the span around the run.
        async with costkey.agent("support-run"):
            return await agents.Runner.run(agent, "weather?", hooks=CostKeyRunHooks(),
                                           run_config=agents.RunConfig(tracing_disabled=True))

    with capturing() as sink:
        result = asyncio.run(run())
    assert result.final_output == "Sunny"
    spans, calls = tree(sink)
    assert ("tool:get_weather", "ok", "agent:weather") in spans
    assert ("agent:weather", "ok", "agent:support-run") in spans
    assert calls == [("openai", "agent:support-run"), ("tavily", "agent:support-run"), ("openai", "agent:support-run")]
    one_trace(sink)


# ── LangChain / LangGraph ──

def _graph(async_nodes: bool):
    pytest.importorskip("langgraph")
    from typing import TypedDict

    from langchain_core.tools import tool
    from langgraph.graph import END, START, StateGraph

    @tool
    def lookup(q: str) -> str:
        """Look something up."""
        httpx.Client(transport=MOCK).post("https://api.exa.ai/search", json={"query": q})
        return "found " + q

    class State(TypedDict):
        q: str
        a: str

    def plan(state):
        httpx.Client(transport=MOCK).post("https://api.openai.com/v1/chat/completions", json={"model": "gpt-4o"})
        return {"q": state["q"]}

    def act(state):
        return {"a": lookup.invoke({"q": state["q"]})}

    async def aplan(state):
        async with httpx.AsyncClient(transport=MOCK) as http:
            await http.post("https://api.anthropic.com/v1/messages", json={"model": "claude"})
        return {"q": state["q"]}

    async def aact(state):
        return {"a": await lookup.ainvoke({"q": state["q"]})}

    graph = StateGraph(State)
    graph.add_node("plan", aplan if async_nodes else plan)
    graph.add_node("act", aact if async_nodes else act)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", END)
    return graph.compile()


@pytest.mark.parametrize("async_nodes", [False, True])
def test_langgraph_nodes_tools_and_calls(async_nodes):
    from costkey.integrations.langchain import CostKeyCallbackHandler

    graph = _graph(async_nodes)
    config = {"callbacks": [CostKeyCallbackHandler()]}
    with capturing() as sink:
        if async_nodes:
            result = asyncio.run(graph.ainvoke({"q": "x"}, config=config))
        else:
            result = graph.invoke({"q": "x"}, config=config)
    assert result["a"] == "found x"
    spans, calls = tree(sink)
    assert spans == [
        ("step:plan", "ok", "workflow:LangGraph"),
        ("tool:lookup", "ok", "step:act"),
        ("step:act", "ok", "workflow:LangGraph"),
        ("workflow:LangGraph", "ok", None),
    ]
    assert calls == [("anthropic" if async_nodes else "openai", "step:plan"), ("exa", "tool:lookup")]
    one_trace(sink)


def test_langchain_chain_error_and_retriever():
    pytest.importorskip("langchain_core")
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.runnables import RunnableLambda

    from costkey.integrations.langchain import CostKeyCallbackHandler

    class Retriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
            httpx.Client(transport=MOCK).post("https://api.voyageai.com/v1/embeddings", json={"input": [query]})
            return [Document(page_content="a"), Document(page_content="b")]

    def boom(_):
        raise ValueError("bad")

    with capturing() as sink:
        docs = Retriever().invoke("q", config={"callbacks": [CostKeyCallbackHandler()]})
        with pytest.raises(ValueError):
            RunnableLambda(boom).invoke(1, config={"callbacks": [CostKeyCallbackHandler()]})
    assert len(docs) == 2
    ended = sink.ended_spans()
    retrieval = next(e for e in ended if e.span_kind == "retrieval")
    assert retrieval.attributes["langchain"]["documents"] == 2
    failed = next(e for e in ended if e.span_status == "error")
    assert failed.error_type == "ValueError"
    assert [call.provider for call in sink.calls()] == ["voyage"]


# ── LlamaIndex ──

def test_llama_index_query_engine_tool_and_workflow():
    pytest.importorskip("llama_index.core")
    from llama_index.core import Document, Settings, VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.llms import MockLLM
    from llama_index.core.tools import FunctionTool
    from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

    from costkey.integrations.llama_index import instrument

    instrument()
    Settings.llm = MockLLM(max_tokens=8)
    Settings.embed_model = MockEmbedding(embed_dim=8)
    index = VectorStoreIndex.from_documents([Document(text="hello world"), Document(text="bye")])

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        httpx.Client(transport=MOCK).post("https://api.openai.com/v1/embeddings", json={"model": "x"})
        return a + b

    class Flow(Workflow):
        @step
        async def go(self, ev: StartEvent) -> StopEvent:
            async with httpx.AsyncClient(transport=MOCK) as http:
                await http.post("https://api.anthropic.com/v1/messages", json={"model": "c"})
            return StopEvent(result="done")

    class Broken(Workflow):
        @step
        async def go(self, ev: StartEvent) -> StopEvent:
            raise KeyError("x")

    async def workflows():
        assert await Flow(timeout=10).run() == "done"
        with pytest.raises(Exception):
            await Broken(timeout=10).run()

    with capturing() as sink:
        index.as_query_engine().query("hi")
        assert FunctionTool.from_defaults(add).call(a=1, b=2).raw_output == 3
        asyncio.run(workflows())
    spans, calls = tree(sink)
    assert ("retrieval:VectorIndexRetriever.retrieve", "ok", "chain:RetrieverQueryEngine.query") in spans
    assert ("tool:FunctionTool.call", "ok", None) in spans
    # Step spans are named by the step's qualified name (local classes here).
    local = "test_llama_index_query_engine_tool_and_workflow.<locals>."
    assert (f"step:{local}Flow.go", "ok", "workflow:Flow.run") in spans
    assert (f"step:{local}Broken.go", "error", "workflow:Broken.run") in spans
    assert not any(name.split(".")[-1].startswith("_") for name, _, _ in spans)
    assert calls == [("openai", "tool:FunctionTool.call"), ("anthropic", f"step:{local}Flow.go")]
    broken = next(e for e in sink.ended_spans() if e.span_name == "Broken.run")
    assert broken.error_type == "KeyError"


# ── Pydantic AI ──

def test_pydantic_ai_capability():
    pytest.importorskip("pydantic_ai")
    from openai import AsyncOpenAI
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from costkey.integrations.pydantic_ai import CostKeyCapability

    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        message = ({"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "double", "arguments": '{"x": 2}'}}]}
            if state["n"] == 1 else {"role": "assistant", "content": "4"})
        return httpx.Response(200, json={
            "id": f"c{state['n']}", "object": "chat.completion", "created": 1, "model": "gpt-4o",
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if state["n"] == 1 else "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}})

    client = AsyncOpenAI(api_key="sk-test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    agent = Agent(OpenAIChatModel("gpt-4o", provider=OpenAIProvider(openai_client=client)), name="calc",
                  capabilities=[CostKeyCapability()])

    @agent.tool_plain
    def double(x: int) -> int:
        return x * 2

    with capturing() as sink:
        assert asyncio.run(agent.run("double 2")).output == "4"
    spans, calls = tree(sink)
    assert spans == [("tool:double", "ok", "agent:calc"), ("agent:calc", "ok", None)]
    assert calls == [("openai", "agent:calc"), ("openai", "agent:calc")]


# ── smolagents ──

def test_smolagents_steps():
    pytest.importorskip("smolagents")
    from smolagents import ToolCallingAgent, tool
    from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole, Model

    from costkey.integrations.smolagents import instrument

    class Fake(Model):
        def __init__(self):
            super().__init__(model_id="fake")
            self.n = 0

        def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
            httpx.Client(transport=MOCK).post("https://api.openai.com/v1/chat/completions", json={"model": "gpt-4o"})
            self.n += 1
            name, arguments = ("add", {"a": 1, "b": 2}) if self.n == 1 else ("final_answer", {"answer": "3"})
            call = ChatMessageToolCall(id=f"c{self.n}", type="function",
                                       function=ChatMessageToolCallFunction(name=name, arguments=arguments))
            return ChatMessage(role=MessageRole.ASSISTANT, content=None, tool_calls=[call])

    @tool
    def add(a: int, b: int) -> int:
        """Add.

        Args:
            a: first
            b: second
        """
        return a + b

    agent = instrument(ToolCallingAgent(tools=[add], model=Fake(), verbosity_level=0, name="adder"))
    with capturing() as sink:
        assert str(agent.run("1+2")) == "3"
    spans, calls = tree(sink)
    assert spans == [("step:step 1", "ok", "agent:adder"), ("step:step 2", "ok", "agent:adder"), ("agent:adder", "ok", None)]
    assert calls == [("openai", "agent:adder"), ("openai", "agent:adder")]
    step_one = sink.ended_spans()[0]
    assert step_one.attributes["smolagents"]["step_number"] == 1
    one_trace(sink)


# ── OpenTelemetry bridge and AutoGen AgentChat ──

def test_opentelemetry_bridge_attributes_and_attribution():
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider

    from costkey.integrations.opentelemetry import CostKeySpanProcessor

    provider = TracerProvider()
    processor = CostKeySpanProcessor()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("my.instrumentation")
    try:
        for capture in (False, True):
            with capturing(capture_body=capture) as sink:
                with tracer.start_as_current_span("chat gpt-4o", attributes={
                    "gen_ai.operation.name": "chat", "gen_ai.request.model": "gpt-4o",
                    "gen_ai.prompt": "secret prompt", "gen_ai.usage.input_tokens": 12,
                }):
                    with tracer.start_as_current_span("execute_tool lookup") as inner:
                        httpx.Client(transport=MOCK).post("https://api.openai.com/v1/chat/completions", json={})
                        inner.record_exception(ValueError("x"))
                        from opentelemetry.trace import Status, StatusCode
                        inner.set_status(Status(StatusCode.ERROR))
            spans, calls = tree(sink)
            assert spans == [("custom:execute_tool lookup", "error", "custom:chat gpt-4o"), ("custom:chat gpt-4o", "ok", None)]
            assert calls == [("openai", "custom:execute_tool lookup")]
            outer = sink.ended_spans()[1]
            assert outer.attributes["otelScope"] == "my.instrumentation"
            assert outer.attributes["otelKind"] == "INTERNAL"
            assert outer.attributes["otelName"] == "chat gpt-4o"
            assert outer.attributes["otelParentSpanId"] is None
            assert sink.ended_spans()[0].attributes["otelParentSpanId"] == outer.attributes["otelSpanId"]
            assert sink.ended_spans()[0].error_type == "ValueError"
            otel = outer.attributes["otel"]
            if capture:
                assert otel["gen_ai.prompt"] == "secret prompt"
            else:
                assert otel == {"gen_ai.operation.name": {"$s": 4}, "gen_ai.request.model": "gpt-4o",
                                "gen_ai.prompt": {"$s": 13}, "gen_ai.usage.input_tokens": 12}
    finally:
        processor.shutdown()


def test_autogen_agentchat_through_the_otel_bridge():
    pytest.importorskip("autogen_agentchat")
    from opentelemetry import trace

    from autogen_agentchat.agents import AssistantAgent
    from autogen_core import FunctionCall
    from autogen_core.models import ChatCompletionClient, CreateResult, ModelInfo, RequestUsage

    from costkey.integrations.autogen import instrument

    processor = instrument()
    assert hasattr(trace.get_tracer_provider(), "add_span_processor")

    class Fake(ChatCompletionClient):
        def __init__(self):
            self.n = 0

        async def create(self, messages, *, tools=[], tool_choice="auto", json_output=None, extra_create_args={},
                         cancellation_token=None):
            async with httpx.AsyncClient(transport=MOCK) as http:
                await http.post("https://api.openai.com/v1/chat/completions", json={"model": "gpt-4o"})
            self.n += 1
            usage = RequestUsage(prompt_tokens=1, completion_tokens=1)
            if self.n == 1:
                return CreateResult(finish_reason="function_calls", usage=usage, cached=False,
                                    content=[FunctionCall(id="c1", name="add", arguments='{"a": 1, "b": 2}')])
            return CreateResult(finish_reason="stop", content="3", usage=usage, cached=False)

        def create_stream(self, *args, **kwargs):
            raise NotImplementedError

        async def close(self):
            pass

        def actual_usage(self):
            return RequestUsage(prompt_tokens=0, completion_tokens=0)

        total_usage = actual_usage

        def count_tokens(self, messages, *, tools=[]):
            return 0

        remaining_tokens = count_tokens

        @property
        def capabilities(self):
            return self.model_info

        @property
        def model_info(self):
            return ModelInfo(vision=False, function_calling=True, json_output=False, family="unknown",
                             structured_output=False)

    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        async with httpx.AsyncClient(transport=MOCK) as http:
            await http.post("https://api.tavily.com/search", json={"query": "q"})
        return a + b

    async def run():
        agent = AssistantAgent("calc", model_client=Fake(), tools=[add], reflect_on_tool_use=True)
        return await agent.run(task="1+2")

    try:
        with capturing() as sink:
            result = asyncio.run(run())
        assert processor is instrument()
    finally:
        processor.shutdown()  # the global provider keeps processors; stop this one
    assert result.messages[-1].content == "3"
    spans, calls = tree(sink)
    assert ("custom:execute_tool add", "ok", "custom:invoke_agent calc") in spans
    assert calls == [("openai", "custom:invoke_agent calc"), ("tavily", "custom:execute_tool add"),
                     ("openai", "custom:invoke_agent calc")]
    tool_span = next(e for e in sink.ended_spans() if e.span_name == "execute_tool add")
    assert tool_span.attributes["otel"]["gen_ai.tool.call.id"] == "c1"
    assert tool_span.attributes["otelScope"] == "autogen-core"


# ── AG2 ──

def test_ag2_middleware():
    pytest.importorskip("ag2")
    from ag2 import Agent, tool
    from ag2.events import ToolCallEvent
    from ag2.testing import TestConfig

    from costkey.integrations.ag2 import costkey_middleware

    @tool
    async def search(q: str) -> str:
        """Search."""
        async with httpx.AsyncClient(transport=MOCK) as http:
            await http.post("https://api.exa.ai/search", json={"query": q})
        return "ok"

    agent = Agent("researcher", config=TestConfig(ToolCallEvent(name="search", arguments='{"q": "x"}'), "done"),
                  tools=[search], middleware=[costkey_middleware("researcher")])
    with capturing() as sink:
        asyncio.run(agent.ask("go"))
    spans, calls = tree(sink)
    assert spans == [("tool:search", "ok", "agent:researcher"), ("agent:researcher", "ok", None)]
    assert calls == [("exa", "tool:search")]


# ── Strands ──

def test_strands_hooks():
    pytest.importorskip("strands")
    from openai import AsyncOpenAI
    from strands import Agent, tool
    from strands.models.openai import OpenAIModel

    from costkey.integrations.strands import CostKeyHooks

    state = {"n": 0}

    def sse(chunks):
        return ("".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()

    def handler(request):
        state["n"] += 1
        base = {"id": f"c{state['n']}", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o"}
        usage = dict(base, choices=[], usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})
        if state["n"] == 1:
            chunks = [dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "tool_calls": [
                {"index": 0, "id": "call_1", "type": "function",
                 "function": {"name": "search", "arguments": '{"q": "x"}'}}]}, "finish_reason": None}]),
                      dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]), usage]
        else:
            chunks = [dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "content": "done"}, "finish_reason": None}]),
                      dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}]), usage]
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse(chunks))

    @tool
    async def search(q: str) -> str:
        """Search the web.

        Args:
            q: query
        """
        async with httpx.AsyncClient(transport=MOCK) as http:
            await http.post("https://api.tavily.com/search", json={"query": q})
        return "nothing"

    async def run():
        client = AsyncOpenAI(api_key="sk-test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        agent = Agent(name="researcher", model=OpenAIModel(client=client, model_id="gpt-4o"), tools=[search],
                      hooks=[CostKeyHooks()], callback_handler=None)
        return await agent.invoke_async("find x")

    with capturing() as sink:
        assert str(asyncio.run(run())).strip() == "done"
    spans, calls = tree(sink)
    assert spans == [("tool:search", "ok", "agent:researcher"), ("agent:researcher", "ok", None)]
    assert sorted(calls) == [("openai", "agent:researcher"), ("openai", "agent:researcher"), ("tavily", "tool:search")]
    assert all(call.telemetry["stream"]["chunkCount"] == 3 for call in sink.calls() if call.provider == "openai")


# ── Google ADK ──

def test_google_adk_plugin():
    pytest.importorskip("google.adk")
    from google.adk.agents import LlmAgent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from costkey.integrations.google_adk import CostKeyPlugin

    class FakeLlm(BaseLlm):
        n: int = 0

        async def generate_content_async(self, llm_request, stream=False):
            async with httpx.AsyncClient(transport=MOCK) as http:
                await http.post("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=S",
                                json={"contents": []})
            self.n += 1
            part = (types.Part(function_call=types.FunctionCall(name="lookup", args={"q": "x"}, id="fc_1"))
                    if self.n == 1 else types.Part(text="done"))
            yield LlmResponse(content=types.Content(role="model", parts=[part]))

    async def lookup(q: str) -> dict:
        """Look up."""
        async with httpx.AsyncClient(transport=MOCK) as http:
            await http.post("https://api.exa.ai/search", json={"query": q})
        return {"r": "ok"}

    async def run():
        agent = LlmAgent(name="helper", model=FakeLlm(model="fake"), tools=[lookup], instruction="help")
        runner = InMemoryRunner(agent=agent, app_name="demo", plugins=[CostKeyPlugin()])
        session = await runner.session_service.create_session(app_name="demo", user_id="u")
        message = types.Content(role="user", parts=[types.Part(text="hi")])
        async for _ in runner.run_async(user_id="u", session_id=session.id, new_message=message):
            pass

    with capturing() as sink:
        asyncio.run(run())
    spans, calls = tree(sink)
    assert spans == [("tool:lookup", "ok", "agent:helper"), ("agent:helper", "ok", "workflow:demo"), ("workflow:demo", "ok", None)]
    assert calls == [("google", "agent:helper"), ("exa", "tool:lookup"), ("google", "agent:helper")]
    assert all("key=" not in call.telemetry["request"]["url"] for call in sink.calls())


# ── Claude Agent SDK (the one integration that emits call events) ──

def test_claude_agent_sdk_hooks_and_usage_messages():
    pytest.importorskip("claude_agent_sdk")
    import anyio
    from claude_agent_sdk import ClaudeAgentOptions, query
    from claude_agent_sdk._internal.transport import Transport

    from costkey.integrations.claude_agent_sdk import instrument, uninstrument

    session = "sess-1"
    usage = {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100, "service_tier": "standard"}

    class FakeCLI(Transport):
        """Speaks the CLI's stream-json control protocol, so the SDK's real Query, hook
        dispatch and message parser run unchanged."""

        def __init__(self):
            self.send, self.receive = anyio.create_memory_object_stream(100)
            self.hook_ids = {}

        async def connect(self):
            pass

        def is_ready(self):
            return True

        async def end_input(self):
            pass

        async def close(self):
            pass

        async def write(self, data):
            message = json.loads(data)
            if message.get("type") == "control_request" and message["request"]["subtype"] == "initialize":
                for event, matchers in (message["request"].get("hooks") or {}).items():
                    self.hook_ids[event] = matchers[0]["hookCallbackIds"][0]
                await self.send.send({"type": "control_response", "response": {
                    "subtype": "success", "request_id": message["request_id"], "response": {}}})
                asyncio.get_running_loop().create_task(self.script())

        async def hook(self, event, data, tool_use_id=None):
            await self.send.send({"type": "control_request", "request_id": f"h-{event}-{tool_use_id}", "request": {
                "subtype": "hook_callback", "callback_id": self.hook_ids[event], "tool_use_id": tool_use_id,
                "input": {"session_id": session, "hook_event_name": event, "transcript_path": "/t", "cwd": "/", **data}}})
            await asyncio.sleep(0.02)

        async def script(self):
            send = self.send
            await send.send({"type": "system", "subtype": "init", "session_id": session, "model": "claude-sonnet-4-5", "uuid": "u0"})
            await send.send({"type": "assistant", "message": {
                "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5",
                "content": [{"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {"command": "ls"}}],
                "stop_reason": "tool_use", "usage": usage}, "parent_tool_use_id": None, "session_id": session, "uuid": "u1"})
            await self.hook("PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "ls"}, "tool_use_id": "tu_1"}, "tu_1")
            await self.hook("PostToolUse", {"tool_name": "Bash", "tool_input": {}, "tool_response": "a", "tool_use_id": "tu_1"}, "tu_1")
            await self.hook("SubagentStart", {"agent_id": "ag_1", "agent_type": "code-reviewer"})
            await self.hook("PreToolUse", {"tool_name": "Read", "tool_input": {}, "tool_use_id": "tu_2", "agent_id": "ag_1"}, "tu_2")
            await self.hook("PostToolUseFailure", {"tool_name": "Read", "tool_input": {}, "tool_use_id": "tu_2",
                                                   "error": "nope", "agent_id": "ag_1"}, "tu_2")
            await self.hook("SubagentStop", {"agent_id": "ag_1", "agent_type": "code-reviewer", "stop_hook_active": False,
                                             "agent_transcript_path": "/x"})
            await send.send({"type": "stream_event", "uuid": "u-se", "session_id": session, "parent_tool_use_id": None,
                             "event": {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}})
            await send.send({"type": "result", "subtype": "success", "duration_ms": 10, "duration_api_ms": 8,
                             "is_error": False, "num_turns": 2, "session_id": session, "total_cost_usd": 0.01,
                             "usage": usage, "modelUsage": {"claude-sonnet-4-5": {"inputTokens": 20, "costUSD": 0.01}},
                             "result": "done", "uuid": "u3"})
            await send.send({"type": "end"})

        async def read_messages(self):
            async for message in self.receive:
                if message.get("type") == "end":
                    return
                yield message

    async def run():
        claude = instrument()
        options = claude.with_hooks(ClaudeAgentOptions(model="claude-sonnet-4-5"))

        async def prompt():
            yield {"type": "user", "message": {"role": "user", "content": "hi"}}

        return [type(m).__name__ async for m in query(prompt=prompt(), options=options, transport=FakeCLI())]

    try:
        with capturing() as sink:
            kinds = asyncio.run(run())
    finally:
        uninstrument()
    assert "ResultMessage" in kinds
    spans, calls = tree(sink)
    assert ("tool:Bash", "ok", "agent:claude-agent") in spans
    assert ("tool:Read", "error", "agent:code-reviewer") in spans
    assert ("agent:code-reviewer", "ok", "agent:claude-agent") in spans
    assert ("agent:claude-agent", "ok", None) in spans
    # Every message carrying usage (assistant, stream event, result) is a call event.
    assert [call.id for call in sink.calls()] == ["u1", "u-se", "u3"]
    assistant = sink.calls()[0]
    assert assistant.provider == "claude-agent-sdk" and assistant.source == "claude-agent-sdk"
    assert assistant.telemetry["source"] == "claude-agent-sdk"
    assert assistant.telemetry["request"] is None
    assert assistant.telemetry["response"]["status"] is None
    assert assistant.telemetry["response"]["headers"] == {}
    assert assistant.telemetry["response"]["bytes"] is None
    body = assistant.telemetry["response"]["body"]
    assert body["message"]["id"] == "msg_1"
    assert body["message"]["usage"] == usage
    assert body["message"]["content"][0]["input"] == {"command": {"$s": 2}}
    assert sink.calls()[2].telemetry["response"]["body"]["total_cost_usd"] == 0.01
    failed = next(e for e in sink.ended_spans() if e.span_name == "Read")
    assert failed.error_type == "PostToolUseFailure"


# ── CrewAI and DSPy (run in an environment with openai<3, see the README) ──

def test_crewai_listener():
    pytest.importorskip("crewai")
    from crewai import Agent, Crew, Task
    from crewai.events import crewai_event_bus
    from crewai.llms.base_llm import BaseLLM
    from crewai.tools import tool

    from costkey.integrations.crewai import CostKeyCrewAIListener

    CostKeyCrewAIListener()

    class FakeLLM(BaseLLM):
        n: int = 0

        def call(self, messages, tools=None, callbacks=None, available_functions=None, from_task=None,
                 from_agent=None, response_model=None):
            httpx.Client(transport=MOCK).post("https://api.openai.com/v1/chat/completions", json={"model": "gpt-4o"})
            self.n += 1
            if self.n == 1:
                return 'Thought: use tool\nAction: lookup\nAction Input: {"q": "x"}'
            return "Thought: done\nFinal Answer: 42"

        def supports_function_calling(self):
            return False

    @tool("lookup")
    def lookup(q: str) -> str:
        """Look up q."""
        httpx.Client(transport=MOCK).post("https://api.tavily.com/search", json={"query": q})
        return "found"

    agent = Agent(role="researcher", goal="g", backstory="b", llm=FakeLLM(model="fake"), tools=[lookup], verbose=False)
    task = Task(description="find x", expected_output="answer", agent=agent)
    with capturing() as sink:
        assert str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()) == "42"
        crewai_event_bus.flush()
    spans, calls = tree(sink)
    assert ("workflow:crew", "ok", None) in spans
    assert ("task:find x", "ok", "workflow:crew") in spans
    assert ("agent:researcher", "ok", "task:find x") in spans
    tool_span = next(span for span in spans if span[0] == "tool:lookup")
    assert tool_span[1] == "ok"
    assert ("tavily", "tool:lookup") in calls
    assert sum(1 for provider, _ in calls if provider == "openai") == 2
    one_trace(sink)


def test_dspy_callback_and_lm15_transport():
    pytest.importorskip("dspy")
    import socket
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import dspy

    from costkey.integrations.dspy import CostKeyCallback

    state = {"n": 0}

    def reply(text):
        state["n"] += 1
        if "next_thought" in text and state["n"] == 1:
            out = ('[[ ## next_thought ## ]]\nlook\n\n[[ ## next_tool_name ## ]]\nlookup\n\n'
                   '[[ ## next_tool_args ## ]]\n{"q": "x"}\n\n[[ ## completed ## ]]')
        elif "next_thought" in text:
            out = ('[[ ## next_thought ## ]]\ndone\n\n[[ ## next_tool_name ## ]]\nfinish\n\n'
                   '[[ ## next_tool_args ## ]]\n{}\n\n[[ ## completed ## ]]')
        else:
            out = "[[ ## reasoning ## ]]\nok\n\n[[ ## answer ## ]]\n42\n\n[[ ## completed ## ]]"
        return {"id": f"c{state['n']}", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": out}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            data = json.dumps(reply(body.decode())).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    original = socket.create_connection

    def create_connection(address, *args, **kwargs):
        # lm15 opens raw sockets: route api.openai.com to the local stand-in.
        if address[0] == "api.openai.com":
            address = ("127.0.0.1", server.server_address[1])
        return original(address, *args, **kwargs)

    def lookup(q: str) -> str:
        """Look up q."""
        httpx.Client(transport=MOCK).post("https://api.tavily.com/search", json={"query": q})
        return "found"

    socket.create_connection = create_connection
    try:
        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key="sk-test", api_base="http://api.openai.com/v1",
                                  cache=False), callbacks=[CostKeyCallback()])
        with capturing() as sink:
            assert dspy.ReAct("question -> answer", tools=[lookup])(question="x?").answer == "42"
    finally:
        socket.create_connection = original
        server.shutdown()
    spans, calls = tree(sink)
    assert ("agent:ReAct", "ok", None) in spans
    assert ("tool:lookup", "ok", "agent:ReAct") in spans
    assert calls.count(("openai", "chain:Predict")) == 3
    assert ("tavily", "tool:lookup") in calls
    openai_call = next(call for call in sink.calls() if call.provider == "openai")
    assert openai_call.telemetry["response"]["body"]["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
