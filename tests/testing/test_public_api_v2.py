"""Tests for the v2 additions to ``langgraph_kit.testing`` (#42 v2).

Covers ``FakeCheckpointer``, the promoted ``scripted_llm`` /
``tool_call_turn`` / ``answer`` / ``last_ai_message`` /
``assert_tool_invoked`` helpers, and the public API surface.
"""

from __future__ import annotations

import importlib.metadata
from typing import TypedDict

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)


class _ChatState(TypedDict, total=False):
    messages: list[BaseMessage]


class TestPublicApiSurface:
    """Every promised name imports cleanly from ``langgraph_kit.testing``."""

    def test_v2_names_importable(self) -> None:
        from langgraph_kit.testing import (
            FakeCheckpointer,
            FakeItem,
            FakeStore,
            answer,
            assert_namespace_contains,
            assert_namespace_empty,
            assert_tool_invoked,
            last_ai_message,
            multi_tool_call_turn,
            scripted_llm,
            tool_call_turn,
        )

        assert callable(FakeCheckpointer)
        assert callable(FakeItem)
        assert callable(FakeStore)
        assert callable(answer)
        assert callable(assert_namespace_contains)
        assert callable(assert_namespace_empty)
        assert callable(assert_tool_invoked)
        assert callable(last_ai_message)
        assert callable(multi_tool_call_turn)
        assert callable(scripted_llm)
        assert callable(tool_call_turn)


class TestScriptedLlmHelpers:
    """``scripted_llm`` / ``tool_call_turn`` / ``answer`` produce a working model."""

    def test_answer_shape(self) -> None:
        from langgraph_kit.testing import answer

        out = answer("hi")
        assert out == {"content": "hi", "tool_calls": []}

    def test_tool_call_turn_default_id(self) -> None:
        from langgraph_kit.testing import tool_call_turn

        out = tool_call_turn("search", {"q": "foo"})
        assert out["content"] == ""
        assert len(out["tool_calls"]) == 1
        call = out["tool_calls"][0]
        assert call["name"] == "search"
        assert call["args"] == {"q": "foo"}
        assert call["id"] == "call_search"

    def test_tool_call_turn_custom_id(self) -> None:
        from langgraph_kit.testing import tool_call_turn

        out = tool_call_turn("search", call_id="my-id-1")
        assert out["tool_calls"][0]["id"] == "my-id-1"

    def test_multi_tool_call_turn_assigns_indexed_ids(self) -> None:
        from langgraph_kit.testing import multi_tool_call_turn

        out = multi_tool_call_turn([("a", None), ("b", {"x": 1})])
        ids = [c["id"] for c in out["tool_calls"]]
        assert ids == ["call_a_0", "call_b_1"]

    def test_scripted_llm_returns_recorded_chat_model(self) -> None:
        from langgraph_kit.replay import RecordedChatModel
        from langgraph_kit.replay.models import LLMInteraction
        from langgraph_kit.testing import answer, scripted_llm

        llm = scripted_llm([answer("first"), answer("second")])
        assert isinstance(llm, RecordedChatModel)
        # Two interactions registered, in declared order.
        interactions = llm.recording.interactions
        assert len(interactions) == 2
        assert isinstance(interactions[0], LLMInteraction)
        assert isinstance(interactions[1], LLMInteraction)
        assert interactions[0].output_message["content"] == "first"
        assert interactions[1].output_message["content"] == "second"

    def test_scripted_llm_empty_turns(self) -> None:
        """``turns=[]`` is the supported way to script "should never call LLM"."""
        from langgraph_kit.testing import scripted_llm

        llm = scripted_llm([])
        assert llm.recording.interactions == []


class TestStateAssertions:
    """``assert_tool_invoked`` / ``last_ai_message`` against state shapes."""

    def test_assert_tool_invoked_finds_message(self) -> None:
        from langgraph_kit.testing import assert_tool_invoked

        state = {
            "messages": [
                HumanMessage(content="hi"),
                ToolMessage(content="42", tool_call_id="x", name="adder"),
            ]
        }
        msg = assert_tool_invoked(state, "adder")
        assert isinstance(msg, ToolMessage)
        assert msg.content == "42"

    def test_assert_tool_invoked_raises_with_summary(self) -> None:
        from langgraph_kit.testing import assert_tool_invoked

        state = {"messages": [HumanMessage(content="hi")]}
        with pytest.raises(AssertionError, match="Expected ToolMessage"):
            assert_tool_invoked(state, "missing")

    def test_last_ai_message_returns_most_recent(self) -> None:
        from langgraph_kit.testing import last_ai_message

        state = {
            "messages": [
                AIMessage(content="early"),
                HumanMessage(content="hi"),
                AIMessage(content="late"),
            ]
        }
        last = last_ai_message(state)
        assert last.content == "late"

    def test_last_ai_message_raises_when_absent(self) -> None:
        from langgraph_kit.testing import last_ai_message

        with pytest.raises(AssertionError, match="No AIMessage"):
            last_ai_message({"messages": [HumanMessage(content="hi")]})


class TestFakeCheckpointer:
    """``FakeCheckpointer`` behaves like ``InMemorySaver`` plus introspection."""

    def test_dump_state_empty_thread(self) -> None:
        from langgraph_kit.testing import FakeCheckpointer

        cp = FakeCheckpointer()
        assert cp.dump_state("nonexistent") == {}

    def test_assert_thread_has_messages_empty(self) -> None:
        from langgraph_kit.testing import FakeCheckpointer

        cp = FakeCheckpointer()
        cp.assert_thread_has_messages("nonexistent", 0)

    def test_assert_thread_has_messages_mismatch_raises(self) -> None:
        from langgraph_kit.testing import FakeCheckpointer

        cp = FakeCheckpointer()
        with pytest.raises(AssertionError, match="Expected thread"):
            cp.assert_thread_has_messages("nonexistent", 5)

    async def test_dump_state_after_real_graph_run(self) -> None:
        """``dump_state`` reads back what a real graph actually wrote.

        Uses a minimal :class:`StateGraph` so the assertion exercises
        the production save/load path through ``InMemorySaver`` rather
        than poking at private fields.
        """
        from langgraph.graph import (  # pyright: ignore[reportMissingImports]
            END,
            START,
            StateGraph,
        )

        from langgraph_kit.testing import FakeCheckpointer

        cp = FakeCheckpointer()

        async def _node(state: _ChatState) -> _ChatState:
            existing = list(state.get("messages") or [])
            return {"messages": [*existing, AIMessage(content="ok")]}

        graph = StateGraph(_ChatState)
        graph.add_node("n", _node)
        graph.add_edge(START, "n")
        graph.add_edge("n", END)
        compiled = graph.compile(checkpointer=cp)
        await compiled.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "t1"}},
        )

        state = cp.dump_state("t1")
        assert "messages" in state
        assert len(state["messages"]) == 2

    async def test_assert_thread_has_messages_after_real_run(self) -> None:
        """The assertion helper passes when the count matches a real graph run."""
        from langgraph.graph import (  # pyright: ignore[reportMissingImports]
            END,
            START,
            StateGraph,
        )

        from langgraph_kit.testing import FakeCheckpointer

        cp = FakeCheckpointer()

        async def _node(state: _ChatState) -> _ChatState:
            existing = list(state.get("messages") or [])
            return {"messages": [*existing, AIMessage(content="ok")]}

        graph = StateGraph(_ChatState)
        graph.add_node("n", _node)
        graph.add_edge(START, "n")
        graph.add_edge("n", END)
        compiled = graph.compile(checkpointer=cp)
        await compiled.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "t2"}},
        )

        cp.assert_thread_has_messages("t2", 2)
        with pytest.raises(AssertionError, match="Expected thread"):
            cp.assert_thread_has_messages("t2", 99)


class TestPytestPluginEntryPoint:
    """The pytest11 entry point is registered correctly."""

    def test_entry_point_registered(self) -> None:
        eps = importlib.metadata.entry_points(group="pytest11")
        names = {ep.name for ep in eps}
        assert "langgraph_kit_testing" in names

    def test_entry_point_target(self) -> None:
        eps = importlib.metadata.entry_points(group="pytest11")
        ours = next(ep for ep in eps if ep.name == "langgraph_kit_testing")
        # The target value form is "module:object" or just "module".
        assert ours.value == "langgraph_kit.testing.pytest_plugin"


class TestPluginFixturesAreUsable:
    """Smoke-test the auto-registered fixtures from ``pytest_plugin``."""

    def test_fake_store_fixture(self, fake_store: object) -> None:
        from langgraph_kit.testing import FakeStore

        assert isinstance(fake_store, FakeStore)

    def test_fake_checkpointer_fixture(self, fake_checkpointer: object) -> None:
        from langgraph_kit.testing import FakeCheckpointer

        assert isinstance(fake_checkpointer, FakeCheckpointer)

    def test_scripted_llm_factory_builds_a_model(
        self, scripted_llm_factory: object
    ) -> None:
        from langgraph_kit.replay import RecordedChatModel
        from langgraph_kit.testing import answer

        assert callable(scripted_llm_factory)
        llm = scripted_llm_factory([answer("ok")])  # pyright: ignore[reportCallIssue]
        assert isinstance(llm, RecordedChatModel)
