"""Scripting and assertion helpers for the e2e test layer.

Tests read as::

    scripted = scripted_llm([
        tool_call_turn("list_skills"),
        answer("done"),
    ])
    with patched_build_llm(scripted):
        graph, _ = build_reference_deep_agent(...)
    result = await graph.ainvoke(...)
    assert_tool_invoked(result, "list_skills")
    assert "done" in last_ai_message(result).content

``scripted_llm`` wraps the ``ConversationRecording`` /
``LLMInteraction`` pydantic scaffolding so a test author doesn't have
to reproduce the three-layer nesting every time.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    AIMessage,
    ToolMessage,
)

from langgraph_kit.replay import (
    ConversationRecording,
    LLMInteraction,
    RecordedChatModel,
)

__all__ = [
    "CapturingScriptedChatModel",
    "answer",
    "assert_tool_invoked",
    "capturing_scripted_llm",
    "last_ai_message",
    "multi_tool_call_turn",
    "scripted_llm",
    "tool_call_turn",
]


def tool_call_turn(
    tool_name: str,
    args: dict[str, Any] | None = None,
    call_id: str | None = None,
) -> dict[str, Any]:
    """Build an ``output_message`` dict for a turn that calls exactly one tool.

    The emitted shape matches what ``RecordedChatModel`` consumes in
    ``_interaction_to_result``: ``{"content": str, "tool_calls": [...]}``.
    LangChain's tool-call format requires ``id``, ``name``, ``args`` on
    each call.
    """
    return {
        "content": "",
        "tool_calls": [
            {
                "id": call_id or f"call_{tool_name}",
                "name": tool_name,
                "args": args or {},
            }
        ],
    }


def multi_tool_call_turn(
    calls: list[tuple[str, dict[str, Any] | None]],
) -> dict[str, Any]:
    """Build an ``output_message`` dict for a turn that calls multiple tools in parallel.

    Each ``calls`` entry is ``(tool_name, args_or_None)``. Call IDs are
    auto-generated as ``call_<name>_<index>`` — tests that need to
    assert on specific IDs should build the dict directly rather than
    using this helper.
    """
    return {
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{name}_{i}",
                "name": name,
                "args": args or {},
            }
            for i, (name, args) in enumerate(calls)
        ],
    }


def answer(content: str) -> dict[str, Any]:
    """Build an ``output_message`` dict for a final text response (no tool calls)."""
    return {"content": content, "tool_calls": []}


def scripted_llm(turns: list[dict[str, Any]]) -> RecordedChatModel:
    """Create a ``RecordedChatModel`` from a list of ``output_message`` dicts.

    Each dict becomes one ``LLMInteraction`` served in order. When the
    graph invokes the model more times than ``len(turns)``, the model
    falls back to fuzzy content matching; when no match is found it
    raises ``ReplayMismatchError``. Tests that expect the LLM to NEVER
    be called (e.g. command short-circuits) pass ``turns=[]``.
    """
    return RecordedChatModel(
        recording=ConversationRecording(
            interactions=[
                LLMInteraction(sequence_num=i + 1, output_message=msg)
                for i, msg in enumerate(turns)
            ],
        )
    )


class CapturingScriptedChatModel(RecordedChatModel):
    """``RecordedChatModel`` that records the messages it was called with.

    Normal ``RecordedChatModel`` serves scripted responses but discards
    what the graph sent it. For assertions like "did the system prompt
    the LLM received contain X?" we need the input side. This subclass
    accumulates each call's input messages in ``captured_calls`` and is
    otherwise identical.

    ``create_agent`` and LangGraph may call the model from multiple
    coroutines/threads. A plain list with ``append`` is GIL-safe for
    single-process async; sequence ordering isn't guaranteed in
    parallel-tool-call scenarios but is correct for the common single-
    model-path flow the e2e scenarios exercise.
    """

    # Declared at the class level so pydantic lets instances carry the
    # attribute. Using ``default_factory`` via Field keeps each instance's
    # list independent.
    captured_calls: list[list[Any]] = []

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Pydantic may have already assigned the class-level default
        # list; swap it for a per-instance list so two models created
        # back-to-back don't share capture state.
        object.__setattr__(self, "captured_calls", [])

    def _generate(  # type: ignore[override]
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.captured_calls.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def capturing_scripted_llm(turns: list[dict[str, Any]]) -> CapturingScriptedChatModel:
    """Return a ``CapturingScriptedChatModel`` with the given scripted turns.

    Identical to ``scripted_llm`` but records inputs. Use for assertions
    like "the prompt the LLM received did / did not contain X".
    """
    return CapturingScriptedChatModel(
        recording=ConversationRecording(
            interactions=[
                LLMInteraction(sequence_num=i + 1, output_message=msg)
                for i, msg in enumerate(turns)
            ],
        )
    )


def _messages(state: Any) -> list[Any]:
    """Extract the message list from a LangGraph state dict or dict-like object."""
    if hasattr(state, "get"):
        messages = state.get("messages")
        if messages is not None:
            return list(messages)
    if hasattr(state, "messages"):
        return list(state.messages)
    msg = f"Cannot find messages in state of type {type(state).__name__}"
    raise AssertionError(msg)


def assert_tool_invoked(state: Any, tool_name: str) -> ToolMessage:
    """Assert a ``ToolMessage`` with ``name == tool_name`` exists in ``state``.

    Returns the matching ``ToolMessage`` so callers can further inspect
    its ``content``. Fails with a message that lists what was actually
    in the state.
    """
    messages = _messages(state)
    for msg in messages:
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == tool_name:
            return msg
    summary = [(type(m).__name__, getattr(m, "name", None)) for m in messages]
    raise AssertionError(
        f"Expected ToolMessage(name={tool_name!r}) in state; got {summary}"
    )


def last_ai_message(state: Any) -> AIMessage:
    """Return the last ``AIMessage`` in ``state``, or fail with a useful error."""
    messages = _messages(state)
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    summary = [type(m).__name__ for m in messages]
    raise AssertionError(
        f"No AIMessage found in state; got {summary}"
    )
