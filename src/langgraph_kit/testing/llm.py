"""Scripted-LLM helpers for kit consumers' tests.

Promotes the e2e layer's ``scripted_llm`` / ``tool_call_turn`` /
``answer`` builders to public API so downstream agents that build on
the kit can drive deterministic graph runs without copy-pasting the
``RecordedChatModel`` scaffolding.

Quick start::

    from langgraph_kit.testing import scripted_llm, tool_call_turn, answer

    llm = scripted_llm([
        tool_call_turn("search", {"q": "foo"}),
        answer("Found it."),
    ])
    with patch("my_app.build_llm", return_value=llm):
        graph = build_my_agent(...)
    result = await graph.ainvoke(...)

For asserting on what the graph *sent* to the LLM (input messages, not
just outputs), pair with the kit's e2e ``capturing_scripted_llm`` —
that helper is e2e-internal and intentionally not promoted here; the
common case is asserting on outputs.
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


def tool_call_turn(
    tool_name: str,
    args: dict[str, Any] | None = None,
    call_id: str | None = None,
) -> dict[str, Any]:
    """Build an ``output_message`` dict for a turn that calls one tool.

    The shape matches what :class:`RecordedChatModel` expects:
    ``{"content": str, "tool_calls": [{"id", "name", "args"}]}``. Use
    when scripting an LLM turn that should fire a tool call.
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
    """Build an ``output_message`` dict for parallel tool calls.

    Each entry in *calls* is ``(tool_name, args_or_None)``. Call IDs
    are auto-generated as ``call_<name>_<index>``; tests asserting on
    specific IDs should build the dict directly.
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
    """Build an ``output_message`` dict for a final text response."""
    return {"content": content, "tool_calls": []}


def scripted_llm(turns: list[dict[str, Any]]) -> RecordedChatModel:
    """Create a :class:`RecordedChatModel` from a list of turns.

    Each ``turns`` element becomes one :class:`LLMInteraction` served
    in order. When the graph invokes the model more times than
    ``len(turns)`` the model falls back to fuzzy content matching;
    when no match is found it raises :class:`ReplayMismatchError`.
    Tests that expect the LLM to never be called pass ``turns=[]``.
    """
    return RecordedChatModel(
        recording=ConversationRecording(
            interactions=[
                LLMInteraction(sequence_num=i + 1, output_message=msg)
                for i, msg in enumerate(turns)
            ],
        )
    )


def _messages(state: Any) -> list[Any]:
    """Extract the message list from a LangGraph state dict-like."""
    if hasattr(state, "get"):
        messages = state.get("messages")
        if messages is not None:
            return list(messages)
    if hasattr(state, "messages"):
        return list(state.messages)
    msg = f"Cannot find messages in state of type {type(state).__name__}"
    raise AssertionError(msg)


def assert_tool_invoked(state: Any, tool_name: str) -> ToolMessage:
    """Assert a ``ToolMessage`` with ``name == tool_name`` exists in *state*.

    Returns the matching message so callers can further assert on its
    ``content``. Fails with a summary of what was actually present.
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
    """Return the last :class:`AIMessage` in *state* or fail with a useful error."""
    messages = _messages(state)
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    summary = [type(m).__name__ for m in messages]
    raise AssertionError(f"No AIMessage found in state; got {summary}")


__all__ = [
    "answer",
    "assert_tool_invoked",
    "last_ai_message",
    "multi_tool_call_turn",
    "scripted_llm",
    "tool_call_turn",
]
