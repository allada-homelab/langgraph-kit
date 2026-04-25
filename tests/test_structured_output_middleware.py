"""Coverage — StructuredOutputMiddleware validation + retry.

Verifies the opt-in contract: when no schema is configured, no
validation runs (caller compose-time choice). When a schema is given,
the middleware:

- skips messages that aren't terminal (tool calls in flight)
- accepts a valid ``<output_schema>`` block
- nudges with the schema rendered as JSON Schema on a missing block
- nudges on malformed JSON
- nudges on schema-mismatch JSON
- gives up after ``max_nudges`` with a single explanatory AIMessage
- resets the per-run counter via ``abefore_agent``
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from langgraph_kit.core.resilience import (
    StructuredOutputMiddleware,
    extract_structured_output,
    format_schema_instruction,
    parse_structured_output,
)


class _Recipe(BaseModel):
    title: str = Field(..., min_length=1)
    ingredients: list[str] = Field(..., min_length=1)
    minutes: int = Field(..., ge=1)


def _state(messages: list[Any]) -> dict[str, Any]:
    return {"messages": list(messages)}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_extract_structured_output_returns_block_contents() -> None:
    text = 'thinking: this is fine\n<output_schema>{"a": 1}</output_schema>\nthe end'
    assert extract_structured_output(text) == '{"a": 1}'


def test_extract_structured_output_returns_none_when_missing() -> None:
    assert extract_structured_output("no block here") is None


def test_extract_structured_output_handles_multiline() -> None:
    text = '<output_schema>\n{\n  "a": 1\n}\n</output_schema>'
    extracted = extract_structured_output(text)
    assert extracted is not None
    assert json.loads(extracted) == {"a": 1}


def test_parse_structured_output_returns_validated_instance() -> None:
    text = (
        '<output_schema>{"title": "tacos", "ingredients": ["corn", "beef"],'
        ' "minutes": 30}</output_schema>'
    )
    parsed = parse_structured_output(text, _Recipe)
    assert parsed is not None
    assert parsed.title == "tacos"
    assert parsed.minutes == 30


def test_parse_structured_output_returns_none_on_validation_error() -> None:
    # missing minutes
    text = '<output_schema>{"title": "x", "ingredients": ["a"]}</output_schema>'
    assert parse_structured_output(text, _Recipe) is None


def test_parse_structured_output_returns_none_on_invalid_json() -> None:
    text = "<output_schema>{not json}</output_schema>"
    assert parse_structured_output(text, _Recipe) is None


def test_format_schema_instruction_contains_schema_name_and_block_marker() -> None:
    instruction = format_schema_instruction(_Recipe)
    assert "<output_schema>" in instruction
    assert "</output_schema>" in instruction
    # The JSON Schema rendering should mention the model fields somewhere.
    assert "ingredients" in instruction
    assert "minutes" in instruction


# ---------------------------------------------------------------------------
# Middleware behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_middleware_passes_through_when_message_has_tool_calls() -> None:
    mw = StructuredOutputMiddleware(_Recipe)
    msg = AIMessage(content="", tool_calls=[{"id": "1", "name": "t", "args": {}}])
    result = await mw.aafter_model(_state([msg]), runtime=None)
    assert result is None


@pytest.mark.asyncio
async def test_middleware_passes_through_when_message_is_empty() -> None:
    """EmptyTurnMiddleware owns empty-message nudges; this middleware defers."""
    mw = StructuredOutputMiddleware(_Recipe)
    msg = AIMessage(content="   ")
    result = await mw.aafter_model(_state([msg]), runtime=None)
    assert result is None


@pytest.mark.asyncio
async def test_middleware_passes_through_on_non_ai_message() -> None:
    mw = StructuredOutputMiddleware(_Recipe)
    msg = HumanMessage(content='<output_schema>{"x": 1}</output_schema>')
    result = await mw.aafter_model(_state([msg]), runtime=None)
    assert result is None


@pytest.mark.asyncio
async def test_middleware_passes_through_when_messages_empty() -> None:
    mw = StructuredOutputMiddleware(_Recipe)
    result = await mw.aafter_model(_state([]), runtime=None)
    assert result is None


@pytest.mark.asyncio
async def test_middleware_returns_none_on_valid_block() -> None:
    mw = StructuredOutputMiddleware(_Recipe)
    payload = {"title": "tacos", "ingredients": ["beef", "corn"], "minutes": 30}
    msg = AIMessage(
        content=f"answer: <output_schema>{json.dumps(payload)}</output_schema>"
    )
    result = await mw.aafter_model(_state([msg]), runtime=None)
    assert result is None


@pytest.mark.asyncio
async def test_middleware_nudges_when_block_missing() -> None:
    mw = StructuredOutputMiddleware(_Recipe, max_nudges=2)
    msg = AIMessage(content="here is my answer with no schema block")
    result = await mw.aafter_model(_state([msg]), runtime=None)
    assert result is not None
    nudge = result["messages"][-1]
    assert isinstance(nudge, HumanMessage)
    assert "_Recipe" in str(nudge.content) or "Recipe" in str(nudge.content)
    assert "<output_schema>" in str(nudge.content)


@pytest.mark.asyncio
async def test_middleware_nudges_on_schema_mismatch() -> None:
    mw = StructuredOutputMiddleware(_Recipe, max_nudges=2)
    # Missing required field "minutes".
    msg = AIMessage(
        content='<output_schema>{"title": "x", "ingredients": ["a"]}</output_schema>'
    )
    result = await mw.aafter_model(_state([msg]), runtime=None)
    assert result is not None
    assert isinstance(result["messages"][-1], HumanMessage)


@pytest.mark.asyncio
async def test_middleware_gives_up_after_max_nudges() -> None:
    mw = StructuredOutputMiddleware(_Recipe, max_nudges=2)
    msg = AIMessage(content="no schema block here")

    # First two attempts produce nudges.
    r1 = await mw.aafter_model(_state([msg]), runtime=None)
    r2 = await mw.aafter_model(_state([msg]), runtime=None)
    assert r1 is not None
    assert r2 is not None
    assert isinstance(r1["messages"][-1], HumanMessage)
    assert isinstance(r2["messages"][-1], HumanMessage)

    # Third attempt → give-up message (an AIMessage explaining the failure)
    # rather than another HumanMessage retry.
    r3 = await mw.aafter_model(_state([msg]), runtime=None)
    assert r3 is not None
    final = r3["messages"][-1]
    assert isinstance(final, AIMessage)
    assert "validation" in str(final.content).lower()
    assert "_Recipe" in str(final.content) or "Recipe" in str(final.content)


@pytest.mark.asyncio
async def test_middleware_resets_counter_on_before_agent() -> None:
    """A reused middleware must not accumulate nudges across runs."""
    mw = StructuredOutputMiddleware(_Recipe, max_nudges=1)
    msg = AIMessage(content="no block")

    # Burn the budget in run 1.
    _ = await mw.aafter_model(_state([msg]), runtime=None)
    give_up = await mw.aafter_model(_state([msg]), runtime=None)
    assert give_up is not None
    assert isinstance(give_up["messages"][-1], AIMessage)

    # Reset for run 2.
    await mw.abefore_agent(_state([]), runtime=None)

    # Run 2 should nudge fresh (HumanMessage), not the give-up AIMessage.
    fresh = await mw.aafter_model(_state([msg]), runtime=None)
    assert fresh is not None
    assert isinstance(fresh["messages"][-1], HumanMessage)


@pytest.mark.asyncio
async def test_middleware_resets_counter_on_valid_response() -> None:
    """A valid response between failures resets the nudge counter."""
    mw = StructuredOutputMiddleware(_Recipe, max_nudges=1)
    bad = AIMessage(content="no block")
    good_payload = {"title": "x", "ingredients": ["a"], "minutes": 5}
    good = AIMessage(
        content=f"<output_schema>{json.dumps(good_payload)}</output_schema>"
    )

    # First failure: nudge. Counter = 1 (== max_nudges, not exceeded).
    r1 = await mw.aafter_model(_state([bad]), runtime=None)
    assert r1 is not None
    assert isinstance(r1["messages"][-1], HumanMessage)

    # Valid response resets counter.
    assert await mw.aafter_model(_state([good]), runtime=None) is None

    # Now a fresh failure should still get a nudge, not give-up.
    r2 = await mw.aafter_model(_state([bad]), runtime=None)
    assert r2 is not None
    assert isinstance(r2["messages"][-1], HumanMessage)
