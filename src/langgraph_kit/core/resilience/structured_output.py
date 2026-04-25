"""StructuredOutputMiddleware — validate the agent's final message against a Pydantic schema.

Opt-in. When a schema is supplied, the middleware inspects the last
``AIMessage`` after every model call. If it carries a tool call or no
content, the middleware is a no-op (the agent is mid-flow). Otherwise
it scans for a single ``<output_schema>{...}</output_schema>`` JSON
block, parses it with the schema, and on failure injects a retry
nudge that includes the JSON-Schema rendering of the model so the
LLM has the contract in front of it.

The convention mirrors the existing :class:`CompactionResult`
prompt-and-parse pattern (XML-wrapped JSON), keeping the validation
provider-agnostic — no provider-side ``response_format`` plumbing.

There is no silent fallback to "free text is OK" — if the schema is
configured, an unvalidated terminal message is treated as a failure
and triggers a nudge, up to ``max_nudges``. After the cap is hit, the
middleware appends a single ``[System: structured-output validation
gave up after N attempts]`` AIMessage so callers see the final state
explicitly rather than silently passing through.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

from langchain.agents.middleware.types import AgentMiddleware as _AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from langgraph_kit.core.resilience._message_text import aimessage_text

_TBaseModel = TypeVar("_TBaseModel", bound=BaseModel)

logger = logging.getLogger(__name__)

# Single-block convention. Multiple <output_schema> blocks would be
# ambiguous — take the *first* and document it; users hitting that case
# should narrow their prompt.
_OUTPUT_BLOCK_RE = re.compile(
    r"<output_schema>\s*(.*?)\s*</output_schema>",
    re.DOTALL,
)


def format_schema_instruction(schema: type[BaseModel]) -> str:
    """Render schema instructions to splice into a system prompt.

    Returns a prompt-ready string the user can interpolate into their
    composed system prompt. The instruction tells the model to produce
    a single ``<output_schema>{...}</output_schema>`` block matching
    the schema. Free-form text outside the block is allowed (the model
    can reason or cite before the structured payload).
    """
    schema_json = schema.model_json_schema()
    return (
        "When you are ready to produce your final answer, include a single "
        "block of the form `<output_schema>{...}</output_schema>` whose "
        "contents are valid JSON conforming to this schema:\n\n"
        f"```json\n{json.dumps(schema_json, indent=2)}\n```\n\n"
        "Anything outside that block is for human-readable narration and is "
        "ignored by validators. Do not nest or repeat the block."
    )


def extract_structured_output(content: str) -> str | None:
    """Pull the JSON inside ``<output_schema>...</output_schema>`` if present."""
    match = _OUTPUT_BLOCK_RE.search(content)
    if match is None:
        return None
    return match.group(1).strip() or None


def parse_structured_output(
    content: str, schema: type[_TBaseModel]
) -> _TBaseModel | None:
    """Extract + parse the schema-conformant block. Returns None on any failure.

    Generic on the schema type so callers get back the concrete subclass
    (``Recipe`` rather than ``BaseModel``), which keeps attribute access
    type-checkable.
    """
    raw = extract_structured_output(content)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    try:
        return schema.model_validate(data)
    except ValidationError:
        return None


class StructuredOutputMiddleware(_AgentMiddleware):  # type: ignore[misc]
    """Validate the agent's terminal message against a Pydantic schema.

    Skips when the last message is not an ``AIMessage`` or when the
    message is mid-tool-call (the agent is still working). On a
    terminal message that doesn't carry a parseable
    ``<output_schema>`` block, injects a retry nudge with the schema
    rendered as JSON Schema. Cap-based: after ``max_nudges`` failures,
    appends a single explanatory message and returns control so the
    run can terminate without looping.
    """

    def __init__(
        self,
        schema: type[BaseModel],
        *,
        max_nudges: int = 2,
    ) -> None:
        super().__init__()
        self._schema = schema
        self._max_nudges = max_nudges
        self._nudge_count = 0

    async def abefore_agent(self, state: Any, runtime: Any) -> None:  # noqa: ARG002
        """Reset per-run nudge counter so the cap is per-invocation."""
        self._nudge_count = 0

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        messages = state.get("messages", [])
        if not messages:
            return None

        last = messages[-1]
        if not isinstance(last, AIMessage):
            return None

        # Mid-tool-call → not a terminal turn yet.
        if last.tool_calls:
            return None

        content = aimessage_text(last)
        if not content.strip():
            # An empty message will be handled by EmptyTurnMiddleware;
            # piling another nudge on top would race that one and
            # double-prompt the model.
            return None

        parsed = parse_structured_output(content, self._schema)
        if parsed is not None:
            self._nudge_count = 0
            logger.info(
                "Structured output validated against %s on attempt %d",
                self._schema.__name__,
                self._nudge_count + 1,
            )
            return None

        self._nudge_count += 1
        if self._nudge_count > self._max_nudges:
            logger.warning(
                "Structured-output validation failed after %d attempts; "
                "giving up rather than looping",
                self._max_nudges,
            )
            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"[System: structured-output validation against "
                            f"`{self._schema.__name__}` failed after "
                            f"{self._max_nudges} retries; surfacing the last "
                            f"unstructured response as-is.]"
                        )
                    )
                ]
            }

        logger.info(
            "Structured-output mismatch (nudge %d/%d); injecting retry with schema",
            self._nudge_count,
            self._max_nudges,
        )
        return {
            "messages": [
                HumanMessage(
                    content=(
                        "[System: your response did not contain a valid "
                        f"`<output_schema>` block matching the "
                        f"`{self._schema.__name__}` schema. Please respond "
                        "again with a single block of the form "
                        "`<output_schema>{...}</output_schema>` whose "
                        "contents conform to this schema:]\n\n"
                        + format_schema_instruction(self._schema)
                    )
                )
            ]
        }
