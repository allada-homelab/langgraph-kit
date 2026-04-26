"""Data models for conversation recording and replay."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class LLMInteraction(BaseModel):
    """A single LLM call/response pair."""

    kind: Literal["llm"] = "llm"
    sequence_num: int
    model: str = ""
    input_messages: list[dict[str, Any]] = Field(default_factory=list)
    output_message: dict[str, Any] = Field(default_factory=dict)
    token_usage: dict[str, Any] | None = None
    timestamp: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolInteraction(BaseModel):
    """A single tool call/response pair."""

    kind: Literal["tool"] = "tool"
    sequence_num: int
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_output: str = ""
    status: Literal["success", "error"] = "success"
    duration_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationRecording(BaseModel):
    """A complete recorded conversation with all LLM and tool interactions."""

    id: str = ""
    agent_id: str = ""
    thread_id: str = ""
    model: str = ""
    created_at: str = ""
    interactions: list[LLMInteraction | ToolInteraction] = Field(default_factory=list)
    user_messages: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def llm_interactions(self) -> list[LLMInteraction]:
        """Return only LLM interactions in order."""
        return [i for i in self.interactions if isinstance(i, LLMInteraction)]

    @property
    def tool_interactions(self) -> list[ToolInteraction]:
        """Return only tool interactions in order."""
        return [i for i in self.interactions if isinstance(i, ToolInteraction)]

    @property
    def tool_sequence(self) -> list[str]:
        """Return the ordered list of tool names called."""
        return [i.tool_name for i in self.tool_interactions]


class RecordingOverrides(BaseModel):
    """Per-index overrides applied at replay time without mutating the recording on disk.

    Lets a developer fork a recording's trajectory by replacing the
    recorded LLM ``output_message`` at specific indices — useful for
    "what if the model had said X here instead?" experiments without
    re-recording. Indices refer to the *LLM-call ordinal* across the
    recording (i.e. position in
    :pyattr:`ConversationRecording.llm_interactions`), not the raw
    ``interactions`` list which interleaves tool calls.

    Negative indices are accepted with Python slice semantics
    (``-1`` = last LLM interaction).

    The ``output_message`` shape mirrors what
    :class:`LLMInteraction.output_message` carries — a dict with at
    minimum ``content`` and optional ``tool_calls``. The dict is fed
    directly to ``AIMessage(content=..., tool_calls=...)`` at lookup
    time, so anything ``AIMessage`` accepts is valid here.

    Examples::

        # Fork: at LLM call 2, return a different final answer.
        overrides = RecordingOverrides(
            llm_outputs={2: {"content": "Sorry, I can't help with that."}}
        )

        # Fork: at the last LLM call, force a different tool to be called.
        overrides = RecordingOverrides(
            llm_outputs={
                -1: {
                    "content": "",
                    "tool_calls": [
                        {"name": "alt_tool", "args": {}, "id": "call_x"}
                    ],
                }
            }
        )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm_outputs: dict[int, dict[str, Any]] = Field(default_factory=dict)
    """Map of LLM-call ordinal to replacement ``output_message`` dict.

    Keys may be negative (Python slice semantics: ``-1`` = last). Keys
    that don't resolve to a real LLM interaction at lookup time are
    silently ignored — by design, so adding overrides for indices that
    may or may not be reached is safe (e.g. when iterating with
    ``stop_at`` truncating the run).
    """

    def resolve(self, total_llm_interactions: int) -> dict[int, dict[str, Any]]:
        """Return overrides keyed by *positive* LLM-call ordinals only.

        Normalizes negative keys into the [0, total) range and drops
        any key that falls outside it. Caller-facing keys are stable
        across recording lengths; resolved keys are what the player
        actually looks up.
        """
        resolved: dict[int, dict[str, Any]] = {}
        for raw_idx, payload in self.llm_outputs.items():
            idx = raw_idx if raw_idx >= 0 else total_llm_interactions + raw_idx
            if 0 <= idx < total_llm_interactions:
                resolved[idx] = payload
        return resolved
