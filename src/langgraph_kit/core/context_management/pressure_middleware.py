"""Deepagents middleware for context pressure monitoring and mitigation."""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
    SystemMessage,
)

from langgraph_kit.core.context_management.compaction import (
    CompactionMode,
    CompactionPromptPack,
    CompactionResult,
)
from langgraph_kit.core.context_management.pressure import (
    MitigationStrategy,
    PressureMonitor,
)
from langgraph_kit.core.internal_tags import (
    CONTEXT_COMPACTION_TAG,
    internal_llm_config,
)

logger = logging.getLogger(__name__)


class PressureMiddleware(AgentMiddleware):  # type: ignore[misc]
    """Deepagents middleware that checks context pressure before each agent turn.

    Returns state updates with compacted messages instead of mutating in place.
    """

    def __init__(
        self,
        monitor: PressureMonitor,
        *,
        llm: Any | None = None,
        compaction_tail_size: int = 6,
    ) -> None:
        super().__init__()
        self._monitor = monitor
        self._llm = llm
        self._tail_size = compaction_tail_size
        self._compaction_prompts = CompactionPromptPack()

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        """Check pressure and return compacted messages if needed."""
        state_dict: dict[str, Any] = dict(state) if isinstance(state, dict) else {}  # pyright: ignore[reportUnknownArgumentType]
        messages: list[Any] = state_dict.get("messages", [])
        if not messages:
            return None

        signals = self._monitor.assess(messages)
        strategy = self._monitor.choose_mitigation(signals)

        if strategy == MitigationStrategy.NONE:
            return None

        logger.info(
            "Context pressure: %.0f%% (%d tokens), strategy: %s",
            signals.pressure_pct * 100,
            signals.estimated_tokens,
            strategy.value,
        )

        if strategy == MitigationStrategy.MICROCOMPACT:
            compacted = self._microcompact(messages)
            if compacted is not None:
                return {"messages": compacted}

        if strategy == MitigationStrategy.FULL_COMPACTION:
            compacted = await self._full_compaction(messages)
            if compacted is not None:
                return {"messages": compacted}

        if strategy == MitigationStrategy.STOP:
            logger.warning("Context pressure critical — recommending stop")

        return None

    async def _full_compaction(self, messages: list[Any]) -> list[Any] | None:
        """Summarize the full conversation via the LLM and replace old messages.

        Returns the new message list, or None if compaction was skipped or failed
        (in which case the monitor's failure counter is incremented so repeated
        failures eventually escalate to STOP via the circuit breaker).
        """
        if self._llm is None:
            logger.warning(
                "FULL_COMPACTION selected but no LLM configured on PressureMiddleware"
            )
            return None

        if len(messages) <= self._tail_size:
            # Nothing to compact down to.
            return None

        prompt = self._compaction_prompts.build_prompt(CompactionMode.FULL)
        compaction_input = [
            SystemMessage(content=prompt),
            HumanMessage(content=_render_conversation(messages)),
        ]

        # Tag the call so consumers streaming via astream_events can filter
        # the compactor's chat_model events out of the user-facing transcript
        # — see langgraph_kit.core.internal_tags for rationale.
        try:
            response = await self._llm.ainvoke(
                compaction_input,
                config=internal_llm_config(
                    CONTEXT_COMPACTION_TAG, run_name="context_compaction"
                ),
            )
        except Exception:
            logger.warning("FULL_COMPACTION LLM call failed", exc_info=True)
            self._monitor.record_compaction_failure()
            return None

        raw: str = _extract_text(response)
        result: CompactionResult | None = self._compaction_prompts.parse_output(
            raw, mode=CompactionMode.FULL
        )
        if result is None:
            logger.warning("FULL_COMPACTION LLM output did not contain a valid summary")
            self._monitor.record_compaction_failure()
            return None

        summary_text = _format_summary(result)
        tail = messages[-self._tail_size :]
        new_messages: list[Any] = [SystemMessage(content=summary_text), *tail]

        self._monitor.record_compaction_success()
        logger.info(
            "FULL_COMPACTION replaced %d messages with summary + %d-msg tail",
            len(messages),
            len(tail),
        )
        return new_messages

    @staticmethod
    def _microcompact(messages: list[Any]) -> list[Any] | None:
        """Build a new message list with old large tool outputs truncated.

        Returns None if no changes were made.
        """
        if len(messages) <= 10:
            return None

        compact_boundary = len(messages) - 10
        # Check copy method once on first message (all messages share the same base class)
        use_model_copy = hasattr(messages[0], "model_copy")
        changed = False
        new_messages: list[Any] = []

        for i, msg in enumerate(messages):
            if i < compact_boundary:
                content = getattr(msg, "content", "")
                msg_type = getattr(msg, "type", "unknown")
                if (
                    isinstance(content, str)
                    and len(content) > 2000
                    and msg_type in ("tool", "function")
                ):
                    truncated = (
                        content[:200]
                        + f"\n...[truncated — original was {len(content):,} chars]"
                    )
                    if use_model_copy:
                        msg = msg.model_copy(update={"content": truncated})
                    else:
                        msg = msg.copy(update={"content": truncated})
                    changed = True
            new_messages.append(msg)

        return new_messages if changed else None


def _render_conversation(messages: list[Any]) -> str:
    """Serialize messages into a plain-text transcript for the compactor."""
    lines: list[str] = []
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:  # pyright: ignore[reportUnknownVariableType]
                if isinstance(part, dict):
                    parts.append(str(part.get("text", "")))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                else:
                    parts.append(str(part))
            content = "\n".join(parts)
        lines.append(f"[{role}] {content}")
    return "\n\n".join(lines)


def _extract_text(response: Any) -> str:
    """Pull plain text out of an LLM response (AIMessage or raw string)."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:  # pyright: ignore[reportUnknownVariableType]
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def _bullets(items: list[str]) -> str:
    """Render a list as an indented bullet block, or '(none)' if empty."""
    return "\n".join(f"  - {item}" for item in items) or "  (none)"


def _format_summary(result: CompactionResult) -> str:
    """Render a CompactionResult as a system-message summary block."""
    return (
        "# Conversation Summary (auto-compacted)\n"
        f"**User intent:** {result.user_intent}\n"
        f"**Current state:** {result.current_state}\n"
        f"**Next step:** {result.next_step}\n"
        f"**Key decisions:**\n{_bullets(result.key_decisions)}\n"
        f"**Important files:**\n{_bullets(result.important_files)}\n"
        f"**Errors & fixes:**\n{_bullets(result.errors_and_fixes)}\n"
        f"**Pending work:**\n{_bullets(result.pending_work)}\n"
    )
