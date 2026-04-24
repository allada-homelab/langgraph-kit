"""Deepagents middleware for context pressure monitoring and mitigation."""

from __future__ import annotations

import io
import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES

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

# Switch to chunked (hierarchical) summarization when the rendered head would
# exceed this many characters. Keeps peak memory bounded to roughly one chunk
# plus the (small) list of per-chunk summaries — instead of holding the whole
# transcript in memory while the LLM call is pending. Default is high enough
# that a typical 128k-token window stays on the single-shot path; only outsized
# contexts (1M-token windows, unbounded tool outputs) pay for chunking.
_CHUNK_RENDER_THRESHOLD = 500_000
_CHUNK_MESSAGES = 40


class PressureMiddleware(AgentMiddleware):  # type: ignore[misc]
    """Deepagents middleware that checks context pressure before each agent turn.

    Returns state updates with compacted messages instead of mutating in place.
    Replacement is done via ``RemoveMessage(REMOVE_ALL_MESSAGES)`` so the intent
    is explicit to the reducer: if the state channel does NOT use ``add_messages``
    (or a compatible reducer), langgraph will surface a loud error instead of
    silently turning replacement into an append.
    """

    def __init__(
        self,
        monitor: PressureMonitor,
        *,
        llm: Any | None = None,
        compaction_tail_size: int = 6,
        partial_keep_size: int = 20,
        chunk_render_threshold: int = _CHUNK_RENDER_THRESHOLD,
        chunk_messages: int = _CHUNK_MESSAGES,
    ) -> None:
        super().__init__()
        self._monitor = monitor
        self._llm = llm
        self._tail_size = compaction_tail_size
        self._partial_keep_size = partial_keep_size
        self._chunk_render_threshold = chunk_render_threshold
        self._chunk_messages = chunk_messages
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

        if strategy == MitigationStrategy.PARTIAL_COMPACTION:
            compacted = await self._partial_compaction(messages)
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

        For very large conversations the head is summarized chunk-by-chunk
        (map-reduce) so we never hold the entire rendered transcript in memory
        alongside the LLM request.
        """
        if self._llm is None:
            logger.warning(
                "FULL_COMPACTION selected but no LLM configured on PressureMiddleware"
            )
            return None

        if len(messages) <= self._tail_size:
            # Nothing to compact down to.
            return None

        head = messages[: -self._tail_size]
        tail = list(messages[-self._tail_size :])

        result = await self._summarize_head(head, mode=CompactionMode.FULL)
        if result is None:
            return None

        summary_text = _format_summary(result)
        # RemoveMessage(REMOVE_ALL_MESSAGES) tells add_messages to drop everything
        # before the marker and keep everything after it. If the consumer uses a
        # non-standard reducer, this surfaces loudly instead of silently appending.
        new_messages: list[Any] = [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            SystemMessage(content=summary_text),
            *tail,
        ]

        self._monitor.record_compaction_success()
        logger.info(
            "FULL_COMPACTION replaced %d messages with summary + %d-msg tail",
            len(messages),
            len(tail),
        )
        return new_messages

    async def _partial_compaction(self, messages: list[Any]) -> list[Any] | None:
        """Summarize the older head of the conversation, keeping a larger tail.

        Cheaper than FULL_COMPACTION because it preserves more recent context
        verbatim. Useful at moderate pressure before things go critical.
        """
        if self._llm is None:
            logger.warning(
                "PARTIAL_COMPACTION selected but no LLM configured on PressureMiddleware"
            )
            return None

        if len(messages) <= self._partial_keep_size + 2:
            # Not enough head to bother summarizing.
            return None

        head = messages[: -self._partial_keep_size]
        tail = list(messages[-self._partial_keep_size :])

        result = await self._summarize_head(head, mode=CompactionMode.PARTIAL)
        if result is None:
            return None

        summary_text = _format_summary(result)
        new_messages: list[Any] = [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            SystemMessage(content=summary_text),
            *tail,
        ]

        self._monitor.record_compaction_success()
        logger.info(
            "PARTIAL_COMPACTION replaced %d head messages with summary + %d-msg tail",
            len(head),
            len(tail),
        )
        return new_messages

    async def _summarize_head(
        self, head: list[Any], *, mode: CompactionMode
    ) -> CompactionResult | None:
        """Summarize a list of messages, using chunked map-reduce for very large heads.

        Returns the parsed CompactionResult, or None if the LLM call or parse
        failed — callers should treat None as a failure and let the monitor's
        circuit breaker track it. Callers must have already verified that
        ``self._llm`` is not None.
        """
        if self._llm is None:
            raise RuntimeError("_summarize_* helpers require self._llm to be set")
        # Decide chunked vs. direct by estimating rendered size. Cheap — avoids
        # building the full rendered string twice.
        approx_chars = _approx_render_chars(head)
        if approx_chars <= self._chunk_render_threshold:
            return await self._summarize_direct(head, mode=mode)
        return await self._summarize_chunked(head, final_mode=mode)

    async def _summarize_direct(
        self, messages: list[Any], *, mode: CompactionMode
    ) -> CompactionResult | None:
        """Render the whole list at once and summarize in a single LLM call."""
        if self._llm is None:
            raise RuntimeError("_summarize_* helpers require self._llm to be set")
        prompt = self._compaction_prompts.build_prompt(mode)
        rendered = _render_conversation(messages)
        compaction_input = [
            SystemMessage(content=prompt),
            HumanMessage(content=rendered),
        ]

        try:
            response = await self._llm.ainvoke(
                compaction_input,
                config=internal_llm_config(
                    CONTEXT_COMPACTION_TAG, run_name="context_compaction"
                ),
            )
        except Exception:
            logger.warning("Compaction LLM call failed", exc_info=True)
            self._monitor.record_compaction_failure()
            return None

        raw: str = _extract_text(response)
        result = self._compaction_prompts.parse_output(raw, mode=mode)
        if result is None:
            logger.warning("Compaction LLM output did not contain a valid summary")
            self._monitor.record_compaction_failure()
            return None
        return result

    async def _summarize_chunked(
        self, messages: list[Any], *, final_mode: CompactionMode
    ) -> CompactionResult | None:
        """Map-reduce summarization for very large heads.

        Phase 1 (map): split messages into chunks of ``_chunk_messages`` and
        summarize each chunk with PARTIAL prompt.
        Phase 2 (reduce): combine the per-chunk summaries into a single final
        summary with ``final_mode`` prompt. The per-chunk summaries are small,
        so the reduce step doesn't reintroduce the memory pressure.
        """
        if self._llm is None:
            raise RuntimeError("_summarize_* helpers require self._llm to be set")
        chunk_size = max(1, self._chunk_messages)
        chunks = [
            messages[i : i + chunk_size] for i in range(0, len(messages), chunk_size)
        ]

        chunk_summaries: list[CompactionResult] = []
        for idx, chunk in enumerate(chunks):
            logger.debug(
                "Chunked compaction: summarizing chunk %d/%d (%d msgs)",
                idx + 1,
                len(chunks),
                len(chunk),
            )
            piece = await self._summarize_direct(chunk, mode=CompactionMode.PARTIAL)
            if piece is None:
                return None  # failure already recorded by _summarize_direct
            chunk_summaries.append(piece)

        # Phase 2: feed the per-chunk summaries back into an LLM call as text.
        combined_text = _render_chunk_summaries(chunk_summaries)
        prompt = self._compaction_prompts.build_prompt(final_mode)
        try:
            response = await self._llm.ainvoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(content=combined_text),
                ],
                config=internal_llm_config(
                    CONTEXT_COMPACTION_TAG, run_name="context_compaction"
                ),
            )
        except Exception:
            logger.warning("Chunked compaction reduce step failed", exc_info=True)
            self._monitor.record_compaction_failure()
            return None

        raw: str = _extract_text(response)
        result = self._compaction_prompts.parse_output(raw, mode=final_mode)
        if result is None:
            logger.warning(
                "Chunked compaction reduce step did not produce a valid summary"
            )
            self._monitor.record_compaction_failure()
            return None
        return result

    @staticmethod
    def _microcompact(messages: list[Any]) -> list[Any] | None:
        """Thin alias — delegates to the module-level :func:`microcompact`."""
        return microcompact(messages)


# Module-level thresholds mirror what builtins previously used. Sharing a
# single implementation removes the drift hazard of having two copies
# with slightly different constants.
MICROCOMPACT_RECENT_WINDOW = 10
MICROCOMPACT_CONTENT_THRESHOLD = 2000
MICROCOMPACT_PREVIEW_CHARS = 200


def microcompact(
    messages: list[Any],
    *,
    recent_window: int = MICROCOMPACT_RECENT_WINDOW,
    content_threshold: int = MICROCOMPACT_CONTENT_THRESHOLD,
    preview_chars: int = MICROCOMPACT_PREVIEW_CHARS,
) -> list[Any] | None:
    """Truncate large tool outputs outside the recent message window.

    Returns a new message list with truncations applied, or ``None`` when
    no changes were needed. Both ``PressureMiddleware`` and the
    ``/compact`` command call this — keeping the logic in one place
    prevents the two call sites from drifting apart.
    """
    if len(messages) <= recent_window:
        return None

    compact_boundary = len(messages) - recent_window
    use_model_copy = hasattr(messages[0], "model_copy")
    changed = False
    new_messages: list[Any] = []

    for i, msg in enumerate(messages):
        if i < compact_boundary:
            content = getattr(msg, "content", "")
            msg_type = getattr(msg, "type", "unknown")
            if (
                isinstance(content, str)
                and len(content) > content_threshold
                and msg_type in ("tool", "function")
            ):
                truncated = (
                    content[:preview_chars]
                    + f"\n...[truncated — original was {len(content):,} chars]"
                )
                msg = (
                    msg.model_copy(update={"content": truncated})
                    if use_model_copy
                    else msg.copy(update={"content": truncated})
                )
                changed = True
        new_messages.append(msg)

    return new_messages if changed else None


def _render_conversation(messages: list[Any]) -> str:
    """Serialize messages into a plain-text transcript for the compactor.

    Uses an incremental StringIO buffer so we never hold both the list of
    per-line strings AND the final joined string in memory at the same time.
    """
    buf = io.StringIO()
    for i, msg in enumerate(messages):
        if i > 0:
            buf.write("\n\n")
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            flat_parts: list[str] = []
            for part in content:  # pyright: ignore[reportUnknownVariableType]
                if isinstance(part, dict):
                    flat_parts.append(str(part.get("text", "")))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                else:
                    flat_parts.append(str(part))
            content = "\n".join(flat_parts)
        buf.write(f"[{role}] {content}")
    return buf.getvalue()


def _approx_render_chars(messages: list[Any]) -> int:
    """Cheap upper bound on the rendered transcript size, without rendering.

    Sums raw content lengths; ignores the small per-message role prefix. Used
    to decide between direct and chunked summarization.
    """
    total = 0
    for msg in messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:  # pyright: ignore[reportUnknownVariableType]
                if isinstance(part, dict):
                    total += len(str(part.get("text", "")))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                else:
                    total += len(str(part))
    return total


def _render_chunk_summaries(results: list[CompactionResult]) -> str:
    """Render per-chunk summaries as input for the reduce step."""
    buf = io.StringIO()
    for i, r in enumerate(results):
        if i > 0:
            buf.write("\n\n")
        buf.write(f"## Chunk {i + 1} summary\n")
        buf.write(_format_summary(r))
    return buf.getvalue()


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
