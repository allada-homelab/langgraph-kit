"""SSE streaming for LangGraph astream_events v2."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from langgraph_kit.core.artifacts import (
    ARTIFACT_SENTINEL,
    init_artifact_queue,
)
from langgraph_kit.core.internal_tags import INTERNAL_TAG
from langgraph_kit.core.ui_events import (
    CITATION_SENTINEL,
    PROGRESS_SENTINEL,
    SUGGESTIONS_SENTINEL,
)
from langgraph_kit.observability import flush_langfuse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)

# Some models (notably Qwen via OpenAI-compatible APIs) leak the tool_calls
# list into text content — e.g. a trailing "[]" or "```json\n[]\n```" at the
# end of the response.  We buffer the tail of the stream and strip these
# artifacts before sending to the client.
_TRAILING_ARTIFACT_RE = re.compile(r"\s*(?:```(?:json)?\s*)?\[\s*\]\s*(?:```\s*)?$")
_BUFFER_SIZE = 30  # characters — enough for "```json\n[]\n```"
_MAX_TOOL_OUTPUT_SSE = 3000  # truncate tool outputs longer than this in the SSE stream

# Map sentinel prefixes to SSE event keys
_SENTINEL_MAP: dict[str, str] = {
    ARTIFACT_SENTINEL: "artifact",
    PROGRESS_SENTINEL: "progress",
    SUGGESTIONS_SENTINEL: "suggestions",
    CITATION_SENTINEL: "citation",
}


def _parse_sentinel(output: str) -> dict[str, Any] | None:
    """Check if a tool output starts with a known sentinel prefix.

    Returns the parsed SSE event dict, or None for normal output.
    """
    for prefix, event_key in _SENTINEL_MAP.items():
        if output.startswith(prefix):
            try:
                payload = json.loads(output[len(prefix) :])
                return {event_key: payload}
            except json.JSONDecodeError:
                logger.warning("Malformed %s JSON in tool output", event_key)
                return None
    return None


async def stream_agent_events(
    graph: Any,
    input_data: dict[str, Any],
    config: dict[str, Any],
    *,
    store: Any | None = None,
) -> AsyncGenerator[str]:
    """Stream LangGraph events as SSE chunks.

    Event types emitted:
      - ``{"token": "..."}`` — text tokens from ``on_chat_model_stream``
      - ``{"artifact": {...}}`` — structured UI artifacts from ``create_artifact`` tool
      - ``{"tool_call_start": {...}}`` — tool invocation started
      - ``{"tool_call_end": {...}}`` — tool invocation completed
      - ``{"interrupt": {...}}`` — graph paused for human input
      - ``[DONE]`` — stream finished

    If ``store`` is provided, marks the thread as busy for the duration
    of the run so that the queue endpoints can detect active threads.
    """
    thread_id = config.get("configurable", {}).get("thread_id")
    tracker = None

    if store is not None and thread_id:
        from langgraph_kit.core.orchestration.queue import (
            ThreadBusyTracker,
        )

        tracker = ThreadBusyTracker(store)
        await tracker.mark_busy(thread_id)

    started_text = False
    init_artifact_queue()

    try:
        # Accumulate text so we can detect trailing artifacts from models
        # that leak tool_calls into content (e.g. Qwen via vLLM).
        accumulated = ""
        yielded_up_to = 0

        async for event in graph.astream_events(
            input_data, config=config, version="v2"
        ):
            # Drop events from kit-internal LLM calls (memory extraction,
            # consolidation, context compaction, routing). Without this
            # filter, their tokens and tool-call metadata leak into the
            # user-facing transcript — see langgraph_kit.core.internal_tags.
            if INTERNAL_TAG in (event.get("tags") or ()):
                continue

            kind = event["event"]

            # --- Tool call start ---
            if kind == "on_tool_start":
                tool_input = event["data"].get("input", {})
                name = event.get("name", "unknown")
                run_id = event.get("run_id", "")
                yield f"data: {json.dumps({'tool_call_start': {'id': run_id, 'name': name, 'args': tool_input}})}\n\n"
                continue

            # --- Tool call end (artifacts + normal output) ---
            if kind == "on_tool_end":
                output = event["data"].get("output", "")
                name = event.get("name", "unknown")
                run_id = event.get("run_id", "")

                # Check for sentinel-prefixed tool outputs → emit as dedicated SSE events
                sentinel_event = (
                    _parse_sentinel(output) if isinstance(output, str) else None
                )
                if sentinel_event:
                    yield f"data: {json.dumps(sentinel_event)}\n\n"
                else:
                    # Normal tool output
                    output_str = str(output) if not isinstance(output, str) else output
                    # Truncate very long outputs for the SSE stream
                    if len(output_str) > _MAX_TOOL_OUTPUT_SSE:
                        output_str = (
                            output_str[:_MAX_TOOL_OUTPUT_SSE] + "...(truncated)"
                        )
                    yield f"data: {json.dumps({'tool_call_end': {'id': run_id, 'name': name, 'output': output_str}})}\n\n"
                continue

            # --- Text token events ---
            if kind != "on_chat_model_stream":
                continue
            chunk = event["data"].get("chunk")
            if chunk is None:
                continue
            # Skip tool call chunks — only yield text tokens
            if getattr(chunk, "tool_call_chunks", None):
                continue
            token = chunk.content
            if not isinstance(token, str) or not token:
                continue

            # Drop leading whitespace prefix to avoid blank chat bubble
            if not started_text:
                token = token.lstrip()
                if not token:
                    continue
                started_text = True

            accumulated += token

            # Yield text that's safely past the buffer window.
            # We hold back the tail so we can strip trailing artifacts
            # once the stream ends.
            safe_end = max(yielded_up_to, len(accumulated) - _BUFFER_SIZE)
            if safe_end > yielded_up_to:
                yield f"data: {json.dumps({'token': accumulated[yielded_up_to:safe_end]})}\n\n"
                yielded_up_to = safe_end

        # --- Flush remaining buffered text, stripping trailing artifacts ---
        remaining = accumulated[yielded_up_to:]
        remaining = _TRAILING_ARTIFACT_RE.sub("", remaining)
        if remaining:
            yield f"data: {json.dumps({'token': remaining})}\n\n"

        # --- Check final state for command results and interrupts ---
        try:
            state = await graph.aget_state(config)

            # If no LLM tokens were streamed, the last message may be a
            # command result injected by middleware.  Emit it as a dedicated
            # event so the frontend can render it as a status banner.
            if not started_text and state and state.values:
                msgs = state.values.get("messages", [])
                if msgs:
                    last = msgs[-1]
                    if getattr(last, "type", "") == "ai" and getattr(
                        last, "content", ""
                    ):
                        yield f"data: {json.dumps({'command_result': {'output': last.content}})}\n\n"

            if hasattr(state, "tasks") and state.tasks:
                for task in state.tasks:
                    for intr in getattr(task, "interrupts", []):
                        value = intr.value if hasattr(intr, "value") else intr
                        yield f"data: {json.dumps({'interrupt': value})}\n\n"
        except Exception:
            logger.debug("Could not check interrupt state", exc_info=True)

        # Emit trace summary if trace export is active
        trace_handler = config.get("metadata", {}).get("_trace_handler")
        if trace_handler is not None:
            trace = trace_handler.get_trace()
            yield f"data: {json.dumps({'trace': {'trace_id': trace.trace_id, 'duration_ms': round(trace.duration_ms, 2), 'span_count': trace.span_count}})}\n\n"
            # Save trace to store if available
            if store is not None:
                try:
                    from langgraph_kit.core.tracing.storage import TraceStore

                    trace_store = TraceStore(store)
                    await trace_store.save_trace(trace.thread_id, trace)
                except Exception:
                    logger.debug("Failed to persist trace", exc_info=True)

        # Emit budget summary if token tracking is active
        budget_callback = config.get("metadata", {}).get("_budget_callback")
        if budget_callback is not None:
            usage_list = budget_callback.get_accumulated()
            if usage_list:
                total = budget_callback.get_total()
                yield f"data: {json.dumps({'budget': {'tokens_used': total.total_tokens, 'input_tokens': total.input_tokens, 'output_tokens': total.output_tokens, 'estimated_cost_usd': round(total.estimated_cost_usd, 6)}})}\n\n"

        yield "data: [DONE]\n\n"
    finally:
        if tracker and thread_id:
            await tracker.mark_idle(thread_id)
        flush_langfuse()
