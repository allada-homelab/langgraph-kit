"""SSE streaming for LangGraph astream_events v2."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any

from langgraph_kit.core.artifacts import ARTIFACT_SENTINEL
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
# list into text content — as a trailing ```json\n[]\n``` (code-fenced).
# We buffer the tail of the stream and strip that specific shape.
# A *bare* trailing ``[]`` is NOT stripped because it's ambiguous — a
# legitimate response like "the function returned `[]`" would otherwise
# lose its payload. Only the fenced form is treated as a tool-call leak.
_TRAILING_ARTIFACT_RE = re.compile(r"\s*```(?:json)?\s*\[\s*\]\s*```\s*$")
_BUFFER_SIZE = 30  # characters — enough for "```json\n[]\n```"
_MAX_TOOL_OUTPUT_SSE = 3000  # truncate tool outputs longer than this in the SSE stream

# Default cadence for heartbeats during quiet periods. Many proxies
# (nginx default proxy_read_timeout = 60s, AWS NLB idle = 350s, Cloudflare
# idle = 100s) drop idle SSE connections; 15s leaves comfortable margin
# without flooding healthy streams with no-ops.
DEFAULT_HEARTBEAT_INTERVAL_SECONDS: float = 15.0


def _sse_chunk(seq: int, payload: dict[str, Any] | str) -> str:
    """Format an SSE chunk with an ``id:`` line and ``data:`` body.

    ``payload`` is JSON-encoded when given as a dict, or used verbatim
    when given as a string (e.g. the ``[DONE]`` sentinel). The ``id:``
    line is part of the SSE wire format and lets reconnecting clients
    tell the server (via the ``Last-Event-ID`` header) which events
    they have already seen — the durable replay log that consumes that
    id is a follow-up; for now the id makes the contract forward-
    compatible.
    """
    body = json.dumps(payload) if isinstance(payload, dict) else payload
    return f"id: {seq}\ndata: {body}\n\n"


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
    heartbeat_interval: float | None = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
) -> AsyncGenerator[str]:
    """Stream LangGraph events as SSE chunks.

    Event types emitted:
      - ``{"token": "..."}`` — text tokens from ``on_chat_model_stream``
      - ``{"artifact": {...}}`` — structured UI artifacts from ``create_artifact`` tool
      - ``{"tool_call_start": {...}}`` — tool invocation started
      - ``{"tool_call_end": {...}}`` — tool invocation completed
      - ``{"node_entered": {"id": ..., "name": ...}}`` — graph node
        starts executing (matches the node ids ``print_graph`` emits;
        the live-overlay viewer toggles ``.active`` styling on the
        node when this fires)
      - ``{"node_exited": {"id": ..., "name": ...}}`` — paired close
      - ``{"interrupt": {...}}`` — graph paused for human input
      - ``{"heartbeat": {"ts": ..., "last_event_id": ...}}`` — emitted
        every ``heartbeat_interval`` seconds during quiet periods so
        proxies / load balancers don't drop the idle connection
      - ``[DONE]`` — stream finished

    Every emitted chunk carries an SSE ``id:`` line with a per-stream
    monotonically-increasing sequence number. Clients can record the
    most recent id (e.g. via ``EventSource.lastEventId``) and pass it
    back as ``Last-Event-ID`` on reconnect; the durable replay log
    that honors that header is a follow-up — for now the id makes the
    contract forward-compatible.

    Parameters
    ----------
    heartbeat_interval:
        Seconds between heartbeat chunks during periods with no real
        events. ``None`` disables heartbeats. Defaults to
        :data:`DEFAULT_HEARTBEAT_INTERVAL_SECONDS` (15 s).

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
    stream_error: Exception | None = None
    # Per-stream monotonic sequence for SSE ``id:`` lines. Resets per
    # request because the durable replay log is not yet wired in;
    # clients should treat ids as unique within a single connection.
    seq = 0
    # Coalesce ``node_entered`` events: LangGraph fires ``on_chain_start``
    # multiple times per node when sub-channels fan in, but the live
    # graph overlay only cares about the *transition*. Track the
    # currently-entered node so duplicates are dropped. ``None`` means
    # "no node currently active" (boundary at session start + after each
    # ``node_exited``). See issue #86.
    last_entered_node: str | None = None

    try:
        # Accumulate text so we can detect trailing artifacts from models
        # that leak tool_calls into content (e.g. Qwen via vLLM).
        accumulated = ""
        yielded_up_to = 0

        # Defense-in-depth: pre-merge any config bound to the graph via
        # ``with_config`` so values like ``recursion_limit`` survive the
        # ``astream_events`` codepath. Graphs built via the kit's builders
        # already patch ``astream_events`` (see ``bind_kit_defaults``), but
        # this helper accepts arbitrary user-supplied graphs too. The
        # caller's config still wins because langgraph's ensure_config
        # applies configs in order with later entries overriding earlier.
        bound_config = getattr(graph, "config", None)
        if bound_config:
            from langgraph._internal._config import (  # pyright: ignore[reportMissingImports]
                ensure_config as _lg_ensure_config,
            )

            effective_config: Any = _lg_ensure_config(bound_config, config)  # pyright: ignore[reportArgumentType]
        else:
            effective_config = config

        try:
            event_iter = graph.astream_events(
                input_data, config=effective_config, version="v2"
            )
        except Exception as exc:  # pragma: no cover — construction-time failure
            logger.exception("astream_events construction raised")
            stream_error = exc
            event_iter = _empty_aiter()

        # Producer-consumer split so heartbeats can race against the
        # next-event fetch without cancelling the upstream iterator
        # (cancelling an async generator mid-iteration is unsafe — the
        # iterator may have buffered state that the cancel doesn't
        # restore). The producer drains _guarded_events into a small
        # queue; the consumer pulls with a timeout to emit heartbeats
        # during quiet periods.
        event_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
        _PRODUCER_DONE = object()  # local sentinel — None is a valid event payload

        async def _producer() -> None:
            try:
                async for ev in _guarded_events(event_iter):
                    await event_queue.put(ev)
            finally:
                await event_queue.put(_PRODUCER_DONE)

        producer_task = asyncio.create_task(_producer())

        try:
            while True:
                if heartbeat_interval is None:
                    event = await event_queue.get()
                else:
                    try:
                        event = await asyncio.wait_for(
                            event_queue.get(), timeout=heartbeat_interval
                        )
                    except TimeoutError:
                        # Quiet period — emit a heartbeat so proxies don't
                        # drop the connection. Carries the most recent
                        # event id so a future replay layer (or a savvy
                        # client) knows where to resume from.
                        yield _sse_chunk(
                            seq,
                            {
                                "heartbeat": {
                                    "ts": time.time(),
                                    "last_event_id": seq - 1,
                                }
                            },
                        )
                        seq += 1
                        continue

                if event is _PRODUCER_DONE:
                    break
                if isinstance(event, _StreamError):
                    stream_error = event.exc
                    break

                # Drop events from kit-internal LLM calls (memory extraction,
                # consolidation, context compaction, routing). Without this
                # filter, their tokens and tool-call metadata leak into the
                # user-facing transcript — see langgraph_kit.core.internal_tags.
                if INTERNAL_TAG in (event.get("tags") or ()):
                    continue

                kind = event["event"]

                # --- Graph node enter/exit (live overlay for print_graph, #86) ---
                # LangGraph fires ``on_chain_start`` / ``on_chain_end`` for
                # every Runnable in the call tree — most of those aren't
                # graph-level nodes (the kit's middleware, deepagents'
                # ``write_todos`` wrapper, langchain's RunnablePassthrough,
                # etc.). Filter to events where ``metadata.langgraph_node``
                # is set — that key is LangGraph's own marker for
                # "this is one of the graph's declared nodes," matching the
                # ids ``print_graph`` emits in its Mermaid output.
                if kind in ("on_chain_start", "on_chain_end"):
                    metadata = event.get("metadata") or {}
                    node_name = metadata.get("langgraph_node")
                    if node_name:
                        run_id = event.get("run_id", "")
                        if kind == "on_chain_start":
                            # Coalesce: don't refire ``node_entered`` for the
                            # same node back-to-back. LangGraph sometimes
                            # emits multiple start events for a single node
                            # (parallel sub-channel fan-in); the overlay
                            # only cares about the transition.
                            if last_entered_node == node_name:
                                continue
                            last_entered_node = node_name
                            yield _sse_chunk(
                                seq,
                                {
                                    "node_entered": {
                                        "id": run_id,
                                        "name": node_name,
                                    }
                                },
                            )
                            seq += 1
                        else:  # on_chain_end
                            if last_entered_node == node_name:
                                last_entered_node = None
                            yield _sse_chunk(
                                seq,
                                {
                                    "node_exited": {
                                        "id": run_id,
                                        "name": node_name,
                                    }
                                },
                            )
                            seq += 1
                    # Either way, fall through — the rest of the chain-event
                    # surface (per-node token streams, tool calls) is
                    # handled below; this block is purely additive overlay.

                # --- Tool call start ---
                if kind == "on_tool_start":
                    tool_input = event["data"].get("input", {})
                    name = event.get("name", "unknown")
                    run_id = event.get("run_id", "")
                    yield _sse_chunk(
                        seq,
                        {
                            "tool_call_start": {
                                "id": run_id,
                                "name": name,
                                "args": tool_input,
                            }
                        },
                    )
                    seq += 1
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
                        yield _sse_chunk(seq, sentinel_event)
                        seq += 1
                    else:
                        # Normal tool output
                        output_str = (
                            str(output) if not isinstance(output, str) else output
                        )
                        # Truncate very long outputs for the SSE stream
                        if len(output_str) > _MAX_TOOL_OUTPUT_SSE:
                            output_str = (
                                output_str[:_MAX_TOOL_OUTPUT_SSE] + "...(truncated)"
                            )
                        yield _sse_chunk(
                            seq,
                            {
                                "tool_call_end": {
                                    "id": run_id,
                                    "name": name,
                                    "output": output_str,
                                }
                            },
                        )
                        seq += 1
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
                    yield _sse_chunk(
                        seq, {"token": accumulated[yielded_up_to:safe_end]}
                    )
                    seq += 1
                    yielded_up_to = safe_end
        finally:
            producer_task.cancel()
            # Drain any queued event so the producer can finish cleanly.
            await asyncio.gather(producer_task, return_exceptions=True)

        # --- Flush remaining buffered text, stripping trailing artifacts ---
        remaining = accumulated[yielded_up_to:]
        remaining = _TRAILING_ARTIFACT_RE.sub("", remaining)
        if remaining:
            yield _sse_chunk(seq, {"token": remaining})
            seq += 1

        # --- Check final state for command results and interrupts ---
        try:
            state = await graph.aget_state(config)

            # If no LLM tokens were streamed, the last message may be a
            # command result injected by middleware. Gate the emission on
            # the ``COMMAND_RESULT_MARKER`` sentinel that CommandMiddleware
            # attaches to AIMessages it creates — otherwise a silent
            # tool-only run would get its last AIMessage mis-reported
            # here as a command result.
            if not started_text and state and state.values:
                msgs = state.values.get("messages", [])
                if msgs:
                    from langgraph_kit.core.commands.middleware import (
                        COMMAND_RESULT_MARKER,
                    )

                    last = msgs[-1]
                    additional = getattr(last, "additional_kwargs", None) or {}
                    is_command = bool(additional.get(COMMAND_RESULT_MARKER))
                    if (
                        is_command
                        and getattr(last, "type", "") == "ai"
                        and getattr(last, "content", "")
                    ):
                        yield _sse_chunk(
                            seq, {"command_result": {"output": last.content}}
                        )
                        seq += 1

            if hasattr(state, "tasks") and state.tasks:
                for task in state.tasks:
                    for intr in getattr(task, "interrupts", []):
                        value = intr.value if hasattr(intr, "value") else intr
                        yield _sse_chunk(seq, {"interrupt": value})
                        seq += 1
        except Exception:
            logger.debug("Could not check interrupt state", exc_info=True)

        # Emit trace summary if trace export is active
        trace_handler = config.get("metadata", {}).get("_trace_handler")
        if trace_handler is not None:
            trace = trace_handler.get_trace()
            yield _sse_chunk(
                seq,
                {
                    "trace": {
                        "trace_id": trace.trace_id,
                        "duration_ms": round(trace.duration_ms, 2),
                        "span_count": trace.span_count,
                    }
                },
            )
            seq += 1
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
                yield _sse_chunk(
                    seq,
                    {
                        "budget": {
                            "tokens_used": total.total_tokens,
                            "input_tokens": total.input_tokens,
                            "output_tokens": total.output_tokens,
                            "estimated_cost_usd": round(total.estimated_cost_usd, 6),
                        }
                    },
                )
                seq += 1

                # Persist accumulated usage into the BudgetManager so the
                # state survives the stream. Without this the callback
                # totals only ever flow to the SSE client — the Store
                # never sees them, and check_budget reads a stale 0-token
                # state on the next turn.
                if store is not None and thread_id:
                    try:
                        from langgraph_kit._config import get_config
                        from langgraph_kit.core.cost.budget import BudgetManager
                        from langgraph_kit.core.cost.models import BudgetConfig

                        app_cfg = get_config()
                        if app_cfg.token_budget_per_thread > 0:
                            manager = BudgetManager(
                                store,
                                BudgetConfig(
                                    max_tokens_per_thread=app_cfg.token_budget_per_thread
                                ),
                            )
                            user_id = config.get("metadata", {}).get("user_id") or ""
                            await manager.record_usage(
                                thread_id, total, user_id=user_id
                            )
                    except Exception:
                        logger.debug("Failed to persist budget usage", exc_info=True)

        # Emit an explicit error event before closing the stream so the
        # client can distinguish "backend failure" from "run completed".
        if stream_error is not None:
            message = _format_stream_error(stream_error)
            yield _sse_chunk(seq, {"error": {"message": message}})
            seq += 1

        yield _sse_chunk(seq, "[DONE]")
        seq += 1
    finally:
        if tracker and thread_id:
            await tracker.mark_idle(thread_id)
        flush_langfuse()


class _StreamError:
    """Sentinel yielded by ``_guarded_events`` when iteration raises."""

    __slots__ = ("exc",)

    def __init__(self, exc: Exception) -> None:
        self.exc = exc


async def _guarded_events(iterator: Any) -> AsyncGenerator[Any]:
    """Forward events from ``iterator``; on exception, yield ``_StreamError``."""
    try:
        async for event in iterator:
            yield event
    except Exception as exc:
        logger.exception("astream_events iteration raised")
        yield _StreamError(exc)


async def _empty_aiter() -> AsyncGenerator[Any]:
    if False:
        yield None  # pragma: no cover
    return


def _format_stream_error(exc: Exception) -> str:
    """Short user-safe error message (exception type + first line)."""
    first_line = str(exc).splitlines()[0] if str(exc) else ""
    return f"{type(exc).__name__}: {first_line}" if first_line else type(exc).__name__
