"""PostRunBackstopMiddleware — safe deterministic cleanup after run completion.

Scoped narrowly to actions that are always safe: flushing observability,
persisting run metadata, and recording completion summaries. Does NOT
attempt to infer or auto-execute missing business-logic actions.
"""

from __future__ import annotations

import contextvars
import logging
import time
import uuid
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware as _AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.config import get_config

from langgraph_kit.core.resilience._message_text import aimessage_text

logger = logging.getLogger(__name__)

RUN_METADATA_NAMESPACE = ("run_metadata",)

# Cap retained run records per thread. The namespace is flat (no
# per-thread sub-namespace), so pruning walks the lot and keeps only
# the newest N per thread_id. Without this the kit wrote unbounded
# metadata into the Store — no TTL, no ceiling.
_DEFAULT_MAX_RECORDS_PER_THREAD = 100

# Per-request start time via ContextVar so concurrent invocations don't
# overwrite each other's timestamps.
_run_start_var: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "post_run_start", default=None
)


class PostRunBackstopMiddleware(_AgentMiddleware):  # type: ignore[misc]
    """Runs deterministic cleanup after the agent completes.

    Records:
    - Run summary (message count, tool calls, errors, duration)
    - Completion metadata to Store (if available)

    Does NOT:
    - Infer missing actions
    - Execute high-stakes operations
    - Make business decisions the model didn't explicitly take
    """

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        _run_start_var.set(time.time())
        return None

    def __init__(
        self,
        *,
        max_records_per_thread: int = _DEFAULT_MAX_RECORDS_PER_THREAD,
    ) -> None:
        super().__init__()
        self._max_per_thread = max_records_per_thread

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        run_start = _run_start_var.get()
        duration = time.time() - run_start if run_start else 0.0

        summary = _build_run_summary(messages, duration)

        logger.info(
            "Run complete: %d messages, %d tool calls, %d errors, %.1fs",
            summary["message_count"],
            summary["tool_calls"],
            summary["tool_errors"],
            summary["duration_seconds"],
        )

        # Persist to Store if available. ``get_config()`` raises when
        # called outside a LangGraph invocation context, so it lives
        # inside the guard — this middleware is a backstop and must not
        # crash the agent loop even if the runtime environment is
        # partially wired (e.g. unit tests calling aafter_agent directly).
        store = getattr(runtime, "store", None)
        if store is not None:
            try:
                config = get_config()
                thread_id = config.get("configurable", {}).get("thread_id", "unknown")
                summary["thread_id"] = thread_id
                # Unique key: thread + microseconds + uuid avoids collisions
                # when two runs finish in the same clock second. Earlier
                # keys were `{thread}_{epoch_seconds}` which collided under
                # busy traffic.
                now_us = time.time()
                key = f"{thread_id}_{now_us:.6f}_{uuid.uuid4().hex[:8]}"
                await store.aput(RUN_METADATA_NAMESPACE, key, summary)
                await _prune_thread_records(store, thread_id, self._max_per_thread)
            except Exception:
                logger.warning("Failed to persist run metadata", exc_info=True)

        _run_start_var.set(None)
        return None


async def _prune_thread_records(store: Any, thread_id: str, max_records: int) -> None:
    """Keep only the ``max_records`` newest records for ``thread_id``.

    The namespace is flat, so pruning walks all items up to a generous
    cap and filters by the thread_id prefix on the key.
    """
    try:
        items = await store.asearch(RUN_METADATA_NAMESPACE, limit=2000)
    except Exception:
        logger.debug("Failed to asearch for prune", exc_info=True)
        return

    owned = [i for i in items if i.key.startswith(f"{thread_id}_")]
    if len(owned) <= max_records:
        return

    owned.sort(key=lambda i: i.key)  # lexical sort == time order per key shape
    to_remove = owned[: len(owned) - max_records]
    for item in to_remove:
        try:
            await store.adelete(RUN_METADATA_NAMESPACE, item.key)
        except Exception:
            logger.debug("Prune delete failed", exc_info=True)


def _build_run_summary(messages: list[Any], duration: float) -> dict[str, Any]:
    """Build a structured summary of the completed run."""
    tool_calls = 0
    tool_errors = 0
    ai_messages = 0
    last_content = ""

    for msg in messages:
        if isinstance(msg, AIMessage):
            ai_messages += 1
            if msg.tool_calls:
                tool_calls += len(msg.tool_calls)
            msg_content = aimessage_text(msg)
            if msg_content.strip():
                last_content = msg_content.strip()[:200]
        elif isinstance(msg, ToolMessage):
            if getattr(msg, "status", None) == "error":
                tool_errors += 1

    return {
        "message_count": len(messages),
        "ai_messages": ai_messages,
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "duration_seconds": round(duration, 2),
        "completed_at": str(int(time.time())),
        "last_response_preview": last_content,
    }
