"""PostRunBackstopMiddleware — safe deterministic cleanup after run completion.

Scoped narrowly to actions that are always safe: flushing observability,
persisting run metadata, and recording completion summaries. Does NOT
attempt to infer or auto-execute missing business-logic actions.
"""

from __future__ import annotations

import contextvars
import logging
import time
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware as _AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.config import get_config

from langgraph_kit.core.resilience._message_text import aimessage_text

logger = logging.getLogger(__name__)

RUN_METADATA_NAMESPACE = ("run_metadata",)

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

        # Persist to Store if available
        store = getattr(runtime, "store", None)
        if store is not None:
            config = get_config()
            thread_id = config.get("configurable", {}).get("thread_id", "unknown")
            key = f"{thread_id}_{summary['completed_at']}"

            try:
                await store.aput(RUN_METADATA_NAMESPACE, key, summary)
            except Exception:
                logger.warning("Failed to persist run metadata", exc_info=True)

        _run_start_var.set(None)
        return None


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
