"""Default audit stop hook for the reference deep agent.

Emits one ``AGENT_RUN_COMPLETE`` audit entry per turn via
:class:`~langgraph_kit.core.audit.AuditStore`. Metadata-only by
design — no message bodies, no tool arguments — so the audit log
captures *what happened* (turn boundaries, message counts, tool-call
counts) without leaking conversation contents.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from langgraph_kit.core.audit import AuditAction, AuditStore

logger = logging.getLogger(__name__)


class ReferenceAuditStopHook:
    """Stop hook that records agent-run completion audit events.

    Non-blocking: any failure to write to the audit store is logged
    and swallowed so audit infrastructure failures never break a turn.
    """

    blocking: bool = False

    def __init__(
        self,
        audit_store: AuditStore,
        *,
        agent_id: str = "reference-deep-agent",
    ) -> None:
        super().__init__()
        self._audit_store = audit_store
        self._agent_id = agent_id

    async def on_turn_complete(self, state: Any) -> None:
        messages: list[Any] = []
        if isinstance(state, dict):
            raw = state.get("messages")
            if isinstance(raw, list):
                messages = raw

        tool_call_count = 0
        last_kind = "none"
        if messages:
            last = messages[-1]
            last_kind = type(last).__name__
            if isinstance(last, AIMessage):
                tool_call_count = len(last.tool_calls or [])

        thread_id = _try_get_thread_id()

        # Metadata-only payload — no message bodies. Verified by the
        # PII test in the reference test suite.
        metadata: dict[str, Any] = {
            "message_count": len(messages),
            "last_message_kind": last_kind,
            "tool_calls_in_last_message": tool_call_count,
        }
        if thread_id:
            metadata["thread_id"] = thread_id

        target = f"thread:{thread_id}" if thread_id else "agent_run"
        await self._audit_store.write(
            actor=f"agent:{self._agent_id}",
            action=AuditAction.AGENT_RUN_COMPLETE,
            target=target,
            metadata=metadata,
        )


def _try_get_thread_id() -> str | None:
    """Best-effort lookup of the current thread_id from the LangGraph config.

    Returns ``None`` outside a LangGraph invocation context (e.g. in
    direct middleware unit tests) so the hook stays a no-op-friendly
    abstraction.
    """
    try:
        from langgraph.config import (  # pyright: ignore[reportMissingImports]
            get_config,
        )

        config = get_config()
    except Exception:
        return None
    if not isinstance(config, dict):
        return None
    configurable = config.get("configurable", {})
    if not isinstance(configurable, dict):
        return None
    thread_id = configurable.get("thread_id")
    return str(thread_id) if thread_id else None
