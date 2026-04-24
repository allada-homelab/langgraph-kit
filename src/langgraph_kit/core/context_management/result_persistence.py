"""Tool result persistence — offloads large outputs to Store and replaces with previews."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

from langgraph_kit.core.tools.result_retrieval import tool_results_namespace

logger = logging.getLogger(__name__)

# Results smaller than this stay inline (no persistence needed)
DEFAULT_PERSIST_THRESHOLD = 4000  # characters

# Preview: first N chars shown inline with the reference
DEFAULT_PREVIEW_LENGTH = 300


class ResultPersistenceMiddleware(AgentMiddleware):  # type: ignore[misc]
    """Persists large tool results to Store, replaces with compact preview + reference.

    When a tool returns a result larger than `persist_threshold` chars:
    1. Persists the full result to Store under ("tool_results", thread_id)
    2. Replaces the inline result with a preview + retrieval reference
    3. The agent can later retrieve the full result using the retrieve tool

    Namespacing by thread_id prevents cross-thread reads of persisted refs.
    """

    def __init__(
        self,
        persist_threshold: int = DEFAULT_PERSIST_THRESHOLD,
        preview_length: int = DEFAULT_PREVIEW_LENGTH,
    ) -> None:
        super().__init__()
        self._threshold = persist_threshold
        self._preview_length = preview_length

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        """Execute tool, then persist and replace if result is large."""
        result = await handler(request)

        # Only process ToolMessage results with string content
        content = getattr(result, "content", None)
        if not isinstance(content, str) or len(content) <= self._threshold:
            return result

        # Persist to store
        store = getattr(getattr(request, "runtime", None), "store", None)
        if store is None:
            return result  # No store available, leave inline

        tool_name = request.tool_call.get("name", "unknown")
        tool_call_id = request.tool_call.get("id", "")
        result_key = hashlib.sha256(f"{tool_call_id}:{tool_name}".encode()).hexdigest()[
            :16
        ]

        # Scope results by thread_id so cross-thread reads are impossible.
        runtime = getattr(request, "runtime", None)
        cfg = getattr(runtime, "config", {}) or {}
        thread_id = (cfg.get("configurable") or {}).get("thread_id")
        if not thread_id:
            # Without a thread_id the namespace would collide across threads;
            # leave the content inline rather than writing to a shared bucket.
            logger.warning(
                "ResultPersistenceMiddleware skipped: no thread_id in runtime config"
            )
            return result

        namespace = tool_results_namespace(thread_id)
        try:
            await store.aput(
                namespace,
                result_key,
                {
                    "content": content,
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "char_count": len(content),
                },
            )
        except Exception:
            # Persistence is the pre-condition for replacing the inline
            # content — if the store write failed, leaving the full
            # content in the message is the *correct* fallback because
            # the agent can't retrieve what was never persisted.
            logger.exception("Failed to persist tool result, leaving inline")
            return result

        # Build preview + reference
        preview = content[: self._preview_length]
        replacement = (
            f"{preview}\n\n"
            f"[Full result persisted — {len(content):,} chars — "
            f"ref: {result_key}]"
        )

        # Create a replacement message with the same id (so ``add_messages``
        # replaces in place). Prefer ``model_copy`` to preserve every other
        # field; fall back to a fresh ToolMessage construction if the
        # message class doesn't support it — never return the original
        # large-content result, because the store already holds the canonical
        # copy and the agent would otherwise double-account for the bytes.
        if hasattr(result, "model_copy"):
            return result.model_copy(update={"content": replacement})

        try:
            from langchain_core.messages import (
                ToolMessage,  # pyright: ignore[reportMissingModuleSource]
            )

            return ToolMessage(
                content=replacement,
                tool_call_id=tool_call_id,
                name=tool_name,
                id=getattr(result, "id", None),
            )
        except Exception:
            # Last-resort: if we can't produce a replacement, roll back
            # the store write so we don't orphan bytes nothing can reach.
            logger.exception(
                "Could not build replacement ToolMessage; rolling back persistence"
            )
            try:
                await store.adelete(namespace, result_key)
            except Exception:
                logger.exception("Rollback delete also failed for key %s", result_key)
            return result
