"""Tool result persistence — offloads large outputs to Store and replaces with previews."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

logger = logging.getLogger(__name__)

# Results smaller than this stay inline (no persistence needed)
DEFAULT_PERSIST_THRESHOLD = 4000  # characters

# Preview: first N chars shown inline with the reference
DEFAULT_PREVIEW_LENGTH = 300


class ResultPersistenceMiddleware(AgentMiddleware):  # type: ignore[misc]
    """Persists large tool results to Store, replaces with compact preview + reference.

    When a tool returns a result larger than `persist_threshold` chars:
    1. Persists the full result to Store under namespace ("tool_results",)
    2. Replaces the inline result with a preview + retrieval reference
    3. The agent can later retrieve the full result using the retrieve tool
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

        try:
            namespace = ("tool_results",)
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

            # Build preview + reference
            preview = content[: self._preview_length]
            replacement = (
                f"{preview}\n\n"
                f"[Full result persisted — {len(content):,} chars — "
                f"ref: {result_key}]"
            )

            # Create new ToolMessage with replaced content
            if hasattr(result, "model_copy"):
                return result.model_copy(update={"content": replacement})
            return result

        except Exception:
            logger.exception("Failed to persist tool result, leaving inline")
            return result
