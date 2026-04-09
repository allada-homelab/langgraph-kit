"""ToolErrorMiddleware — catch tool exceptions and convert to structured error results.

Keeps the run alive so the model can recover from tool failures instead of
crashing the entire agent loop.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware as _AgentMiddleware
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)

# Exceptions that suggest the tool could succeed on retry (e.g. transient network)
_RETRYABLE_TYPES = frozenset(
    {
        "TimeoutError",
        "ConnectionError",
        "HTTPError",
        "RateLimitError",
        "ServiceUnavailableError",
    }
)


class ToolErrorMiddleware(_AgentMiddleware):  # type: ignore[misc]
    """Wraps every tool call — converts exceptions into structured ToolMessage errors.

    The model receives the error details and can decide to retry, try a
    different tool, or report the failure to the user. The run is never
    killed by a tool exception.
    """

    def __init__(self, *, max_retries: int = 1) -> None:
        super().__init__()
        self._max_retries = max_retries

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        tool_name = request.tool_call.get("name", "unknown")
        tool_call_id = request.tool_call.get("id", "")

        for attempt in range(1 + self._max_retries):
            try:
                return await handler(request)
            except Exception as exc:
                error_type = type(exc).__name__
                retryable = error_type in _RETRYABLE_TYPES

                if attempt < self._max_retries and retryable:
                    logger.warning(
                        "Tool '%s' failed (attempt %d/%d, retryable): %s",
                        tool_name,
                        attempt + 1,
                        1 + self._max_retries,
                        exc,
                    )
                    continue

                # Final failure — build structured error response
                logger.error(
                    "Tool '%s' failed permanently: %s", tool_name, exc, exc_info=True
                )

                tb = traceback.format_exception_only(type(exc), exc)
                error_detail = "".join(tb).strip()

                return ToolMessage(
                    content=(
                        f"Tool '{tool_name}' failed.\n"
                        f"error_type: {error_type}\n"
                        f"error: {error_detail}\n"
                        f"retryable: {retryable}\n\n"
                        "You can retry with different arguments, try a different "
                        "approach, or report this error to the user."
                    ),
                    tool_call_id=tool_call_id,
                    status="error",
                )

        # Unreachable — the loop always returns or continues
        return await handler(request)  # pragma: no cover
