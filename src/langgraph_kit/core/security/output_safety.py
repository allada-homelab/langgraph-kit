"""``OutputSafetyMiddleware`` — outbound assistant-message scanner + redactor.

Runs after every model turn. Inspects the most recent ``AIMessage``;
when its content contains anything matched by
:func:`scan_for_unsafe_output` the content is rewritten in place with
``[REDACTED]`` and the pattern names are stored on the message's
``additional_kwargs`` for audit.

Default mode is ``"redact"`` — the symmetric counterpart to the
inbound scanner's ``"warn"`` default. ``"warn"`` here means flag-only
(do not mutate the content); ``"off"`` disables.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage

from .output_patterns import (
    REDACTION_PLACEHOLDER,
    OutputMatch,
    redact,
    scan_for_unsafe_output,
)

logger = logging.getLogger(__name__)

# Stored on AIMessage.additional_kwargs whenever the middleware
# touched the message. Value is a list of pattern names so audit can
# describe what was redacted without re-leaking the matched content.
OUTPUT_SAFETY_FLAG: str = "_lgk_output_safety_match"


SafetyMode = Literal["off", "warn", "redact"]


class OutputSafetyMiddleware(AgentMiddleware):  # type: ignore[misc]
    """Scan outbound assistant messages and either flag or redact unsafe content.

    ``mode``:

    - ``"redact"`` (default): replaces every match with
      ``[REDACTED]`` and tags ``additional_kwargs`` with
      :data:`OUTPUT_SAFETY_FLAG` carrying the matched pattern names.
    - ``"warn"``: only tags. Useful in shadow-mode rollouts where
      operators want detection metrics before flipping to
      destructive redaction.
    - ``"off"``: skip scanning entirely.

    The middleware is per-turn idempotent — a message that already
    carries the flag is not re-scanned (avoids re-redacting a literal
    ``[REDACTED]`` placeholder).
    """

    def __init__(self, *, mode: SafetyMode = "redact") -> None:
        super().__init__()
        self._mode = mode

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        if self._mode == "off":
            return None

        messages = state.get("messages", [])
        if not messages:
            return None
        last = messages[-1]
        if not isinstance(last, AIMessage):
            return None

        existing = (last.additional_kwargs or {}).get(OUTPUT_SAFETY_FLAG)
        if existing:
            return None

        content = last.content
        if not isinstance(content, str) or not content:
            return None

        if self._mode == "warn":
            matches = scan_for_unsafe_output(content)
            if not matches:
                return None
            self._tag_message(last, matches)
            return None

        # mode == "redact"
        new_content, matches = redact(content)
        if not matches:
            return None

        last.content = new_content
        self._tag_message(last, matches)
        return None

    def _tag_message(self, msg: AIMessage, matches: list[OutputMatch]) -> None:
        pattern_names = sorted({m.pattern for m in matches})
        merged = dict(msg.additional_kwargs or {})
        merged[OUTPUT_SAFETY_FLAG] = pattern_names
        msg.additional_kwargs = merged
        logger.warning(
            "OutputSafety: %s patterns=%s on message_id=%s",
            "redacted" if self._mode == "redact" else "flagged",
            pattern_names,
            getattr(msg, "id", "<unknown>"),
        )


__all__ = [
    "OUTPUT_SAFETY_FLAG",
    "REDACTION_PLACEHOLDER",
    "OutputSafetyMiddleware",
    "SafetyMode",
]
