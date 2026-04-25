"""``PromptInjectionGuardMiddleware`` — inbound message scanner.

Runs on the most recent ``HumanMessage`` in graph state and looks for
prompt-injection phrasings (regex first; an optional classifier hook
catches paraphrases).

Two modes:

- ``"warn"`` (default): logs the match at WARNING and tags the
  message's ``additional_kwargs`` with :data:`PROMPT_INJECTION_FLAG`
  so downstream layers (audit log, observability, future quarantine
  routing) can react. Does not change the agent's tool surface or
  refuse the message — false positives on legitimate user input are
  worse than the attack succeeding once.
- ``"off"``: middleware is a no-op (no scan, no flag).

The richer ``"quarantine"`` mode (narrow tool surface for the affected
turn) waits on the audit-log infrastructure tracked in #24 and the
flag-driven tool-binding plumbing it requires; it'll layer on top of
this scanner without breaking existing callers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage

from .injection_patterns import (
    INJECTION_PATTERNS,
    InjectionMatch,
    scan_for_injection,
)

logger = logging.getLogger(__name__)

# Key set on AIMessage.additional_kwargs / HumanMessage.additional_kwargs
# when a scan matched. Value is a list of pattern names that hit, so
# downstream consumers can render or audit specifics. Stored under a
# kit-specific prefix to avoid colliding with any vendor-defined keys.
PROMPT_INJECTION_FLAG: str = "_lgk_prompt_injection_match"


GuardMode = Literal["off", "warn"]


class PromptInjectionGuardMiddleware(AgentMiddleware):  # type: ignore[misc]
    """Scan inbound user messages for prompt-injection patterns.

    The middleware looks at the most recent ``HumanMessage`` in graph
    state on each turn. When a pattern hits, it logs at WARNING and
    annotates the message's ``additional_kwargs`` with the matched
    pattern names — observability and audit layers can react without
    needing their own scanner.
    """

    def __init__(
        self,
        *,
        mode: GuardMode = "warn",
        patterns: list[Any] | None = None,
        classifier: Callable[[str], float] | None = None,
        classifier_threshold: float = 0.7,
    ) -> None:
        super().__init__()
        self._mode = mode
        self._patterns = patterns if patterns is not None else INJECTION_PATTERNS
        self._classifier = classifier
        self._threshold = classifier_threshold

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        if self._mode == "off":
            return None

        messages = state.get("messages", [])
        if not messages:
            return None
        last_user = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)),
            None,
        )
        if last_user is None:
            return None

        # Skip messages we have already flagged in a prior turn — the
        # middleware is per-turn idempotent.
        existing = (last_user.additional_kwargs or {}).get(PROMPT_INJECTION_FLAG)
        if existing:
            return None

        text = (
            last_user.content
            if isinstance(last_user.content, str)
            else str(last_user.content)
        )
        matches: list[InjectionMatch] = scan_for_injection(
            text, patterns=self._patterns
        )

        # Optional classifier — only invoked when pattern miss leaves
        # an unflagged message. Cheap to short-circuit when it already
        # matched, and lets users keep the regex as fast-path.
        if not matches and self._classifier is not None:
            try:
                score = float(self._classifier(text))
            except Exception:
                logger.exception("Prompt-injection classifier raised; ignoring score")
                score = 0.0
            if score >= self._threshold:
                matches.append(
                    InjectionMatch(
                        pattern=f"classifier({score:.2f})",
                        match=text[:120],
                        span=(0, min(len(text), 120)),
                    )
                )

        if not matches:
            return None

        pattern_names = [m.pattern for m in matches]
        logger.warning(
            "PromptInjectionGuard: matched patterns=%s on message_id=%s",
            pattern_names,
            getattr(last_user, "id", "<unknown>"),
        )

        # Mutate additional_kwargs in place so the audit + future
        # quarantine layers can read it without rebuilding the
        # message object. additional_kwargs is a dict on every
        # langchain Message; setting it is the standard extension
        # path.
        merged_kwargs = dict(last_user.additional_kwargs or {})
        merged_kwargs[PROMPT_INJECTION_FLAG] = pattern_names
        last_user.additional_kwargs = merged_kwargs

        return None
