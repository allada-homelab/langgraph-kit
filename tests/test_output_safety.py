"""Coverage — outbound PII / secret redaction added by issue #23.

`OutputSafetyMiddleware` scans the most recent `AIMessage` after every
turn. Default mode `"redact"` rewrites the content in place with
`[REDACTED]`; `"warn"` only flags via `additional_kwargs`; `"off"`
skips. The middleware is per-turn idempotent so a previously-tagged
message isn't re-scanned (avoids re-redacting the placeholder).
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_kit.core.security import (
    OUTPUT_SAFETY_FLAG,
    REDACTION_PLACEHOLDER,
    OutputMatch,
    OutputSafetyMiddleware,
    redact,
    scan_for_unsafe_output,
)

# ---------------------------------------------------------------------------
# Pattern scanner
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected_pattern"),
    [
        ("contact me at alice@example.com", "email_address"),
        ("call (555) 123-4567 today", "us_phone_number"),
        ("ssn 123-45-6789 is sensitive", "ssn_us"),
        ("api_key=sk-abc123", "api_key_kv"),
        ("Bearer eyJabcdef.ghijkl", "bearer_token"),
        ("ghp_" + "A" * 36, "github_pat"),
        ("AKIAIOSFODNN7EXAMPLE", "aws_access_key_id"),
        ("token: hunter2", "secret_kv"),
    ],
)
def test_scanner_matches_known_unsafe_substrings(
    text: str, expected_pattern: str
) -> None:
    matches = scan_for_unsafe_output(text)
    assert any(m.pattern == expected_pattern for m in matches), (
        f"missing {expected_pattern} for {text!r}: got {[m.pattern for m in matches]}"
    )


@pytest.mark.parametrize(
    "benign",
    [
        "",
        "Just plain prose with no secrets.",
        "Order #1234567 was shipped — that's a 7-digit ID, not a card.",
        "Versions like v1.2.3 should not match.",
    ],
)
def test_scanner_passes_benign_text(benign: str) -> None:
    assert scan_for_unsafe_output(benign) == []


def test_credit_card_luhn_filters_random_digit_runs() -> None:
    """A 16-digit Luhn-invalid number must not be flagged. A real
    Luhn-valid test number (4111 1111 1111 1111 — Visa test) is."""
    assert scan_for_unsafe_output("invoice 1234 5678 9012 3456 issued") == []
    matches = scan_for_unsafe_output("paid with 4111 1111 1111 1111 yesterday")
    assert any(m.pattern == "credit_card" for m in matches)


def test_redact_replaces_only_matched_substrings() -> None:
    text = "Reach out to alice@example.com for the api_key=sk-abc12345 — thanks!"
    redacted, matches = redact(text)
    assert REDACTION_PLACEHOLDER in redacted
    assert "alice@example.com" not in redacted
    assert "sk-abc12345" not in redacted
    # Surrounding text survives.
    assert "Reach out to" in redacted
    assert "thanks!" in redacted
    pattern_names = {m.pattern for m in matches}
    assert pattern_names == {"email_address", "api_key_kv"}


def test_redact_handles_overlapping_patterns() -> None:
    """Overlapping matches resolve to the longer one — no double redaction."""
    redacted, _ = redact("api_key=sk-abc12345xyz")
    # Both api_key_kv and openai_or_anthropic_api_key match the value
    # portion. Should produce one [REDACTED], not nested ones.
    assert redacted.count(REDACTION_PLACEHOLDER) == 1


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


def _state(messages: list[Any]) -> dict[str, Any]:
    return {"messages": list(messages)}


@pytest.mark.asyncio
async def test_off_mode_is_a_noop() -> None:
    mw = OutputSafetyMiddleware(mode="off")
    msg = AIMessage(content="email me at bob@example.com")
    await mw.aafter_model(_state([msg]), runtime=None)
    assert msg.content == "email me at bob@example.com"
    assert OUTPUT_SAFETY_FLAG not in (msg.additional_kwargs or {})


@pytest.mark.asyncio
async def test_warn_mode_flags_without_mutating_content() -> None:
    mw = OutputSafetyMiddleware(mode="warn")
    msg = AIMessage(content="email me at bob@example.com")
    await mw.aafter_model(_state([msg]), runtime=None)
    assert msg.content == "email me at bob@example.com"
    assert msg.additional_kwargs[OUTPUT_SAFETY_FLAG] == ["email_address"]


@pytest.mark.asyncio
async def test_redact_mode_rewrites_content_in_place() -> None:
    mw = OutputSafetyMiddleware(mode="redact")
    msg = AIMessage(content="email me at bob@example.com please")
    await mw.aafter_model(_state([msg]), runtime=None)
    assert "bob@example.com" not in msg.content
    assert REDACTION_PLACEHOLDER in msg.content
    assert msg.additional_kwargs[OUTPUT_SAFETY_FLAG] == ["email_address"]


@pytest.mark.asyncio
async def test_redact_leaves_clean_messages_unchanged() -> None:
    mw = OutputSafetyMiddleware(mode="redact")
    original = "Just plain prose, no secrets."
    msg = AIMessage(content=original)
    await mw.aafter_model(_state([msg]), runtime=None)
    assert msg.content == original
    assert OUTPUT_SAFETY_FLAG not in (msg.additional_kwargs or {})


@pytest.mark.asyncio
async def test_redact_is_idempotent_per_turn() -> None:
    """Running the middleware twice on a redacted message must not
    re-redact the placeholder or change the flag."""
    mw = OutputSafetyMiddleware(mode="redact")
    msg = AIMessage(content="contact alice@example.com")
    await mw.aafter_model(_state([msg]), runtime=None)
    first_flag = list(msg.additional_kwargs[OUTPUT_SAFETY_FLAG])
    first_content = msg.content

    await mw.aafter_model(_state([msg]), runtime=None)
    assert msg.content == first_content
    assert msg.additional_kwargs[OUTPUT_SAFETY_FLAG] == first_flag


@pytest.mark.asyncio
async def test_skip_when_last_message_is_not_ai() -> None:
    mw = OutputSafetyMiddleware(mode="redact")
    msg = HumanMessage(content="api_key=sk-12345")
    await mw.aafter_model(_state([msg]), runtime=None)
    assert msg.content == "api_key=sk-12345"


@pytest.mark.asyncio
async def test_skip_when_messages_empty() -> None:
    mw = OutputSafetyMiddleware(mode="redact")
    result = await mw.aafter_model(_state([]), runtime=None)
    assert result is None


@pytest.mark.asyncio
async def test_skip_when_ai_content_is_not_string() -> None:
    """Some chat models return list-of-content-blocks rather than a
    plain string. The middleware leaves those alone (the existing
    redaction path is string-only)."""
    mw = OutputSafetyMiddleware(mode="redact")
    msg = AIMessage(content=[{"type": "text", "text": "alice@example.com"}])  # pyright: ignore[reportArgumentType]
    await mw.aafter_model(_state([msg]), runtime=None)
    # No mutation, no flag — list-shaped content is out of scope for v1.
    assert OUTPUT_SAFETY_FLAG not in (msg.additional_kwargs or {})


def test_output_match_type_is_namedtuple() -> None:
    """OutputMatch should be a NamedTuple (so consumers can pattern-
    match by index in a tight loop without dict overhead)."""
    om = OutputMatch(pattern="email_address", match="x@y.z", span=(0, 5))
    pattern, match, span = om
    assert pattern == "email_address"
    assert match == "x@y.z"
    assert span == (0, 5)
