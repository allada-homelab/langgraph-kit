"""Patterns for outbound safety: PII + credentials / secrets.

Combined surface used by ``OutputSafetyMiddleware`` to scan agent
responses before they reach the user. Each pattern carries a stable
name string so audit / observability layers can describe what was
redacted without exposing the raw match.

Two pattern families:

- **Secrets / credentials** — same shapes as the inbound
  guard from #5 (API keys, GitHub PATs, Bearer tokens, AWS keys, etc.).
  Kept here as well so the security module is self-contained — the
  memory module's private ``_SECRET_PATTERNS`` covers the same ground
  for memory publishing and is left untouched.
- **PII** — email, phone, SSN-shaped numbers, credit-card-shaped
  numbers. Conservative regexes; the false-positive bar matters here
  more than recall because every redaction mutates user-visible
  output.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Secrets / credentials
# ---------------------------------------------------------------------------

_API_KEY_KV = re.compile(r"(?:api[_-]?key|apikey)\s*[:=]\s*\S+", re.IGNORECASE)
_SECRET_KV = re.compile(
    r"(?:secret|token|password|passwd|pwd)\s*[:=]\s*\S+",
    re.IGNORECASE,
)
_BEARER_TOKEN = re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*")
_PRIVATE_KEY = re.compile(
    r"-----BEGIN\s+(?:RSA\s+|EC\s+|OPENSSH\s+)?PRIVATE\s+KEY-----"
)
_GITHUB_PAT = re.compile(r"\bgh[prost]_[A-Za-z0-9]{36,}\b")
_GITHUB_PAT_FINE = re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")
_OPENAI_KEY = re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{32,}\b")
_AWS_ACCESS_KEY = re.compile(r"\bAKIA[A-Z0-9]{16}\b")
_SLACK_TOKEN = re.compile(r"\bxox[bpras]-[A-Za-z0-9\-]{10,}\b")
_STRIPE_KEY = re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{20,}\b")
_GCP_SA = re.compile(r'"type"\s*:\s*"service_account"')

# ---------------------------------------------------------------------------
# PII
# ---------------------------------------------------------------------------

# RFC-5322-style email — pragmatic subset. Matches the vast majority of
# real-world emails without trying to be a full validator.
_EMAIL = re.compile(r"\b[\w.%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# US-style phone number with optional country / formatting. Avoids
# matching plain integers by requiring at least one separator.
_US_PHONE = re.compile(
    r"(?:(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4})\b",
)

# US Social Security number (NNN-NN-NNNN). Conservative: requires the
# dashes — bare 9-digit numbers are too noisy to flag.
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Credit-card-shaped 13-19 digit run with optional spaces or dashes.
# Pre-validation only; downstream consumers should run a Luhn check
# before treating it as a real card. Conservative: requires three or
# more groups of 4 to avoid matching arbitrary product / order numbers.
_CC_CANDIDATE = re.compile(
    r"\b(?:\d{4}[\s\-]){3,4}\d{1,4}\b",
)


def _luhn_ok(digits: str) -> bool:
    """Luhn check for a digits-only string. Used to suppress CC-shaped
    false positives — random 16-digit numbers fail Luhn."""
    if not digits:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        if not ch.isdigit():
            return False
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# Each entry: (compiled_pattern, kit-internal name, optional post-filter)
# The post-filter receives the matched substring and returns True to
# keep the match, False to drop. This is where Luhn lives so a regex
# can stay greedy without flooding output with random-number redactions.
_OUTBOUND_PATTERNS: list[tuple[re.Pattern[str], str, object]] = [
    (_API_KEY_KV, "api_key_kv", None),
    (_SECRET_KV, "secret_kv", None),
    (_BEARER_TOKEN, "bearer_token", None),
    (_PRIVATE_KEY, "private_key_block", None),
    (_GITHUB_PAT, "github_pat", None),
    (_GITHUB_PAT_FINE, "github_pat_fine_grained", None),
    (_OPENAI_KEY, "openai_or_anthropic_api_key", None),
    (_AWS_ACCESS_KEY, "aws_access_key_id", None),
    (_SLACK_TOKEN, "slack_token", None),
    (_STRIPE_KEY, "stripe_key", None),
    (_GCP_SA, "gcp_service_account_marker", None),
    (_EMAIL, "email_address", None),
    (_US_PHONE, "us_phone_number", None),
    (_SSN, "ssn_us", None),
    (
        _CC_CANDIDATE,
        "credit_card",
        # Drop the digits, run Luhn — keeps random 16-digit IDs from
        # being redacted.
        lambda match: _luhn_ok("".join(c for c in match if c.isdigit())),
    ),
]


REDACTION_PLACEHOLDER: str = "[REDACTED]"


class OutputMatch(NamedTuple):
    """A redaction-worthy substring located in an outbound message."""

    pattern: str
    match: str
    span: tuple[int, int]


def scan_for_unsafe_output(text: str) -> list[OutputMatch]:
    """Return every PII / secret pattern hit in *text*, in left-to-right order.

    Empty or non-string input yields an empty list — never raises.
    Overlapping matches keep the longer one.
    """
    if not text:
        return []
    candidates: list[OutputMatch] = []
    for pattern, name, post_filter in _OUTBOUND_PATTERNS:
        for m in pattern.finditer(text):
            substring = m.group(0)
            if callable(post_filter) and not post_filter(substring):
                continue
            candidates.append(
                OutputMatch(pattern=name, match=substring, span=(m.start(), m.end()))
            )

    # Resolve overlaps: prefer the longest match for any byte position.
    candidates.sort(key=lambda om: (om.span[0], -(om.span[1] - om.span[0])))
    selected: list[OutputMatch] = []
    last_end = -1
    for om in candidates:
        if om.span[0] >= last_end:
            selected.append(om)
            last_end = om.span[1]
    return selected


def redact(text: str) -> tuple[str, list[OutputMatch]]:
    """Replace every match in *text* with :data:`REDACTION_PLACEHOLDER`.

    Returns ``(redacted_text, matches)`` where ``matches`` describes what
    was replaced (in original-text coordinates) so the caller can audit.
    """
    matches = scan_for_unsafe_output(text)
    if not matches:
        return text, []

    # Replace right-to-left so earlier indices stay valid as we mutate.
    parts = list(text)
    for om in reversed(matches):
        parts[om.span[0] : om.span[1]] = REDACTION_PLACEHOLDER
    return "".join(parts), matches
