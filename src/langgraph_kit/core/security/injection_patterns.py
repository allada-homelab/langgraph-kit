"""Regex patterns for known prompt-injection phrasings.

This is **best-effort** — a regex set will never catch every paraphrase.
Treat detection as defense-in-depth, never as a gate. Pair with a
classifier (opt-in via the middleware's ``classifier=`` hook) when you
need broader paraphrase coverage.

Each pattern carries a short comment naming the attack family it
covers; that provenance matters when tuning false positives.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# Imperative override — the canonical injection prefix.
_IGNORE_PREVIOUS = re.compile(
    r"\bignore\s+(?:all\s+|the\s+|every\s+)?"
    r"(?:previous|prior|preceding|above|earlier)\s+"
    r"(?:instructions?|rules?|context|messages?|prompts?)\b",
    re.IGNORECASE,
)

# Persona override.
_YOU_ARE_NOW = re.compile(
    r"\byou\s+are\s+(?:now|actually|really)\s+(?:a|an|the)\b",
    re.IGNORECASE,
)
_FORGET_ROLE = re.compile(
    r"\b(?:forget|disregard)\s+(?:your|the|all)\s+"
    r"(?:role|persona|character|instructions?|prompts?)\b",
    re.IGNORECASE,
)

# Pseudo-XML wrappers commonly used to spoof system / admin scopes.
_PSEUDO_SYSTEM_TAG = re.compile(
    r"<\s*/?\s*(?:system|admin|developer|instructions?|rules?)\s*>",
    re.IGNORECASE,
)

# Jailbreak vocabulary.
_JAILBREAK = re.compile(
    r"\b(?:jailbreak|jail\s*break|developer\s+mode|do\s+anything\s+now|DAN\s+mode)\b",
    re.IGNORECASE,
)

# Prompt-leak attempts.
_REVEAL_SYSTEM = re.compile(
    r"\b(?:reveal|print|show|leak|repeat|output)\s+"
    r"(?:your|the)\s+"
    r"(?:system\s+prompt|hidden\s+prompt|internal\s+rules?|"
    r"initial\s+instructions?|configuration)",
    re.IGNORECASE,
)

# Override-via-quotes / authoritative framing.
_OVERRIDE_VOICE = re.compile(
    r"\b(?:as\s+an?\s+)?admin(?:istrator)?\s*[:\-—,]\s*",
    re.IGNORECASE,
)


INJECTION_PATTERNS: list[re.Pattern[str]] = [
    _IGNORE_PREVIOUS,
    _YOU_ARE_NOW,
    _FORGET_ROLE,
    _PSEUDO_SYSTEM_TAG,
    _JAILBREAK,
    _REVEAL_SYSTEM,
    _OVERRIDE_VOICE,
]


class InjectionMatch(NamedTuple):
    """A pattern match found in user-supplied text."""

    pattern: str  # the pattern's name/source — useful for audit + tuning
    match: str  # the substring that matched
    span: tuple[int, int]


_PATTERN_NAMES: dict[re.Pattern[str], str] = {
    _IGNORE_PREVIOUS: "ignore_previous_instructions",
    _YOU_ARE_NOW: "you_are_now_persona_override",
    _FORGET_ROLE: "forget_role",
    _PSEUDO_SYSTEM_TAG: "pseudo_system_tag",
    _JAILBREAK: "jailbreak_vocabulary",
    _REVEAL_SYSTEM: "reveal_system_prompt",
    _OVERRIDE_VOICE: "admin_voice_override",
}


def scan_for_injection(
    text: str,
    *,
    patterns: list[re.Pattern[str]] | None = None,
) -> list[InjectionMatch]:
    """Return every injection pattern hit in *text*.

    Order is deterministic (input-pattern order, then left-to-right).
    Empty input or no matches yields an empty list — never raises.
    """
    if not text:
        return []
    candidates = patterns if patterns is not None else INJECTION_PATTERNS
    matches: list[InjectionMatch] = []
    for pat in candidates:
        for m in pat.finditer(text):
            name = _PATTERN_NAMES.get(pat, pat.pattern[:60])
            matches.append(
                InjectionMatch(
                    pattern=name,
                    match=m.group(0),
                    span=(m.start(), m.end()),
                )
            )
    return matches
