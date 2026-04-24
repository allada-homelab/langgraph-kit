"""Shared JSON parsing utilities for memory extraction and consolidation."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Markdown code-fence wrappers LLMs commonly use: ```json\n...\n``` or ```\n...\n```.
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.MULTILINE)


def parse_json_array(
    raw: str, *, context: str = "LLM response"
) -> list[dict[str, Any]]:
    """Parse a JSON array from a raw string, with tolerant fallback.

    Order of attempts:
        1. Direct ``json.loads`` of the stripped input.
        2. Strip any ```` ``` ```` / ```` ```json ```` code-fence wrapping
           and retry.
        3. Bracket-balanced scan to locate the first top-level ``[ ... ]``
           substring and parse that.

    Returns ``[]`` on total failure after a warning log. Only list-of-dict
    results are returned — other shapes (e.g. ``[1, 2, 3]``) also produce
    ``[]`` because callers dereference ``.get(...)`` on each item.

    The bracket-balanced scan replaces an earlier ``r"\\[.*\\]"`` regex that
    was greedy across the whole input — two JSON-looking arrays in the
    same response made the regex span both and ``json.loads`` reject the
    combined string, which silently dropped everything.
    """
    text = raw.strip()

    # 1. Direct JSON parse.
    result = _try_load_as_list(text)
    if result is not None:
        return result

    # 2. Strip a surrounding markdown code fence and retry.
    unfenced = _FENCE_RE.sub("", text).strip()
    if unfenced != text:
        result = _try_load_as_list(unfenced)
        if result is not None:
            return result

    # 3. Bracket-balanced scan: find the first top-level array.
    candidate = _first_balanced_array(text)
    if candidate is not None:
        result = _try_load_as_list(candidate)
        if result is not None:
            return result

    logger.warning("Failed to parse %s", context)
    return []


def _try_load_as_list(text: str) -> list[dict[str, Any]] | None:
    """Return parsed list only if the input is a valid JSON list of dicts."""
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    # Filter to dict elements — callers assume item.get(...) works.
    return [item for item in parsed if isinstance(item, dict)]


def _first_balanced_array(text: str) -> str | None:
    """Return the first top-level ``[...]`` substring using bracket balancing.

    Skips brackets that appear inside strings (quoted by ``"`` with
    escape handling). Returns ``None`` if no balanced array is found.
    """
    depth = 0
    start = -1
    in_str = False
    escape = False

    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start != -1:
                return text[start : i + 1]
    return None
