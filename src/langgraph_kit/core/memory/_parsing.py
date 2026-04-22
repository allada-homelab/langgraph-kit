"""Shared JSON parsing utilities for memory extraction and consolidation."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_RE_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


def parse_json_array(
    raw: str, *, context: str = "LLM response"
) -> list[dict[str, Any]]:
    """Parse a JSON array from a raw string, with fallback regex extraction.

    Tries direct JSON parse first, then attempts to extract a JSON array
    from surrounding text (e.g., markdown fences or explanatory prose).
    """
    text = raw.strip()

    # Try direct JSON parse
    try:
        parsed: Any = json.loads(text)
        if isinstance(parsed, list):
            return parsed  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass

    # Try extracting JSON array from surrounding text
    match = _RE_JSON_ARRAY.search(text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parsed  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse %s", context)
    return []
