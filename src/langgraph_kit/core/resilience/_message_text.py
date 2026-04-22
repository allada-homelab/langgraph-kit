"""Shared helper for extracting plain-text content from an ``AIMessage``.

``AIMessage.content`` is typed ``str | list[str | dict]``; the list form
carries multi-part content (e.g. tool-use deltas) that the resilience
middleware never inspects. Centralizing the narrow-to-``str`` conversion
keeps the pyright suppressions in one place.
"""

from __future__ import annotations

from typing import Any


def aimessage_text(msg: Any) -> str:
    """Return ``msg.content`` as a ``str``, or ``""`` for non-``str`` content."""
    raw: Any = msg.content  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    return raw if isinstance(raw, str) else ""
