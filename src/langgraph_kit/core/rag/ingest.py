"""Document model and the default word-based chunker.

Kept minimal on purpose — heavier chunkers (code-aware, structural,
semantic) belong in an extras module so the kit base install stays
dependency-free.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

# Default targets ~3.2k chars per chunk with 200-char overlap. The 800
# "tokens" is approximate — we operate in characters so we don't depend
# on any specific tokenizer. 3.2k chars is roughly 800 tokens for
# English prose at the LLaMA / GPT-3.5 ratio (~4 chars per token).
DEFAULT_CHUNK_SIZE: int = 3200
DEFAULT_CHUNK_OVERLAP: int = 200

# Pluggable chunker contract — takes raw text and returns ordered chunks.
Chunker = Callable[[str], list[str]]


class Document(BaseModel):
    """A document to be ingested into a :class:`RetrievalIndex`."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def word_chunker(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into overlapping chunks at word boundaries.

    Tokenises whitespace-separated words and greedily packs them into
    chunks of approximately ``chunk_size`` characters. Each subsequent
    chunk re-includes enough trailing words from the previous chunk to
    cover ``chunk_overlap`` characters. Words are never split, so a
    very long single word lands in its own oversized chunk rather than
    breaking mid-token.
    """
    if chunk_overlap >= chunk_size:
        msg = "chunk_overlap must be smaller than chunk_size"
        raise ValueError(msg)

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0
    for word in words:
        # +1 accounts for the joining space we'd add when assembling.
        delta = len(word) + (1 if current else 0)
        if current_chars + delta > chunk_size and current:
            chunks.append(" ".join(current))
            # Build overlap from the tail of the just-flushed chunk.
            overlap_words: list[str] = []
            overlap_chars = 0
            for w in reversed(current):
                w_delta = len(w) + (1 if overlap_words else 0)
                if overlap_chars + w_delta > chunk_overlap:
                    break
                overlap_words.insert(0, w)
                overlap_chars += w_delta
            current = overlap_words
            current_chars = overlap_chars

        if current:
            current_chars += 1  # joining space
        current.append(word)
        current_chars += len(word)

    if current:
        chunks.append(" ".join(current))

    return chunks
