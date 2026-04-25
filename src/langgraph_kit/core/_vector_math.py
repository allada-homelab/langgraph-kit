"""Tiny vector helpers shared by the memory and RAG indexes.

Lives at ``langgraph_kit.core._vector_math`` (underscore-prefixed to mark
it as kit-internal — public callers should not depend on its surface).
The helpers are dependency-free so adding embeddings doesn't pull in
numpy or scipy on installs that haven't opted into RAG / semantic
memory.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length numeric vectors.

    Returns 0.0 (rather than raising) on length mismatch or when either
    vector has zero magnitude — those degenerate cases mean "no
    meaningful similarity to score" and ranking on them would be
    misleading.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def top_k_by_score(
    scored: Iterable[tuple[float, object]],
    k: int,
    *,
    drop_zero: bool = True,
) -> list[tuple[float, object]]:
    """Return the top-``k`` ``(score, payload)`` tuples by descending score.

    ``drop_zero`` filters out exact-zero scores before ranking — handy
    for cosine-similarity pipelines where 0.0 reliably signals
    "completely orthogonal" rather than "tied for last".
    """
    items = list(scored)
    if drop_zero:
        items = [(s, p) for s, p in items if s > 0.0]
    items.sort(key=lambda pair: pair[0], reverse=True)
    return items[:k]
