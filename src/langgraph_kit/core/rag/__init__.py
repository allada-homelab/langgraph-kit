"""RAG primitives: document ingestion, retrieval index, and search-knowledge tool.

Foundation layer for issue #16 — chunking + embedding + vector search
on top of any LangGraph ``BaseStore``. Provider-agnostic: the embedding
function is supplied by the caller (same convention as the opt-in
semantic memory search added in #8). No new heavyweight dependencies.

Citation verification and the grounding-eval rubric are tracked
separately in a follow-up issue.
"""

from .ingest import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    Document,
    word_chunker,
)
from .retrieval import RetrievalIndex, RetrievedChunk
from .search_tool import build_search_knowledge_tool

__all__ = [
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "Document",
    "RetrievalIndex",
    "RetrievedChunk",
    "build_search_knowledge_tool",
    "word_chunker",
]
