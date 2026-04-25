"""Factory for the ``search_knowledge`` agent tool.

Builds a :class:`ToolCapability` that wraps a :class:`RetrievalIndex`.
The resulting tool is what the LLM calls (``search_knowledge`` returns
a formatted block of ranked chunks); the index itself is the persistent
state.
"""

from __future__ import annotations

import logging

from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk

from .retrieval import RetrievalIndex

logger = logging.getLogger(__name__)


_PROMPT_GUIDANCE = (
    "Use search_knowledge to find supporting passages from the configured "
    "knowledge base before answering knowledge-grounded questions. The tool "
    "returns ranked chunks; cite them in your answer using add_citation."
)


def build_search_knowledge_tool(
    index: RetrievalIndex,
    *,
    default_top_k: int = 5,
    tool_id: str = "rag.search_knowledge",
    description: str = "Search the configured knowledge base for relevant passages.",
) -> ToolCapability:
    """Wrap a :class:`RetrievalIndex` as an agent tool.

    The tool function name is always ``search_knowledge`` so prompts
    referencing it stay consistent across agents; the registry id can be
    customised when multiple indexes coexist.
    """

    async def search_knowledge(query: str, top_k: int | None = None) -> str:
        """Search the knowledge base for passages relevant to ``query``.

        Args:
            query: Natural-language search query.
            top_k: Optional override for the number of chunks returned.
        """
        k = top_k or default_top_k
        chunks = await index.asearch(query=query, top_k=k)
        if not chunks:
            return "No matching passages found in the knowledge base."

        lines = [f"Found {len(chunks)} relevant passage(s):\n"]
        for chunk in chunks:
            preview = chunk.text if len(chunk.text) <= 800 else chunk.text[:800] + "..."
            lines.append(
                f"- [{chunk.chunk_id}] (score={chunk.score:.3f}, doc={chunk.doc_id})"
            )
            lines.append(f"  {preview}")
        return "\n".join(lines)

    return ToolCapability(
        id=tool_id,
        name="search_knowledge",
        description=description,
        fn=search_knowledge,
        tags=["rag", "retrieval"],
        risk=ToolRisk.READ_ONLY,
        prompt_guidance=_PROMPT_GUIDANCE,
    )
