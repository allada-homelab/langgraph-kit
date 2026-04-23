"""Tags attached to internal LLM calls so they can be filtered from user-facing streams.

Several middlewares and helpers in this kit (auto memory extraction,
consolidation, context compaction, LLM-based agent routing) issue their own
``LLM.ainvoke`` calls while a graph run is in flight. Those calls emit
``on_chat_model_stream`` events that ``astream_events(version="v2")`` forwards
to consumers, indistinguishable from the main agent's reply unless they carry
identifying metadata. The result, reported downstream, was the extractor's
JSON candidate array leaking into the user-facing chat bubble after the real
reply finished.

Every internal call now tags itself with :data:`INTERNAL_TAG` plus a
call-site-specific tag. Consumers can filter on either::

    async for event in graph.astream_events(..., version="v2"):
        if INTERNAL_TAG in (event.get("tags") or ()):
            continue
        ...

The kit's own :func:`langgraph_kit.streaming.stream_agent_events` helper
applies this filter automatically.
"""

from __future__ import annotations

INTERNAL_TAG = "langgraph_kit:internal"
"""Umbrella tag on every internal LLM call. Filter on this to drop them all."""

MEMORY_EXTRACTION_TAG = "langgraph_kit:memory_extraction"
MEMORY_CONSOLIDATION_TAG = "langgraph_kit:memory_consolidation"
CONTEXT_COMPACTION_TAG = "langgraph_kit:context_compaction"
AGENT_ROUTING_TAG = "langgraph_kit:agent_routing"


def internal_llm_config(
    specific_tag: str, *, run_name: str | None = None
) -> dict[str, object]:
    """Return a ``RunnableConfig`` dict that marks an LLM call as internal.

    Pass the returned dict as the ``config=`` kwarg to ``ainvoke``/``invoke``
    (or feed it to ``llm.with_config(...)``) so the resulting
    ``on_chat_model_*`` events carry both :data:`INTERNAL_TAG` and the
    call-site-specific ``specific_tag``.
    """
    cfg: dict[str, object] = {"tags": [INTERNAL_TAG, specific_tag]}
    if run_name:
        cfg["run_name"] = run_name
    return cfg


__all__ = [
    "AGENT_ROUTING_TAG",
    "CONTEXT_COMPACTION_TAG",
    "INTERNAL_TAG",
    "MEMORY_CONSOLIDATION_TAG",
    "MEMORY_EXTRACTION_TAG",
    "internal_llm_config",
]
