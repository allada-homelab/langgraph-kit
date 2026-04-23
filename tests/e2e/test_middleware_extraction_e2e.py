"""Cluster B edges — ``ExtractionMiddleware`` positive-emit round-trip.

Every agent build wires ``ExtractionMiddleware`` via the middleware stack
(see ``build_middleware_stack``). After each agent turn, it delegates to
:class:`AutoMemoryExtractor`, which calls the LLM with an extraction
prompt. When the LLM returns a JSON array of memory candidates, the
extractor persists them via :class:`PersistentMemoryManager`.

The e2e angle: we want a regression guard that the *full chain* works —
scripted LLM → extractor's LLM call served → JSON array parsed →
``MemoryRecord`` reaches the ``MockStore`` under the expected namespace.
Previously the extractor had only unit coverage (mocked LLM, direct call
to ``extract()``). If the middleware ever stopped forwarding the post-turn
state, or the extractor's namespace mapping drifted, the unit tests
wouldn't catch it. This test does.

Skip-when-agent-wrote: already unit-tested. This test covers the
opposite path — agent did NOT write memory, so extraction IS expected
to run and emit.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
)

from langgraph_kit.graphs._builder import build_deep_agent
from langgraph_kit.replay import (
    ConversationRecording,
    LLMInteraction,
    RecordedChatModel,
)
from tests.e2e.helpers import answer

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_extraction_emits_recorded_memory_to_store(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> None:
    """End-to-end: scripted turn → extraction LLM → MemoryRecord in store.

    Two scripted LLM calls are expected:
    1. The agent's normal turn (returns final content directly).
    2. The extraction worker's call (returns a JSON array of memory
       candidates).

    The recording serves them in order. ``RecordedChatModel`` advances
    on each ``_generate`` regardless of the caller — the extraction
    LLM's config carries a ``MEMORY_EXTRACTION_TAG`` but that doesn't
    affect the recorded-model dispatch.
    """
    # Candidate payload that the extractor's LLM "returns". The
    # extractor parses this as JSON-array and creates a MemoryRecord.
    candidate = {
        "action": "create",
        "title": "User prefers tacos",
        "type": "user",
        "scope": "user",
        "summary": "culinary preference",
        "body": "User has stated a preference for tacos.",
    }
    extraction_payload = json.dumps([candidate])

    # RecordedChatModel serves interactions sequentially. The agent's
    # first call returns `answer("ok")`; the extractor's LLM call
    # (fires in aafter_agent) gets `answer(extraction_payload)`.
    recording = ConversationRecording(
        interactions=[
            LLMInteraction(sequence_num=1, output_message=answer("ok")),
            LLMInteraction(sequence_num=2, output_message=answer(extraction_payload)),
        ]
    )
    model = RecordedChatModel(recording=recording)

    with patched_build_llm(model):
        graph, _ = build_deep_agent(
            agent_name="extraction-emit-e2e",
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )

    await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="I really like tacos, fyi"),
            ]
        },
        config={"configurable": {"thread_id": "extraction-emit"}},  # pyright: ignore[reportArgumentType]
    )

    # The extractor writes to ``("memory", "user", "user")`` (prefix
    # "memory" + scope "user" + type "user"). MockStore exposes raw
    # state via _data.
    memory_ns = ("memory", "user", "user")
    memory_records = e2e_store._data.get(memory_ns, {})
    assert memory_records, (
        "ExtractionMiddleware should have written at least one MemoryRecord"
        f" under {memory_ns}. MockStore namespaces: {list(e2e_store._data.keys())}"
    )
    # Sanity: one of the records matches our candidate
    matched = [
        v
        for v in memory_records.values()
        if "tacos" in str(v.get("title", "")).lower()
        or "tacos" in str(v.get("body", "")).lower()
    ]
    assert matched, (
        f"No extracted record mentions 'tacos' — payload wasn't threaded through."
        f" Found: {list(memory_records.values())!r}"
    )
    first = matched[0]
    assert first.get("type") == "user", (
        f"Extracted record should have type='user'; got {first!r}"
    )
    assert first.get("scope") == "user", (
        f"Extracted record should have scope='user'; got {first!r}"
    )
    assert first.get("source") == "auto_extraction", (
        f"Extracted record should be marked as auto_extraction; got {first!r}"
    )
