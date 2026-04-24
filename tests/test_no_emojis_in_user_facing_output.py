"""Regression: ensure user-facing output paths do not emit emojis.

The project's CLAUDE.md sets "no emojis unless explicitly requested" as
a baseline convention. Two offenders were shipping emojis by default:
  * async-task status icons (running/success/error/cancelled)
  * trace flowchart span-kind icons (llm/tool/chain/node)

Both have been replaced with plaintext labels.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from langgraph_kit.core.orchestration.async_tasks import (
    AsyncTask,
    AsyncTaskManager,
    AsyncTaskStatus,
    build_async_task_tools,
)
from langgraph_kit.core.tracing.mermaid import trace_to_mermaid
from langgraph_kit.core.tracing.models import TraceRecord, TraceSpan

from .conftest import MockStore

# Matches any character in the "Symbols and Pictographs", "Emoticons",
# "Dingbats", and related blocks. Narrow enough not to flag accented
# letters or CJK, broad enough to catch the culprits (⏳ ✅ ❌ 🚫 🤖 🔧 ⛓️ 📦).
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001f6ff"  # misc symbols + transport
    "\U0001f900-\U0001f9ff"  # supplemental
    "\U00002600-\U000027bf"  # misc symbols + dingbats
    "]"
)


def _assert_no_emojis(text: str, where: str) -> None:
    match = _EMOJI_RE.search(text)
    assert match is None, f"Found emoji {match.group()!r} in {where}: {text!r}"


@pytest.mark.asyncio
async def test_list_async_tasks_is_emoji_free() -> None:
    store = MockStore()
    manager = AsyncTaskManager(store, parent_thread_id="parent-1")

    # Seed one task of each status so every branch of the formatter runs.
    for i, status in enumerate(
        [
            AsyncTaskStatus.RUNNING,
            AsyncTaskStatus.SUCCESS,
            AsyncTaskStatus.ERROR,
            AsyncTaskStatus.CANCELLED,
        ]
    ):
        task = AsyncTask(
            task_id=f"t-{i}",
            thread_id=f"sub-{i}",
            agent_name="worker",
            description=f"task-{status.value}",
            status=status,
        )
        await store.aput(
            ("async_tasks", "parent-1"),
            task.task_id,
            task.model_dump(mode="json"),
        )

    # Exercise the CLI-facing tool directly.
    tools: list[Any] = build_async_task_tools(manager=manager, available_graphs={})
    # Resolve by __name__ — tools are plain async functions here.
    list_tool = None
    for t in tools:
        if getattr(t, "__name__", "") == "list_async_tasks":
            list_tool = t
            break
    assert list_tool is not None, "list_async_tasks tool not found"
    out = await list_tool()
    _assert_no_emojis(str(out), "list_async_tasks output")


def test_trace_to_mermaid_flowchart_is_emoji_free() -> None:
    trace = TraceRecord(
        trace_id="t1",
        name="root",
        started_at="2026-04-24T00:00:00Z",
        ended_at="2026-04-24T00:00:01Z",
        duration_ms=1000.0,
        spans=[
            TraceSpan(
                span_id="s1",
                name="step-1",
                kind="llm",
                started_at="2026-04-24T00:00:00Z",
                ended_at="2026-04-24T00:00:00.5Z",
                duration_ms=500.0,
            ),
            TraceSpan(
                span_id="s2",
                name="step-2",
                kind="tool",
                started_at="2026-04-24T00:00:00.5Z",
                ended_at="2026-04-24T00:00:01Z",
                duration_ms=500.0,
            ),
        ],
    )
    flow = trace_to_mermaid(trace, style="flowchart")
    _assert_no_emojis(flow, "flowchart diagram")
    # Sanity: plaintext kind labels are present so the diagram isn't just stripped.
    assert "LLM:" in flow
    assert "TOOL:" in flow


def test_trace_to_mermaid_sequence_is_emoji_free() -> None:
    trace = TraceRecord(
        trace_id="t2",
        name="root",
        started_at="2026-04-24T00:00:00Z",
        ended_at="2026-04-24T00:00:01Z",
        duration_ms=1000.0,
        spans=[
            TraceSpan(
                span_id="s1",
                name="step",
                kind="llm",
                started_at="2026-04-24T00:00:00Z",
                ended_at="2026-04-24T00:00:01Z",
                duration_ms=1000.0,
            ),
        ],
    )
    seq = trace_to_mermaid(trace, style="sequence")
    _assert_no_emojis(seq, "sequence diagram")
