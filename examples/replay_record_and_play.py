"""Replay: record a hermetic run, then load and inspect the recording.

What this shows
---------------
- Attaching :class:`ConversationRecorder` as a LangChain callback to
  capture every LLM + tool interaction during a graph run
- Serializing the recording to disk (``recorder.save(path)``) and
  reloading it (``ConversationRecording.model_validate_json(...)``)
- Inspecting captured ``llm_interactions`` and ``tool_interactions``

This is how the kit's e2e suite produces deterministic regression
fixtures. For full record-once / replay-many-times in tests, use
:class:`ReplayRunner` (see ``tests/e2e/test_replay_recorder_e2e.py``).

How to run
----------
    uv run python -m examples.replay_record_and_play

Expected output
---------------
    Recording saved to /tmp/lgk-example-XXXX/conv.json (NN bytes)
    Recorded LLM interactions: 1
    Recorded tool interactions: 0
    Last assistant content: I recorded this run.
"""

from __future__ import annotations

import asyncio

from examples._lib import (
    answer,
    banner,
    line,
    make_in_memory_persistence,
    patch_build_llm,
    scripted_llm,
    tmp_workspace,
)


async def main() -> None:
    banner("replay_record_and_play")

    from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
        HumanMessage,
    )

    from langgraph_kit.graphs.echo_agent import build_graph
    from langgraph_kit.replay import (
        ConversationRecorder,
        ConversationRecording,
    )

    with tmp_workspace() as workspace:
        # 1. Build a hermetic graph and attach a recorder.
        llm = scripted_llm([answer("I recorded this run.")])
        with patch_build_llm(llm):
            checkpointer, store = make_in_memory_persistence()
            graph = build_graph(checkpointer, store)

            recorder = ConversationRecorder(
                agent_id="example-replay", thread_id="thread-1"
            )

            await graph.ainvoke(
                {"messages": [HumanMessage(content="Say something memorable.")]},
                config={  # pyright: ignore[reportArgumentType]
                    "configurable": {"thread_id": "thread-1"},
                    "callbacks": [recorder],
                },
            )

        # 2. Save the recording.
        recording_path = workspace / "conv.json"
        recorder.save(recording_path)
        size = recording_path.stat().st_size
        line(f"Recording saved to {recording_path} ({size} bytes)")

        # 3. Reload and inspect.
        loaded = ConversationRecording.model_validate_json(recording_path.read_text())
        line(f"Recorded LLM interactions: {len(loaded.llm_interactions)}")
        line(f"Recorded tool interactions: {len(loaded.tool_interactions)}")

        if loaded.llm_interactions:
            last = loaded.llm_interactions[-1]
            content = last.output_message.get("content", "")
            line(f"Last assistant content: {content}")


if __name__ == "__main__":
    asyncio.run(main())
