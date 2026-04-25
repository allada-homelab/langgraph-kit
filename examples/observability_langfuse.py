"""Observability: Langfuse tracing wired into the kit (real-LLM only).

What this shows
---------------
- :class:`AgentConfig` fields that enable Langfuse:
  ``langfuse_host``, ``langfuse_public_key``, ``langfuse_secret_key``,
  ``langfuse_tracing_enabled``
- Running a real agent turn so the trace appears in the Langfuse UI
- Calling :func:`flush_langfuse` at shutdown so spans aren't dropped

This demo is hermetic-skip by design: a hermetic LLM never produces
meaningful traces, so the example exits 0 in scripted mode with a
"skipped" message and runs against a real model in the nightly
real-LLM workflow only.

How to run
----------
    LANGGRAPH_KIT_EXAMPLES_LLM=real \\
        AGENT_LLM_API_KEY=sk-... \\
        LANGFUSE_HOST=https://cloud.langfuse.com \\
        LANGFUSE_PUBLIC_KEY=pk-... \\
        LANGFUSE_SECRET_KEY=sk-... \\
        uv run python -m examples.observability_langfuse

Expected output (real LLM)
--------------------------
    Tracing run via Langfuse to https://cloud.langfuse.com
    user: How do I observe agent behaviour?
    assistant: <real-model output>
    Trace flushed; check the Langfuse UI for the run details.
"""

from __future__ import annotations

import asyncio
import os

from examples._lib import (
    assert_real_llm_or_skip,
    banner,
    configure_real_llm,
    line,
    make_in_memory_persistence,
    tmp_workspace,
)

REQUIRES_NETWORK = True  # Skipped in per-PR hermetic CI; runs in nightly.


async def main() -> None:
    banner("observability_langfuse")
    assert_real_llm_or_skip()

    from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
        HumanMessage,
    )

    from langgraph_kit import AgentConfig, configure
    from langgraph_kit.graphs.echo_agent import build_graph
    from langgraph_kit.observability import flush_langfuse

    host = os.environ.get("LANGFUSE_HOST", "")
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if not (host and pk and sk):
        line(
            "Skipping: LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY "
            "must all be set."
        )
        return

    with tmp_workspace() as workspace:
        configure_real_llm(workspace)
        # Re-configure with the Langfuse fields layered on top.
        from langgraph_kit import get_config

        cur = get_config()
        configure(
            AgentConfig(
                llm_model=cur.llm_model,
                llm_api_key=cur.llm_api_key,
                database_url=cur.database_url,
                langfuse_host=host,
                langfuse_public_key=pk,
                langfuse_secret_key=sk,
                langfuse_tracing_enabled=True,
            )
        )
        line(f"Tracing run via Langfuse to {host}")

        checkpointer, store = make_in_memory_persistence()
        graph = build_graph(checkpointer, store)

        user = "How do I observe agent behaviour?"
        line(f"user: {user}")
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=user)]},
            config={"configurable": {"thread_id": "demo-langfuse"}},
        )
        last = result["messages"][-1]
        line(f"assistant: {getattr(last, 'content', '')}")

        flush_langfuse()
        line("Trace flushed; check the Langfuse UI for the run details.")


if __name__ == "__main__":
    asyncio.run(main())
