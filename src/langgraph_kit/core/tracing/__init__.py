"""Execution trace collection and visualization.

Collect execution traces showing node timing, tool calls, and LLM interactions.
Export as structured JSON or Mermaid diagrams for visual debugging.

Usage::

    from langgraph_kit.core.tracing import TraceCallbackHandler, trace_to_mermaid

    handler = TraceCallbackHandler(agent_id="my-agent", thread_id="t1")
    config["callbacks"] = [handler]
    await graph.ainvoke(input_data, config=config)
    trace = handler.get_trace()
    print(trace_to_mermaid(trace))
"""

from langgraph_kit.core.tracing.handler import TraceCallbackHandler
from langgraph_kit.core.tracing.mermaid import trace_to_mermaid
from langgraph_kit.core.tracing.models import TraceRecord, TraceSpan, TraceSummary
from langgraph_kit.core.tracing.storage import TraceStore

__all__ = [
    "TraceCallbackHandler",
    "TraceRecord",
    "TraceSpan",
    "TraceStore",
    "TraceSummary",
    "trace_to_mermaid",
]
