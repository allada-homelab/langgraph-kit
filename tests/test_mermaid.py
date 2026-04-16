"""Tests for tracing mermaid diagram generation."""

from __future__ import annotations

from langgraph_kit.core.tracing.mermaid import _safe_name, trace_to_mermaid
from langgraph_kit.core.tracing.models import TraceRecord, TraceSpan


def _make_trace(*spans: TraceSpan) -> TraceRecord:
    return TraceRecord(
        trace_id="t1",
        agent_id="test",
        thread_id="th1",
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:00:01Z",
        duration_ms=1000.0,
        spans=list(spans),
    )


class TestSafeName:
    def test_strips_quotes(self) -> None:
        assert _safe_name('say "hello"') == "say 'hello'"

    def test_strips_newlines(self) -> None:
        assert _safe_name("line1\nline2") == "line1 line2"

    def test_truncates_long_names(self) -> None:
        result = _safe_name("a" * 100)
        assert len(result) == 50


class TestTraceToMermaid:
    def test_sequence_diagram_header(self) -> None:
        trace = _make_trace()
        result = trace_to_mermaid(trace, style="sequence")
        assert result.startswith("sequenceDiagram")
        assert "participant Agent" in result

    def test_sequence_with_llm_span(self) -> None:
        span = TraceSpan(
            span_id="s1", kind="llm", name="gpt-4o", duration_ms=500.0
        )
        trace = _make_trace(span)
        result = trace_to_mermaid(trace)
        assert "Agent->>LLM: gpt-4o" in result
        assert "500ms" in result

    def test_sequence_with_tool_span(self) -> None:
        span = TraceSpan(
            span_id="s1", kind="tool", name="search_files", duration_ms=100.0
        )
        trace = _make_trace(span)
        result = trace_to_mermaid(trace)
        assert "Agent->>Tool: search_files" in result

    def test_flowchart_header(self) -> None:
        trace = _make_trace()
        result = trace_to_mermaid(trace, style="flowchart")
        assert result.startswith("flowchart TD")

    def test_flowchart_with_span(self) -> None:
        span = TraceSpan(
            span_id="s1", kind="tool", name="grep", duration_ms=50.0
        )
        trace = _make_trace(span)
        result = trace_to_mermaid(trace, style="flowchart")
        assert "grep" in result

    def test_empty_trace(self) -> None:
        trace = _make_trace()
        seq = trace_to_mermaid(trace, style="sequence")
        assert "sequenceDiagram" in seq
        flow = trace_to_mermaid(trace, style="flowchart")
        assert "flowchart TD" in flow
