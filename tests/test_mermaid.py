# pyright: reportPrivateUsage=false
# Strict-mode prep: this test probes private module internals
# (``_safe_name``) by design.  Disabling reportPrivateUsage here keeps
# the file clean under a future ``typeCheckingMode = "strict"`` flip
# without weakening type safety elsewhere.
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
        span = TraceSpan(span_id="s1", kind="llm", name="gpt-4o", duration_ms=500.0)
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
        span = TraceSpan(span_id="s1", kind="tool", name="grep", duration_ms=50.0)
        trace = _make_trace(span)
        result = trace_to_mermaid(trace, style="flowchart")
        assert "grep" in result

    def test_empty_trace(self) -> None:
        trace = _make_trace()
        seq = trace_to_mermaid(trace, style="sequence")
        assert "sequenceDiagram" in seq
        flow = trace_to_mermaid(trace, style="flowchart")
        assert "flowchart TD" in flow


class TestSequenceNestedRendering:
    """Sequence-mode child rendering — regression tests for the
    pre-fix ``depth == 0`` gate that dropped nested chain spans."""

    def test_nested_chain_is_rendered_as_note(self) -> None:
        """A chain inside a chain shows up as a Note rather than being dropped."""
        outer = TraceSpan(
            span_id="outer",
            kind="chain",
            name="outer_chain",
            duration_ms=100.0,
            children=[
                TraceSpan(
                    span_id="inner",
                    kind="chain",
                    name="inner_subgraph",
                    duration_ms=40.0,
                ),
            ],
        )
        trace = _make_trace(outer)
        result = trace_to_mermaid(trace, style="sequence")

        assert "User->>Agent: invoke (outer_chain)" in result
        assert "Note over Agent: chain (inner_subgraph)" in result
        assert "Agent-->>User: response" in result

    def test_deeply_nested_llm_span_under_chains_is_rendered(self) -> None:
        """An LLM span at depth > 0 must reach the diagram (regression for the gate)."""
        deep = TraceSpan(
            span_id="root",
            kind="chain",
            name="root",
            children=[
                TraceSpan(
                    span_id="mid",
                    kind="chain",
                    name="mid",
                    children=[
                        TraceSpan(
                            span_id="leaf",
                            kind="llm",
                            name="claude-opus",
                            duration_ms=250.0,
                        ),
                    ],
                ),
            ],
        )
        trace = _make_trace(deep)
        result = trace_to_mermaid(trace, style="sequence")

        # Leaf LLM call rendered even though it's nested two chains deep.
        assert "Agent->>LLM: claude-opus" in result

    def test_max_depth_truncates_with_explicit_note(self) -> None:
        """Past max_depth, render a truncation note instead of recursing forever."""
        # Build a chain 5 levels deep.
        leaf = TraceSpan(span_id="l5", kind="chain", name="leaf")
        for i in range(4, 0, -1):
            leaf = TraceSpan(
                span_id=f"l{i}", kind="chain", name=f"l{i}", children=[leaf]
            )
        trace = _make_trace(leaf)

        seq = trace_to_mermaid(trace, style="sequence", max_depth=2)
        assert "truncated at depth 2" in seq

    def test_flowchart_max_depth_truncates_with_node(self) -> None:
        """Flowchart honors max_depth too — emits a single ...truncated node."""
        leaf = TraceSpan(span_id="l5", kind="tool", name="deep_tool")
        for i in range(4, 0, -1):
            leaf = TraceSpan(
                span_id=f"l{i}", kind="chain", name=f"l{i}", children=[leaf]
            )
        trace = _make_trace(leaf)

        flow = trace_to_mermaid(trace, style="flowchart", max_depth=2)
        assert "...truncated at depth 2" in flow
