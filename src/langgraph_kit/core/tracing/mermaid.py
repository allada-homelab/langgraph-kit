"""Generate Mermaid diagram markup from execution traces."""

from __future__ import annotations

from langgraph_kit.core.tracing.models import TraceRecord, TraceSpan


def trace_to_mermaid(trace: TraceRecord, *, style: str = "sequence") -> str:
    """Convert a trace to a Mermaid diagram string.

    Parameters
    ----------
    trace:
        The execution trace to visualize.
    style:
        ``"sequence"`` for a sequence diagram, ``"flowchart"`` for a flowchart.
    """
    if style == "flowchart":
        return _flowchart(trace)
    return _sequence(trace)


def _sequence(trace: TraceRecord) -> str:
    """Generate a Mermaid sequence diagram."""
    lines = ["sequenceDiagram"]
    lines.append("    participant User")
    lines.append("    participant Agent")
    lines.append("    participant LLM")
    lines.append("    participant Tool")

    for span in trace.spans:
        _sequence_span(span, lines)

    return "\n".join(lines)


def _sequence_span(span: TraceSpan, lines: list[str], depth: int = 0) -> None:
    """Recursively add sequence diagram entries for a span and its children."""
    duration = f" ({span.duration_ms:.0f}ms)" if span.duration_ms else ""
    name = _safe_name(span.name)

    if span.kind == "llm":
        lines.append(f"    Agent->>LLM: {name}{duration}")
        lines.append("    LLM-->>Agent: response")
    elif span.kind == "tool":
        lines.append(f"    Agent->>Tool: {name}{duration}")
        lines.append("    Tool-->>Agent: result")
    elif span.kind == "chain" and depth == 0:
        lines.append(f"    User->>Agent: invoke ({name})")

    for child in span.children:
        _sequence_span(child, lines, depth + 1)

    if span.kind == "chain" and depth == 0:
        lines.append(f"    Agent-->>User: response{duration}")


def _flowchart(trace: TraceRecord) -> str:
    """Generate a Mermaid flowchart showing node transitions."""
    lines = ["flowchart TD"]
    node_ids: list[str] = []

    for i, span in enumerate(trace.spans):
        _flowchart_span(span, lines, node_ids, i)

    # Connect sequential root spans
    for i in range(len(node_ids) - 1):
        lines.append(f"    {node_ids[i]} --> {node_ids[i + 1]}")

    return "\n".join(lines)


def _flowchart_span(
    span: TraceSpan,
    lines: list[str],
    node_ids: list[str],
    idx: int,
) -> None:
    """Add a flowchart node for a span."""
    node_id = f"n{idx}"
    name = _safe_name(span.name)
    duration = f"<br/>{span.duration_ms:.0f}ms" if span.duration_ms else ""
    kind_tag = {
        "llm": "LLM",
        "tool": "TOOL",
        "chain": "CHAIN",
        "node": "NODE",
    }.get(span.kind, span.kind.upper() if span.kind else "SPAN")
    lines.append(f'    {node_id}["{kind_tag}: {name}{duration}"]')
    node_ids.append(node_id)


def _safe_name(name: str) -> str:
    """Sanitize a name for Mermaid syntax."""
    return name.replace('"', "'").replace("\n", " ")[:50]
