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
    """Generate a Mermaid flowchart including parent→child edges."""
    lines = ["flowchart TD"]
    counter = [0]  # unique node-id generator shared across recursion
    root_ids: list[str] = []

    for span in trace.spans:
        root_id = _flowchart_span(span, lines, counter, parent_node_id=None)
        root_ids.append(root_id)

    # Connect sequential root spans (keeps the prior top-level ordering).
    for i in range(len(root_ids) - 1):
        lines.append(f"    {root_ids[i]} --> {root_ids[i + 1]}")

    return "\n".join(lines)


def _flowchart_span(
    span: TraceSpan,
    lines: list[str],
    counter: list[int],
    parent_node_id: str | None,
) -> str:
    """Add a flowchart node for ``span`` plus edges to its children.

    Returns the generated node_id so callers can wire edges.
    """
    node_id = f"n{counter[0]}"
    counter[0] += 1
    name = _safe_name(span.name)
    duration = f"<br/>{span.duration_ms:.0f}ms" if span.duration_ms else ""
    kind_tag = {
        "llm": "LLM",
        "tool": "TOOL",
        "chain": "CHAIN",
        "node": "NODE",
    }.get(span.kind, span.kind.upper() if span.kind else "SPAN")
    lines.append(f'    {node_id}["{kind_tag}: {name}{duration}"]')

    # Draw parent→child edge when nested.
    if parent_node_id is not None:
        lines.append(f"    {parent_node_id} --> {node_id}")

    # Recurse. Prior implementation only drew root spans — all nesting
    # was silently dropped from the flowchart.
    for child in span.children:
        _flowchart_span(child, lines, counter, parent_node_id=node_id)

    return node_id


# Characters that break Mermaid's parser inside node labels even within
# double quotes: brackets define shape containers, parens + pipes alias
# node forms, the arrow token ``-->`` is the edge literal, and backticks
# start inline code spans in markdown renderings.
_MERMAID_RESERVED = str.maketrans(
    {
        "[": "(",
        "]": ")",
        "{": "(",
        "}": ")",
        "|": "-",
        "`": "'",
    }
)


def _safe_name(name: str) -> str:
    """Sanitize a name for Mermaid syntax.

    Replaces quotes, newlines, and Mermaid-reserved characters before
    truncating to 50 chars so node labels render cleanly regardless of
    what span names upstream frameworks emit. Also collapses ``-->`` so
    it doesn't look like an edge literal embedded in a label.
    """
    cleaned = (
        name.replace('"', "'")
        .replace("\n", " ")
        .replace("-->", " to ")
        .translate(_MERMAID_RESERVED)
    )
    return cleaned[:50]
