"""Render a compiled LangGraph agent's static structure.

The kit already ships trace-replay rendering
(``langgraph_kit.core.tracing.mermaid``); this module is the
*structure* counterpart — what does the graph *look like* before
it runs, independent of any one execution.

Usage::

    from langgraph_kit.core.visualization import print_graph

    graph = build_my_agent(checkpointer, store)
    mermaid = print_graph(graph)
    print(mermaid)  # paste into MkDocs / GitHub / mermaid.live

    # ASCII (requires grandalf — install via langgraph-kit[viz] or grandalf):
    print(print_graph(graph, format="ascii"))

The actual rendering is delegated to LangChain Core's
``Graph.draw_mermaid`` / ``Graph.draw_ascii`` (which the kit already
takes as a transitive dependency); this wrapper adds:

1. A consistent kit-level entry point so callers don't have to
   memorize ``graph.get_graph().draw_mermaid()``.
2. Validation that the input *is* a compiled graph (clear error
   instead of an opaque ``AttributeError`` when someone passes the
   uncompiled ``StateGraph``).
3. An ``expand_subgraphs`` knob that maps to LangGraph's ``xray``
   subgraph-introspection flag — useful for orchestration-heavy
   agents where the supervisor's structure is opaque without it.

Live execution overlay (``__node_entered__`` / ``__node_exited__``
SSE events) is tracked as a separate follow-up; this PR ships the
static side only.
"""

from __future__ import annotations

from typing import Any, Literal

GraphFormat = Literal["mermaid", "ascii"]
"""Output formats accepted by :func:`print_graph`.

* ``"mermaid"`` — text suitable for GitHub / MkDocs Mermaid blocks
  and `mermaid.live <https://mermaid.live>`_. No extra dependency.
* ``"ascii"`` — plaintext box-and-arrow rendering. Requires
  ``grandalf`` at runtime (LangChain Core delegates to it). Install
  via ``pip install grandalf`` or the kit's optional ``viz`` extra
  if/when one is added — for now it's expected to be installed
  alongside any tooling that wants ASCII output.
"""


def print_graph(
    graph: Any,
    *,
    format: GraphFormat = "mermaid",  # noqa: A002 - "format" is the natural keyword for output format
    expand_subgraphs: bool = False,
    with_styles: bool = True,
) -> str:
    """Render *graph*'s static structure and return the markup.

    Parameters
    ----------
    graph:
        A LangGraph ``CompiledStateGraph`` (or anything else with a
        ``get_graph()`` method that returns a LangChain Core
        ``Graph``). The kit's ``build_*_agent`` functions return
        compatible objects.
    format:
        See :data:`GraphFormat`.
    expand_subgraphs:
        Pass ``True`` to recurse into compiled subgraphs (LangGraph's
        ``xray=True``). Useful for supervisor / orchestration agents
        where the top-level node hides delegation structure. Defaults
        to ``False`` so a "what does this agent look like?" rendering
        stays small.
    with_styles:
        ``mermaid`` only — toggles the default node-coloring CSS
        injected by LangChain Core. Pass ``False`` when embedding in
        a context that supplies its own theme. Ignored for ASCII.

    Returns
    -------
    str
        The markup. ``"flowchart TD"`` (Mermaid) or a multi-line
        box-and-arrow ASCII drawing.

    Raises
    ------
    TypeError
        If *graph* lacks a ``get_graph`` method — almost certainly
        means an uncompiled ``StateGraph`` was passed instead of the
        compiled output.
    ValueError
        If *format* isn't one of :data:`GraphFormat`'s values.
    """
    if not hasattr(graph, "get_graph"):
        msg = (
            f"print_graph expected a compiled graph (with .get_graph()); "
            f"got {type(graph).__name__}. Did you forget to call .compile()?"
        )
        raise TypeError(msg)

    drawable = graph.get_graph(xray=expand_subgraphs)

    if format == "mermaid":
        return drawable.draw_mermaid(with_styles=with_styles)
    if format == "ascii":
        return drawable.draw_ascii()

    msg = f"Unsupported format {format!r}; expected one of: mermaid, ascii"
    raise ValueError(msg)


__all__ = ["GraphFormat", "print_graph"]
