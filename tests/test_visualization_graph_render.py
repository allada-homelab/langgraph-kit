"""Tests for ``langgraph_kit.core.visualization.print_graph`` (issue #21 v1)."""

# NOTE: intentionally does NOT use ``from __future__ import annotations`` —
# LangGraph's ``StateGraph`` evaluates the TypedDict's ``Annotated`` field
# at runtime via ``typing.get_type_hints()``, which needs live type objects
# (matches the same constraint already documented in
# ``tests/test_replay_runner.py``).

from typing import Annotated, Any

import pytest
from langgraph.checkpoint.memory import (  # pyright: ignore[reportMissingImports]
    InMemorySaver,
)
from langgraph.graph import (  # pyright: ignore[reportMissingModuleSource]
    END,
    START,
    StateGraph,
)
from langgraph.graph.message import (  # pyright: ignore[reportMissingModuleSource]
    add_messages,
)
from typing_extensions import TypedDict

from langgraph_kit.core.visualization import print_graph


class _SimpleState(TypedDict):
    messages: Annotated[list[Any], add_messages]


def _build_simple_graph() -> Any:
    """Two-node toy graph: START -> alpha -> beta -> END."""

    async def alpha(state: dict, _config: Any) -> dict:
        return {"messages": []}

    async def beta(state: dict, _config: Any) -> dict:
        return {"messages": []}

    builder = StateGraph(_SimpleState)
    builder.add_node("alpha", alpha)  # pyright: ignore[reportArgumentType]
    builder.add_node("beta", beta)  # pyright: ignore[reportArgumentType]
    builder.add_edge(START, "alpha")
    builder.add_edge("alpha", "beta")
    builder.add_edge("beta", END)
    return builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Happy path — Mermaid
# ---------------------------------------------------------------------------


class TestPrintGraphMermaid:
    def test_returns_mermaid_flowchart_for_simple_graph(self) -> None:
        graph = _build_simple_graph()
        markup = print_graph(graph)
        # Modern LangChain Core wraps the diagram in a Mermaid YAML
        # frontmatter (``---\nconfig:\n...---``) followed by the
        # ``graph TD`` / ``flowchart TD`` block. Either prefix means
        # we got Mermaid markup.
        assert "graph TD" in markup or "flowchart TD" in markup, markup[:80]
        # Both node names should appear in the output.
        assert "alpha" in markup
        assert "beta" in markup

    def test_includes_styles_by_default(self) -> None:
        """``with_styles=True`` emits the default node-coloring CSS."""
        graph = _build_simple_graph()
        styled = print_graph(graph)
        unstyled = print_graph(graph, with_styles=False)
        # Styled output should be longer (extra ``classDef`` / ``class``
        # lines from LangChain Core's default styling).
        assert len(styled) > len(unstyled)

    def test_explicit_format_mermaid_works(self) -> None:
        graph = _build_simple_graph()
        explicit = print_graph(graph, format="mermaid")
        default = print_graph(graph)
        assert explicit == default


# ---------------------------------------------------------------------------
# Happy path — ASCII (skipped if grandalf isn't installed)
# ---------------------------------------------------------------------------


class TestPrintGraphAscii:
    def test_renders_ascii_when_grandalf_available(self) -> None:
        pytest.importorskip("grandalf")
        graph = _build_simple_graph()
        markup = print_graph(graph, format="ascii")
        # ASCII renderings include node names verbatim.
        assert "alpha" in markup
        assert "beta" in markup


# ---------------------------------------------------------------------------
# Validation / error paths
# ---------------------------------------------------------------------------


class TestPrintGraphValidation:
    def test_uncompiled_graph_raises_typeerror_with_helpful_message(self) -> None:
        """Passing a ``StateGraph`` (not its ``.compile()`` output) is a common mistake."""
        builder = StateGraph(_SimpleState)
        builder.add_edge(START, END)
        # Don't call .compile(); pass the builder directly.
        with pytest.raises(TypeError, match=r"forget to call \.compile"):
            print_graph(builder)

    def test_unknown_format_raises_valueerror(self) -> None:
        graph = _build_simple_graph()
        with pytest.raises(ValueError, match="Unsupported format"):
            print_graph(graph, format="dot")  # type: ignore[arg-type]

    def test_non_graphlike_object_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match=r"\.compile"):
            print_graph(object())


# ---------------------------------------------------------------------------
# expand_subgraphs flag
# ---------------------------------------------------------------------------


class TestPrintGraphSubgraphs:
    def test_expand_subgraphs_calls_get_graph_with_xray(self) -> None:
        """Verify the kwarg gets translated correctly to LangGraph's xray flag."""
        # Use a minimal stand-in instead of a real subgraph since constructing one
        # adds a lot of boilerplate; just spy on the get_graph kwargs.
        captured: dict[str, Any] = {}

        class FakeDrawable:
            def draw_mermaid(self, *, with_styles: bool = True) -> str:
                _ = with_styles
                return "graph TD\n  A --> B\n"

        class FakeGraph:
            def get_graph(self, xray: bool = False) -> FakeDrawable:
                captured["xray"] = xray
                return FakeDrawable()

        markup = print_graph(FakeGraph(), expand_subgraphs=True)
        assert "graph TD" in markup
        assert captured["xray"] is True

    def test_expand_subgraphs_default_false(self) -> None:
        captured: dict[str, Any] = {}

        class FakeDrawable:
            def draw_mermaid(self, *, with_styles: bool = True) -> str:
                _ = with_styles
                return "graph TD\n"

        class FakeGraph:
            def get_graph(self, xray: bool = False) -> FakeDrawable:
                captured["xray"] = xray
                return FakeDrawable()

        print_graph(FakeGraph())
        assert captured["xray"] is False
