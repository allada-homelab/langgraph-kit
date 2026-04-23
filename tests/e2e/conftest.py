"""E2E-layer conftest — scoped fixtures and warning filters.

Everything here only applies to tests under ``tests/e2e/`` (pytest's
conftest scope is directory-relative). Unit tests above this directory
are unaffected.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest
from langgraph.checkpoint.memory import (  # pyright: ignore[reportMissingImports]
    InMemorySaver,
)

# ``mock_store`` is defined in the root conftest at ``tests/conftest.py``
# and is auto-discovered by pytest for fixtures in subdirectories. We
# re-expose it under the name ``e2e_store`` so e2e tests read more
# clearly and so we have a dedicated hook if the two ever need to
# diverge (e.g. populated seed data per e2e scenario).


@pytest.fixture(autouse=True)
def _silence_deepagents_deprecations() -> Iterator[None]:
    """Filter deepagents v0.6 DeprecationWarnings that fire during real graph invocation.

    The kit currently passes a callable backend factory to
    ``create_deep_agent`` and constructs ``StateBackend(runtime)`` with
    the runtime arg — both deprecated in deepagents v0.6 and slated for
    removal in v0.7. Migrating the kit is outside the scope of the e2e
    plan's "Not a refactor" discipline (see ``TESTING_ROADMAP.md``
    Phase 4e); unit tests never tripped these warnings because they
    mocked ``create_deep_agent`` and never actually invoked the graph.

    This fixture is autouse so every e2e test inherits the filter
    without boilerplate.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, module=r"deepagents\..*"
        )
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=r"langgraph_kit\.core\.graph_builder\.backend",
        )
        yield


@pytest.fixture
def checkpointer() -> InMemorySaver:
    """Per-test in-memory checkpointer. Each test gets a fresh one."""
    return InMemorySaver()


@pytest.fixture
def e2e_store(mock_store: Any) -> Any:
    """Alias the root-conftest MockStore as a dedicated e2e fixture.

    Distinct name so the purpose reads clearly in e2e scenarios and so
    we have an obvious hook if e2e ever needs a seeded variant.
    """
    return mock_store


@pytest.fixture
def patched_build_llm() -> Any:
    """Return a helper that patches ``build_llm`` for the caller's ``with`` block.

    Usage::

        def test_thing(patched_build_llm, ...):
            scripted = scripted_llm([...])
            with patched_build_llm(scripted):
                graph, _ = build_reference_deep_agent(...)
            result = await graph.ainvoke(...)

    The patch MUST be applied at build time (``build_deep_agent``
    resolves ``build_llm`` eagerly during graph construction). Runtime
    patching would be too late. Exiting the ``with`` restores the real
    builder, but the compiled graph still holds the scripted model.
    """

    def _patch(llm: Any) -> Any:
        return patch("langgraph_kit.graphs._builder.build_llm", return_value=llm)

    return _patch
