"""Demo deferred tools wired into the reference deep agent.

These are intentionally side-effect-free stubs whose only purpose is to
exercise the ``tool_search`` + ``call_deferred_tool`` discovery loop in
the reference build. Real integrations (HTTP fetch, code indexing, SQL)
belong in their own modules — clone the registration helper below and
swap the stubs out.

The trio (``web_fetch_demo``, ``code_indexer_demo``, ``db_query_demo``)
covers three discovery shapes deliberately:

- a tool with a URL-like argument (web fetch) — search by ``"web"`` /
  ``"http"``
- a tool with a query argument over a "large" catalog (code indexer) —
  search by ``"code"`` / ``"index"``
- a tool with a single structured-string argument (db query) — search
  by ``"sql"`` / ``"database"``
"""

from __future__ import annotations

from typing import Any

from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk
from langgraph_kit.core.tools.deferred import DeferredToolRegistry


async def _web_fetch_demo(url: str) -> str:
    """Stub: pretend to fetch a URL."""
    return f"[demo] would fetch: {url}"


async def _code_indexer_demo(query: str, *, limit: int = 5) -> str:
    """Stub: pretend to search a large code index."""
    return f"[demo] would search code index for: {query!r} (limit={limit})"


async def _db_query_demo(sql: str) -> str:
    """Stub: pretend to run a read-only SQL query."""
    return f"[demo] would run SQL: {sql}"


def _build_demo_capabilities() -> list[ToolCapability]:
    """Return the three demo capabilities with stable ids and tags.

    Kept tag-rich so :meth:`DeferredToolRegistry.search` returns sensible
    matches across the typical query shapes a user would attempt
    (``"web"``, ``"http"``, ``"code"``, ``"sql"``, ``"database"``).
    """
    return [
        ToolCapability(
            id="ref_web_fetch_demo",
            name="web_fetch_demo",
            description=(
                "Demo deferred tool — fetches the contents of a URL over HTTP. "
                "Returns a stub string; replace with a real HTTP fetcher in "
                "your build."
            ),
            fn=_web_fetch_demo,
            tags=["web", "http", "fetch", "network"],
            risk=ToolRisk.READ_ONLY,
        ),
        ToolCapability(
            id="ref_code_indexer_demo",
            name="code_indexer_demo",
            description=(
                "Demo deferred tool — searches a large code index by keyword "
                "and returns matching snippets. Returns a stub string; "
                "replace with a real indexer in your build."
            ),
            fn=_code_indexer_demo,
            tags=["code", "index", "search", "grep"],
            risk=ToolRisk.READ_ONLY,
        ),
        ToolCapability(
            id="ref_db_query_demo",
            name="db_query_demo",
            description=(
                "Demo deferred tool — runs a read-only SQL query against a "
                "database. Returns a stub string; replace with a real DB "
                "connector in your build."
            ),
            fn=_db_query_demo,
            tags=["sql", "database", "query", "data"],
            risk=ToolRisk.READ_ONLY,
        ),
    ]


def register_reference_deferred_tools(deferred: DeferredToolRegistry) -> None:
    """Populate the deferred catalog with the reference demo tools."""
    deferred.register_many(_build_demo_capabilities())


def make_reference_deferred_configurator(
    extra: Any | None = None,
) -> Any:
    """Build a ``configure_deferred_tools=`` callback for the reference build.

    The callback registers the default demo tools first, then runs the
    optional ``extra`` callback. Because :class:`DeferredToolRegistry` is
    keyed by capability id, anything ``extra`` registers under one of
    the demo ids will override the demo — keeping the documented
    "caller wins on collisions" precedence.
    """

    def _configure(deferred: DeferredToolRegistry) -> None:
        register_reference_deferred_tools(deferred)
        if extra is not None:
            extra(deferred)

    return _configure
