"""In-memory fakes for kit-internal protocols.

A "fake" (vs a "mock") behaves like the real thing — it just lives in
process memory instead of hitting a real database. Tests that drive
kit code through fakes are exercising the same code paths as
production; only the persistence layer is swapped.
"""

from __future__ import annotations

from typing import Any


class FakeItem:
    """Shape returned by :py:meth:`FakeStore.aget` and
    :py:meth:`FakeStore.asearch`.

    Mirrors LangGraph's ``Item`` (``key`` / ``value`` / ``namespace``)
    so kit code that consumes the real Store works against this fake
    unchanged. Construct directly when a test wants to inject a
    pre-baked item (e.g. seed memory state without running the kit's
    write path).
    """

    def __init__(
        self,
        key: str,
        value: dict[str, Any],
        namespace: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.key = key
        self.value = value
        self.namespace = namespace

    def __repr__(self) -> str:
        return (
            f"FakeItem(key={self.key!r}, "
            f"namespace={self.namespace!r}, "
            f"value_keys={sorted(self.value.keys())!r})"
        )


class FakeStore:
    """Process-local ``BaseStore`` substitute for kit tests.

    Implements the subset of LangGraph's Store protocol that the kit
    actually calls: ``aput``, ``aget``, ``asearch``, ``adelete``,
    ``alist_namespaces``. Every kit module that takes a ``store``
    parameter accepts this in place of an ``InMemoryStore`` /
    ``AsyncPostgresStore`` — same shape, no I/O.

    Use cases:

    - Unit tests for memory / queue / inbox / workspace primitives
      that are Store-backed.
    - Integration tests that compose multiple Store-using subsystems
      without paying SQLite startup cost.
    - Test fixtures that need to assert on what was written
      (the ``_data`` attribute is intentionally accessible).

    Construct directly: ``store = FakeStore()``. Or use the
    matching ``mock_store`` pytest fixture in
    ``tests/conftest.py`` for backwards compatibility with the kit's
    pre-extraction tests.

    Not a substitute for the real Store:

    - No persistence across process restarts.
    - No vector-similarity search (``asearch`` returns insertion
      order; LangChain's real backends rank by query similarity
      when an embedding fn is configured).
    - No transaction semantics.
    """

    def __init__(self) -> None:
        super().__init__()
        # ``_data[namespace][key] -> value``. Public-ish for tests
        # that want to inspect raw state; renames here are a
        # breaking change.
        self._data: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
    ) -> None:
        """Store *value* under (*namespace*, *key*). Upsert semantics."""
        if namespace not in self._data:
            self._data[namespace] = {}
        self._data[namespace][key] = value

    async def aget(self, namespace: tuple[str, ...], key: str) -> FakeItem | None:
        """Return the item at (*namespace*, *key*), or ``None`` if missing."""
        val = self._data.get(namespace, {}).get(key)
        if val is None:
            return None
        return FakeItem(key=key, value=val, namespace=namespace)

    async def asearch(
        self,
        namespace: tuple[str, ...],
        query: str | None = None,  # noqa: ARG002 - protocol parity
        limit: int = 10,
    ) -> list[FakeItem]:
        """Return up to *limit* items in *namespace* (insertion order).

        ``query`` is accepted for protocol parity but ignored — the
        real ``asearch`` ranks by embedding similarity to *query*; the
        fake just returns insertion order. Tests that depend on
        similarity ranking should use a real Store + embedding fn.
        """
        return [
            FakeItem(key=k, value=v, namespace=namespace)
            for k, v in list(self._data.get(namespace, {}).items())[:limit]
        ]

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        """Drop the item at (*namespace*, *key*). No-op if missing."""
        if namespace in self._data:
            self._data[namespace].pop(key, None)

    async def alist_namespaces(self, prefix: tuple[str, ...]) -> list[tuple[str, ...]]:
        """Return every non-empty namespace that starts with *prefix*."""
        return [
            ns for ns in self._data if ns[: len(prefix)] == prefix and self._data[ns]
        ]


__all__ = ["FakeItem", "FakeStore"]
