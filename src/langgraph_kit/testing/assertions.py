"""Pytest-friendly assertions for kit-based tests.

Each helper is a thin wrapper over a Store-protocol read + a clear
``AssertionError`` on mismatch. The kit's tests use these directly;
downstream consumers can import them via :mod:`langgraph_kit.testing`.

Two assertion shapes for now:

- :func:`assert_namespace_contains` — at least one item in
  *namespace* matches a value-predicate.
- :func:`assert_namespace_empty` — *namespace* is empty (or doesn't
  exist).

Both work against any object that exposes the kit's read-side Store
protocol (``aget`` / ``asearch``), so they cover the real
``InMemoryStore`` / ``AsyncPostgresStore`` and :class:`FakeStore`
without branching on type. Memory- / queue- / inbox-specific
assertions are deferred — they can layer on these primitives.
"""

from __future__ import annotations

from typing import Any


async def assert_namespace_contains(
    store: Any,
    namespace: tuple[str, ...],
    *,
    where: Any = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Assert at least one item in *namespace* matches *where*.

    Returns the matching item's ``value`` dict (handy for chained
    assertions: "yes the memory was saved AND its content is X").

    Parameters
    ----------
    store:
        Any object with an ``asearch(namespace, limit=...)`` method —
        the kit's real Store backends and :class:`FakeStore` both work.
    namespace:
        Store namespace tuple (e.g. ``("workspace", "board-1")``).
    where:
        Optional predicate. ``None`` (default) matches "any item
        present"; a callable receives the item's ``value`` dict and
        should return truthy on match. Use for stronger assertions
        like ``where=lambda v: v["title"] == "preferences"``.
    description:
        Optional human-readable description for the assertion error
        message (e.g. ``"the user's preferences memory"``). Falls
        back to a generic message if omitted.

    Raises
    ------
    AssertionError
        If the namespace is empty or no item satisfies *where*.
    """
    items = await store.asearch(namespace, limit=10_000)
    if not items:
        msg = description or f"namespace {namespace!r}"
        raise AssertionError(
            f"Expected at least one item in {msg}, but the namespace is empty"
        )
    if where is None:
        return _value(items[0])
    for item in items:
        value = _value(item)
        if where(value):
            return value
    msg = description or f"namespace {namespace!r}"
    raise AssertionError(
        f"No item in {msg} satisfies the predicate. "
        f"Found {len(items)} item(s); first value: {_value(items[0])!r}"
    )


async def assert_namespace_empty(
    store: Any,
    namespace: tuple[str, ...],
    *,
    description: str | None = None,
) -> None:
    """Assert *namespace* is empty (no items, or doesn't exist).

    Useful for "the test path should NOT have written anything to
    this namespace" — common after testing error paths or
    permission-checked writes.

    Raises
    ------
    AssertionError
        If any item exists in the namespace.
    """
    items = await store.asearch(namespace, limit=1)
    if items:
        msg = description or f"namespace {namespace!r}"
        raise AssertionError(
            f"Expected {msg} to be empty, but it contains "
            f"{len(items)}+ item(s); first key: {items[0].key!r}"
        )


def _value(item: Any) -> dict[str, Any]:
    """Pull the value dict off an item, falling back to ``item`` itself.

    The real Store and :class:`FakeStore` both return wrapper objects
    with a ``.value`` attribute. Some tests pass raw dicts directly;
    fall through so callers don't need to wrap them.
    """
    return item.value if hasattr(item, "value") else item


__all__ = [
    "assert_namespace_contains",
    "assert_namespace_empty",
]
