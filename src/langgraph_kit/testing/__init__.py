"""Public testing utilities for langgraph-kit consumers.

Promotes the kit's internal pytest fixtures to a stable public API so
downstream tests don't have to copy-paste them. The names exported
here are treated as the kit's strictest API: changes go through the
deprecation cycle one major version ahead of code-path APIs.

Currently exposes:

- :class:`FakeStore` — in-memory ``BaseStore`` shim with the full
  protocol (``aput`` / ``aget`` / ``asearch`` / ``adelete`` /
  ``alist_namespaces``). Use whenever a test needs to call into
  Store-backed kit code without spinning up SQLite or Postgres.
- :class:`FakeItem` — the ``aget`` / ``asearch`` return-shape
  (``key`` / ``value`` / ``namespace``).
- :func:`assert_namespace_contains` — assertion helper for "did the
  thing-under-test write what I expected to this Store namespace?"
- :func:`assert_namespace_empty` — assertion helper for the
  no-side-effect side of the same coin.

Deferred to follow-ups (each its own small PR):

- ``FakeCheckpointer`` (thin wrapper over ``InMemorySaver`` with
  ``dump_state`` and ``assert_thread_has_messages`` affordances).
- ``scripted_llm`` / ``tool_call_turn`` / ``answer`` — the
  ``RecordedChatModel`` script builders currently in
  ``tests/e2e/conftest.py``.
- Pytest ``entry_points.pytest11`` plugin so fixtures auto-register
  on ``pytest`` invocation; today callers do
  ``from langgraph_kit.testing import FakeStore`` explicitly.
"""

from __future__ import annotations

from langgraph_kit.testing.assertions import (
    assert_namespace_contains,
    assert_namespace_empty,
)
from langgraph_kit.testing.fakes import FakeItem, FakeStore

__all__ = [
    "FakeItem",
    "FakeStore",
    "assert_namespace_contains",
    "assert_namespace_empty",
]
