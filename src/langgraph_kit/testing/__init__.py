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
- :class:`FakeCheckpointer` — :class:`InMemorySaver` subclass with
  ``dump_state`` and ``assert_thread_has_messages`` for tests that
  need to inspect what a graph wrote.
- :func:`scripted_llm`, :func:`tool_call_turn`,
  :func:`multi_tool_call_turn`, :func:`answer` — builders for the
  :class:`RecordedChatModel` so tests can drive a graph
  deterministically without touching a real LLM.
- :func:`assert_tool_invoked`, :func:`last_ai_message` — assertions
  for inspecting the resulting ``state["messages"]``.
- :func:`assert_namespace_contains`, :func:`assert_namespace_empty` —
  assertions for verifying what was written to a Store namespace.

Pytest plugin: when this kit is installed in a downstream project,
the ``langgraph_kit.testing.pytest_plugin`` module is auto-discovered
via :pep:`517` ``[project.entry-points.pytest11]`` and exposes
``fake_store``, ``fake_checkpointer``, and ``scripted_llm_factory``
as fixtures with no explicit conftest wiring required.
"""

from __future__ import annotations

from langgraph_kit.testing.assertions import (
    assert_namespace_contains,
    assert_namespace_empty,
)
from langgraph_kit.testing.checkpointer import FakeCheckpointer
from langgraph_kit.testing.fakes import FakeItem, FakeStore
from langgraph_kit.testing.llm import (
    answer,
    assert_tool_invoked,
    last_ai_message,
    multi_tool_call_turn,
    scripted_llm,
    tool_call_turn,
)

__all__ = [
    "FakeCheckpointer",
    "FakeItem",
    "FakeStore",
    "answer",
    "assert_namespace_contains",
    "assert_namespace_empty",
    "assert_tool_invoked",
    "last_ai_message",
    "multi_tool_call_turn",
    "scripted_llm",
    "tool_call_turn",
]
