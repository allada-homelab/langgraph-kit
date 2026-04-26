"""Pytest plugin that auto-registers kit testing fixtures.

Wired up via ``[project.entry-points.pytest11]`` in ``pyproject.toml``.
Once the kit is installed in a project's venv, ``pytest`` discovers
this plugin automatically — downstream test suites can use
``fake_store``, ``fake_checkpointer``, and ``scripted_llm_factory``
without an explicit conftest import.

Pytest discovery rule: an installed package may register entry points
under ``pytest11``; the plugin's public hooks and fixtures are then
loaded for every test session in any environment that has the
package. https://docs.pytest.org/en/stable/how-to/writing_plugins.html
"""

from __future__ import annotations

from typing import Any

import pytest

from langgraph_kit.testing.checkpointer import FakeCheckpointer
from langgraph_kit.testing.fakes import FakeStore
from langgraph_kit.testing.llm import scripted_llm


@pytest.fixture
def fake_store() -> FakeStore:
    """Per-test :class:`FakeStore`. Each test gets a fresh, empty one."""
    return FakeStore()


@pytest.fixture
def fake_checkpointer() -> FakeCheckpointer:
    """Per-test :class:`FakeCheckpointer`. Each test gets a fresh saver."""
    return FakeCheckpointer()


@pytest.fixture
def scripted_llm_factory() -> Any:
    """Return :func:`scripted_llm` so tests can build a scripted model.

    Naming distinguishes the *factory* fixture (a callable that
    builds a model from turns) from the *helper function* downstream
    code imports directly. Both are equivalent — the fixture exists
    so test code can take ``scripted_llm_factory`` as an argument and
    not need a top-level import.
    """
    return scripted_llm
