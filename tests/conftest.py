"""Conftest for agent tests — shared fixtures, no database required.

Issue #42 promoted ``MockStore`` / ``MockItem`` to a public
``langgraph_kit.testing`` module as ``FakeStore`` / ``FakeItem``.
This conftest re-exports the legacy names so the (large) existing
test corpus keeps working without a sweeping rename. New tests
should import from ``langgraph_kit.testing`` directly.
"""

from __future__ import annotations

import pytest

from langgraph_kit.testing import FakeItem as MockItem
from langgraph_kit.testing import FakeStore as MockStore

__all__ = ["MockItem", "MockStore"]


@pytest.fixture
def mock_store() -> MockStore:
    return MockStore()


@pytest.fixture(scope="session", autouse=True)
def db() -> None:
    """No-op override of the root conftest db fixture."""
    return
