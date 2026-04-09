"""Conftest for agent tests — shared fixtures, no database required."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest


class MockItem:
    """Mimics a LangGraph Store Item."""

    def __init__(
        self, key: str, value: dict[str, Any], namespace: tuple[str, ...]
    ) -> None:
        self.key = key
        self.value = value
        self.namespace = namespace


class MockStore:
    """In-memory mock of LangGraph BaseStore for testing."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}

    async def aput(
        self, namespace: tuple[str, ...], key: str, value: dict[str, Any]
    ) -> None:
        if namespace not in self._data:
            self._data[namespace] = {}
        self._data[namespace][key] = value

    async def aget(self, namespace: tuple[str, ...], key: str) -> MockItem | None:
        val = self._data.get(namespace, {}).get(key)
        if val is None:
            return None
        return MockItem(key=key, value=val, namespace=namespace)

    async def asearch(
        self,
        namespace: tuple[str, ...],
        query: str | None = None,
        limit: int = 10,
    ) -> list[MockItem]:
        return [
            MockItem(key=k, value=v, namespace=namespace)
            for k, v in list(self._data.get(namespace, {}).items())[:limit]
        ]

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        if namespace in self._data:
            self._data[namespace].pop(key, None)

    async def alist_namespaces(self, prefix: tuple[str, ...]) -> list[tuple[str, ...]]:
        return [
            ns for ns in self._data if ns[: len(prefix)] == prefix and self._data[ns]
        ]


@pytest.fixture
def mock_store() -> MockStore:
    return MockStore()


@pytest.fixture(scope="session", autouse=True)
def db() -> Generator[None]:
    """No-op override of the root conftest db fixture."""
    return None
