"""Tests for the public ``langgraph_kit.testing`` module (issue #42 v1)."""

from __future__ import annotations

import pytest

from langgraph_kit.testing import (
    FakeItem,
    FakeStore,
    assert_namespace_contains,
    assert_namespace_empty,
)

# ---------------------------------------------------------------------------
# Public-API surface check
# ---------------------------------------------------------------------------


class TestPublicApiSurface:
    def test_exports_are_importable(self) -> None:
        """Smoke check: every name promoted to the public API resolves."""
        from langgraph_kit import testing

        for name in (
            "FakeItem",
            "FakeStore",
            "assert_namespace_contains",
            "assert_namespace_empty",
        ):
            assert hasattr(testing, name), f"missing public export: {name}"

    def test_all_matches_actual_exports(self) -> None:
        """``__all__`` should describe what the module actually exposes."""
        from langgraph_kit import testing

        for name in testing.__all__:
            assert hasattr(testing, name), f"__all__ promises {name!r} but it's missing"


# ---------------------------------------------------------------------------
# FakeStore — protocol parity with the real Store
# ---------------------------------------------------------------------------


class TestFakeStore:
    @pytest.mark.asyncio
    async def test_aput_then_aget_round_trip(self) -> None:
        store = FakeStore()
        await store.aput(("ns",), "k", {"x": 1})
        item = await store.aget(("ns",), "k")
        assert item is not None
        assert isinstance(item, FakeItem)
        assert item.key == "k"
        assert item.value == {"x": 1}
        assert item.namespace == ("ns",)

    @pytest.mark.asyncio
    async def test_aget_missing_returns_none(self) -> None:
        store = FakeStore()
        assert await store.aget(("ns",), "missing") is None

    @pytest.mark.asyncio
    async def test_asearch_returns_insertion_order(self) -> None:
        """The fake doesn't rank by similarity — insertion order only."""
        store = FakeStore()
        for i in range(3):
            await store.aput(("ns",), f"k{i}", {"i": i})
        items = await store.asearch(("ns",))
        assert [item.value["i"] for item in items] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_asearch_respects_limit(self) -> None:
        store = FakeStore()
        for i in range(5):
            await store.aput(("ns",), f"k{i}", {"i": i})
        items = await store.asearch(("ns",), limit=2)
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_adelete_removes_item(self) -> None:
        store = FakeStore()
        await store.aput(("ns",), "k", {"x": 1})
        await store.adelete(("ns",), "k")
        assert await store.aget(("ns",), "k") is None

    @pytest.mark.asyncio
    async def test_adelete_missing_is_noop(self) -> None:
        store = FakeStore()
        await store.adelete(("ns",), "no-such-key")  # must not raise

    @pytest.mark.asyncio
    async def test_alist_namespaces_returns_matching_prefixes(self) -> None:
        store = FakeStore()
        await store.aput(("workspace", "a"), "doc", {"x": 1})
        await store.aput(("workspace", "b"), "doc", {"x": 2})
        await store.aput(("inbox", "a"), "msg", {"text": "hi"})
        ws_namespaces = await store.alist_namespaces(("workspace",))
        assert sorted(ws_namespaces) == [("workspace", "a"), ("workspace", "b")]
        # Other prefix doesn't leak.
        inbox_namespaces = await store.alist_namespaces(("inbox",))
        assert inbox_namespaces == [("inbox", "a")]

    @pytest.mark.asyncio
    async def test_alist_namespaces_skips_empty_namespaces(self) -> None:
        """A namespace whose only item was deleted shouldn't appear."""
        store = FakeStore()
        await store.aput(("workspace", "x"), "doc", {})
        await store.adelete(("workspace", "x"), "doc")
        # Empty namespace remains in ``_data`` as an empty dict; alist
        # should still skip it because nothing's stored there now.
        result = await store.alist_namespaces(("workspace",))
        assert result == []


# ---------------------------------------------------------------------------
# FakeItem
# ---------------------------------------------------------------------------


class TestFakeItem:
    def test_construct_directly(self) -> None:
        """Tests can mint pre-baked items without running the kit's write path."""
        item = FakeItem(key="k", value={"x": 1}, namespace=("ns",))
        assert item.key == "k"
        assert item.value == {"x": 1}
        assert item.namespace == ("ns",)

    def test_repr_includes_value_keys_only(self) -> None:
        """Repr summarizes value keys (don't dump full payload into logs)."""
        item = FakeItem(
            key="k", value={"secret": "super-private", "id": 1}, namespace=("ns",)
        )
        rep = repr(item)
        assert "key='k'" in rep
        assert "namespace=('ns',)" in rep
        # Value keys are listed; values are NOT.
        assert "'id'" in rep
        assert "'secret'" in rep
        assert "super-private" not in rep


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


class TestAssertNamespaceContains:
    @pytest.mark.asyncio
    async def test_returns_value_dict_on_match(self) -> None:
        store = FakeStore()
        await store.aput(("ns",), "k", {"title": "preferences"})
        value = await assert_namespace_contains(store, ("ns",))
        assert value == {"title": "preferences"}

    @pytest.mark.asyncio
    async def test_with_predicate_finds_matching_item(self) -> None:
        store = FakeStore()
        await store.aput(("ns",), "k1", {"title": "color", "value": "blue"})
        await store.aput(("ns",), "k2", {"title": "size", "value": "L"})
        match = await assert_namespace_contains(
            store, ("ns",), where=lambda v: v["title"] == "size"
        )
        assert match["value"] == "L"

    @pytest.mark.asyncio
    async def test_empty_namespace_raises(self) -> None:
        store = FakeStore()
        with pytest.raises(AssertionError, match="empty"):
            await assert_namespace_contains(store, ("ns",))

    @pytest.mark.asyncio
    async def test_no_predicate_match_raises(self) -> None:
        store = FakeStore()
        await store.aput(("ns",), "k", {"title": "color"})
        with pytest.raises(AssertionError, match=r"No item.*satisfies"):
            await assert_namespace_contains(
                store, ("ns",), where=lambda v: v["title"] == "missing"
            )

    @pytest.mark.asyncio
    async def test_description_threaded_into_error_message(self) -> None:
        store = FakeStore()
        with pytest.raises(AssertionError, match="user's preferences memory"):
            await assert_namespace_contains(
                store, ("ns",), description="user's preferences memory"
            )


class TestAssertNamespaceEmpty:
    @pytest.mark.asyncio
    async def test_empty_namespace_passes(self) -> None:
        store = FakeStore()
        await assert_namespace_empty(store, ("ns",))  # must not raise

    @pytest.mark.asyncio
    async def test_namespace_with_item_raises(self) -> None:
        store = FakeStore()
        await store.aput(("ns",), "k", {"x": 1})
        with pytest.raises(AssertionError, match=r"Expected.*to be empty"):
            await assert_namespace_empty(store, ("ns",))

    @pytest.mark.asyncio
    async def test_description_threaded_into_error_message(self) -> None:
        store = FakeStore()
        await store.aput(("ns",), "k", {"x": 1})
        with pytest.raises(AssertionError, match="audit log"):
            await assert_namespace_empty(store, ("ns",), description="audit log")


# ---------------------------------------------------------------------------
# Backwards-compat re-exports from tests/conftest.py
# ---------------------------------------------------------------------------


class TestBackwardsCompatReexports:
    def test_old_mockstore_name_still_works(self) -> None:
        """Existing tests that import ``MockStore`` from conftest still resolve."""
        from tests.conftest import MockStore as LegacyMockStore

        assert LegacyMockStore is FakeStore

    def test_old_mockitem_name_still_works(self) -> None:
        from tests.conftest import MockItem as LegacyMockItem

        assert LegacyMockItem is FakeItem
