"""Coverage — schema-versioning + lazy forward migration helpers (#34).

Tests use a private ``MigrationRegistry`` so they don't touch the
default registry that the kit's own models will eventually populate.
That isolation is the whole point of accepting a registry argument
on :func:`migrate_dict`.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from langgraph_kit.core.migrations import (
    DEFAULT_REGISTRY,
    Migration,
    MigrationRegistry,
    MissingMigrationError,
    Versioned,
    migrate_dict,
)

# ---------------------------------------------------------------------------
# Fixtures: a tiny model that evolves v1 → v2 → v3
# ---------------------------------------------------------------------------


class _NoteV3(Versioned):
    """Latest schema: v3 has ``title`` (renamed from v2's ``name``)
    and a required ``priority``.
    """

    MODEL_VERSION: ClassVar[int] = 3

    id: str
    title: str
    priority: str = "normal"


def _v1_to_v2(payload: dict[str, Any]) -> dict[str, Any]:
    """v1 had no title/name; default an empty name so v2 validates."""
    payload.setdefault("name", "")
    return payload


def _v2_to_v3(payload: dict[str, Any]) -> dict[str, Any]:
    """v2's ``name`` becomes v3's ``title``; ``priority`` default."""
    if "name" in payload:
        payload["title"] = payload.pop("name")
    payload.setdefault("priority", "normal")
    return payload


# ---------------------------------------------------------------------------
# Versioned mixin
# ---------------------------------------------------------------------------


def test_versioned_default_model_version_is_1() -> None:
    class _M(Versioned):
        MODEL_VERSION: ClassVar[int] = 1

    instance = _M()
    assert instance.model_version == 1


def test_versioned_subclass_can_override_model_version() -> None:
    class _M(Versioned):
        MODEL_VERSION: ClassVar[int] = 5

    assert _M.MODEL_VERSION == 5


# ---------------------------------------------------------------------------
# MigrationRegistry
# ---------------------------------------------------------------------------


def test_registry_register_and_get() -> None:
    reg = MigrationRegistry()
    m = Migration(_NoteV3, 1, _v1_to_v2)
    reg.register(m)
    assert reg.get(_NoteV3, 1) is m
    assert reg.get(_NoteV3, 2) is None


def test_registry_rejects_duplicate_step() -> None:
    reg = MigrationRegistry()
    reg.register(Migration(_NoteV3, 1, _v1_to_v2))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(Migration(_NoteV3, 1, _v1_to_v2))


def test_registry_clear_drops_migrations() -> None:
    reg = MigrationRegistry()
    reg.register(Migration(_NoteV3, 1, _v1_to_v2))
    reg.clear()
    assert reg.get(_NoteV3, 1) is None
    # Re-register works after clear.
    reg.register(Migration(_NoteV3, 1, _v1_to_v2))


def test_migration_rejects_invalid_source_version() -> None:
    with pytest.raises(ValueError, match="source_version"):
        Migration(_NoteV3, 0, _v1_to_v2)


# ---------------------------------------------------------------------------
# migrate_dict
# ---------------------------------------------------------------------------


def test_migrate_dict_walks_v1_to_target_through_chain() -> None:
    reg = MigrationRegistry()
    reg.register(Migration(_NoteV3, 1, _v1_to_v2))
    reg.register(Migration(_NoteV3, 2, _v2_to_v3))

    payload = {"model_version": 1, "id": "n1"}
    out = migrate_dict(_NoteV3, payload, registry=reg)

    assert out["model_version"] == 3
    assert out["title"] == ""
    assert out["priority"] == "normal"
    # Original isn't mutated.
    assert "title" not in payload
    assert payload["model_version"] == 1


def test_migrate_dict_returns_copy_when_already_at_target() -> None:
    reg = MigrationRegistry()
    payload = {"model_version": 3, "id": "x", "title": "y", "priority": "high"}
    out = migrate_dict(_NoteV3, payload, registry=reg)
    assert out == payload
    assert out is not payload  # caller can mutate without aliasing


def test_migrate_dict_treats_missing_version_as_v1() -> None:
    """Legacy data without an explicit ``model_version`` key must be
    walked from v1 — that's the whole point of "lazy on read"."""
    reg = MigrationRegistry()
    reg.register(Migration(_NoteV3, 1, _v1_to_v2))
    reg.register(Migration(_NoteV3, 2, _v2_to_v3))
    payload = {"id": "old", "name": "legacy"}  # no model_version key
    out = migrate_dict(_NoteV3, payload, registry=reg)
    assert out["model_version"] == 3
    assert out["title"] == "legacy"


def test_migrate_dict_raises_on_missing_step() -> None:
    """A gap in the chain must surface loudly so deployments don't
    silently misread old data as if it were upgraded."""
    reg = MigrationRegistry()
    reg.register(Migration(_NoteV3, 2, _v2_to_v3))  # v1 step missing
    payload = {"model_version": 1, "id": "x"}
    with pytest.raises(MissingMigrationError):
        migrate_dict(_NoteV3, payload, registry=reg)


def test_migrate_dict_then_validate_round_trip() -> None:
    """End-to-end: legacy v1 dict → migrate → model_validate succeeds
    against the latest schema."""
    reg = MigrationRegistry()
    reg.register(Migration(_NoteV3, 1, _v1_to_v2))
    reg.register(Migration(_NoteV3, 2, _v2_to_v3))

    legacy = {"id": "n1"}  # no model_version, no name, no title
    migrated = migrate_dict(_NoteV3, legacy, registry=reg)
    note = _NoteV3.model_validate(migrated)
    assert note.id == "n1"
    assert note.title == ""
    assert note.priority == "normal"
    assert note.model_version == 3


# ---------------------------------------------------------------------------
# DEFAULT_REGISTRY isolation
# ---------------------------------------------------------------------------


def test_tests_do_not_pollute_default_registry() -> None:
    """The kit-internal default registry should be untouched by these
    tests — they all pass a private registry to :func:`migrate_dict`.
    Guard against future test additions accidentally registering on
    the global one."""
    # If a future test pollutes the default registry, this guard
    # catches it before it leaks across test modules.
    assert DEFAULT_REGISTRY.get(_NoteV3, 1) is None
    assert DEFAULT_REGISTRY.get(_NoteV3, 2) is None
