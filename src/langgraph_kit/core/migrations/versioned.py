"""Schema-versioning infrastructure for persisted Pydantic models.

Issue #34. The kit persists several Pydantic models to the Store
(``MemoryRecord``, ``SessionNotebook``, ``AsyncTask``, ``MetricSummary``,
…). Schema changes today break existing Store data — fields rename
silently, removed fields raise validation errors, new required fields
have no defaults for old rows.

This module ships the minimum infrastructure to evolve those models
forward without a stop-the-world upgrade:

- :class:`Versioned` — Pydantic mixin that adds ``model_version: int``.
- :class:`Migration` — a single forward step ``v → v+1``.
- :class:`MigrationRegistry` — keyed by ``(model_class, source_version)``;
  one registered migration per step. Forward-only by design (down
  migrations are easy to lose data on and rarely actually needed).
- :func:`migrate_dict` — read-time entry point. Walks the registry
  forward until the dict is at the model's ``MODEL_VERSION``.

Usage convention:

1. Each persisted model declares ``MODEL_VERSION`` (class var) and
   inherits :class:`Versioned`.
2. When the schema changes, bump ``MODEL_VERSION`` and register a
   :class:`Migration` for the previous version.
3. Read paths call ``migrate_dict(cls, payload)`` before
   ``cls.model_validate(...)`` — old rows are upgraded transparently
   and the upgraded form is written back on the next persist.

The registry is global. Tests can swap in a private registry via
``MigrationRegistry()`` and pass it to ``migrate_dict``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable


class MissingMigrationError(LookupError):
    """No migration is registered for the requested step."""


class Versioned(BaseModel):
    """Mixin for persisted Pydantic models.

    Subclasses set ``MODEL_VERSION`` to the current schema version.
    Defaults to ``1`` so legacy rows (no ``model_version`` in the
    payload) are treated as v1 and walked through any registered
    migrations on the way to the current version.
    """

    MODEL_VERSION: ClassVar[int] = 1

    model_version: int = Field(
        default=1,
        description="Schema version of the persisted record.",
    )


class Migration:
    """A single forward-step migration ``source_version → source_version + 1``.

    The transform receives a raw dict (pre-validation) and returns a
    raw dict the next-version model will accept. Operating in dict
    space lets a migration drop, rename, or split fields that the new
    model wouldn't validate.
    """

    def __init__(
        self,
        model_cls: type[Versioned],
        source_version: int,
        transform: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        super().__init__()
        if source_version < 1:
            msg = "source_version must be >= 1"
            raise ValueError(msg)
        self.model_cls = model_cls
        self.source_version = source_version
        self.transform = transform

    def apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        new_payload = self.transform(dict(payload))
        new_payload["model_version"] = self.source_version + 1
        return new_payload


class MigrationRegistry:
    """Holds the registered migrations for one or more :class:`Versioned` models."""

    def __init__(self) -> None:
        super().__init__()
        # (model_cls, source_version) -> Migration
        self._migrations: dict[tuple[type[Versioned], int], Migration] = {}

    def register(self, migration: Migration) -> None:
        key = (migration.model_cls, migration.source_version)
        if key in self._migrations:
            msg = (
                f"Migration already registered for "
                f"{migration.model_cls.__name__} v{migration.source_version}"
            )
            raise ValueError(msg)
        self._migrations[key] = migration

    def get(self, model_cls: type[Versioned], source_version: int) -> Migration | None:
        return self._migrations.get((model_cls, source_version))

    def clear(self) -> None:
        """Drop every registered migration. Intended for tests."""
        self._migrations.clear()


# Default registry the kit's own model migrations will be registered against.
DEFAULT_REGISTRY: MigrationRegistry = MigrationRegistry()


def migrate_dict(
    model_cls: type[Versioned],
    payload: dict[str, Any],
    *,
    registry: MigrationRegistry | None = None,
) -> dict[str, Any]:
    """Walk migrations forward until *payload* reaches ``model_cls.MODEL_VERSION``.

    Returns a *new* dict; the input is not mutated. Raises
    :class:`MissingMigrationError` when a step has no registered
    migration — the caller can decide whether to fail loudly or fall
    back to best-effort decoding.
    """
    reg = registry or DEFAULT_REGISTRY
    target = model_cls.MODEL_VERSION
    current = payload.get("model_version", 1)
    if current >= target:
        # Nothing to do — and we never down-migrate.
        return dict(payload)

    walked = dict(payload)
    while current < target:
        migration = reg.get(model_cls, current)
        if migration is None:
            msg = (
                f"No migration registered for {model_cls.__name__} "
                f"v{current} → v{current + 1}"
            )
            raise MissingMigrationError(msg)
        walked = migration.apply(walked)
        current += 1
    return walked
