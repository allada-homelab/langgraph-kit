"""Schema-versioning + lazy migration for persisted Pydantic models.

See :mod:`langgraph_kit.core.migrations.versioned` for the design.
"""

from .versioned import (
    DEFAULT_REGISTRY,
    Migration,
    MigrationRegistry,
    MissingMigrationError,
    Versioned,
    migrate_dict,
)

__all__ = [
    "DEFAULT_REGISTRY",
    "Migration",
    "MigrationRegistry",
    "MissingMigrationError",
    "Versioned",
    "migrate_dict",
]
