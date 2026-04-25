"""Append-only audit log: who did what, to what, when.

Exports:

- :class:`AuditAction` — bounded set of audit-worthy actions.
- :class:`AuditEntry` — the immutable five-tuple.
- :class:`AuditStore` — Store-backed writer/reader.

Producers should call :meth:`AuditStore.write` rather than poking the
underlying store directly so future additions (signature chaining,
external sinks, retention policies) land in a single place.
"""

from .models import AuditAction, AuditEntry
from .store import AuditStore

__all__ = [
    "AuditAction",
    "AuditEntry",
    "AuditStore",
]
