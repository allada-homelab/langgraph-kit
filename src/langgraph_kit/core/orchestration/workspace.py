"""Shared workspace primitive for multi-agent coordination.

An :class:`AgentWorkspace` is a typed Pydantic document that lives in
a dedicated Store namespace and supports concurrent reads + patches
with optimistic-concurrency semantics. Two agents holding handles to
the same workspace can both read the document and submit patches; the
last writer to *land* a patch wins, but stale patches (built against a
revision that's already been superseded) raise
:class:`WorkspaceConflict` so the caller can re-read and retry instead
of silently overwriting another agent's work.

Usage::

    class TaskBoard(BaseModel):
        todo: list[str] = []
        done: list[str] = []

    workspace = AgentWorkspace(store, "task-board-1", TaskBoard)
    await workspace.aput(TaskBoard(todo=["x", "y"]))

    # Agent A:
    doc, rev = await workspace.aget_with_revision()
    doc.done.append(doc.todo.pop(0))
    await workspace.apatch(doc, expected_revision=rev)

    # Agent B (concurrent):
    doc, rev = await workspace.aget_with_revision()
    doc.todo.append("z")
    await workspace.apatch(doc, expected_revision=rev)  # may raise WorkspaceConflict if A landed first

Scope (issue #20 v1):

- Workspace document, optimistic concurrency via revision counter.
- Cross-thread sharing within a single tenant (per the open question
  in the issue, multi-tenant scoping is deferred).

Deferred to follow-ups:

- Change feed / ``asubscribe`` async iterator (polling-based; not
  hard but its own ergonomics decision).
- Workspace-scoped negotiation primitives (propose/accept/reject) —
  state-machine sugar on top of the doc, separable.
"""

from __future__ import annotations

import logging
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)


WORKSPACE_NAMESPACE_PREFIX = ("workspace",)
"""Top-level Store-namespace prefix; tenant scoping is the caller's job."""

_DOC_KEY = "doc"
"""Single key inside the namespace — the workspace IS the document."""

DEFAULT_RETRY_BUDGET = 5
"""How many ``apatch`` retries before giving up on stale revisions.

A handful of retries handles natural contention; if more are needed
the workspace is being mis-used as a high-throughput message bus
(use :class:`AgentMailbox` for that).
"""


T = TypeVar("T", bound=BaseModel)


def _workspace_namespace(workspace_id: str) -> tuple[str, ...]:
    return (*WORKSPACE_NAMESPACE_PREFIX, workspace_id)


class WorkspaceConflict(Exception):
    """Raised when a patch's expected revision doesn't match the live one.

    The caller should re-read with :py:meth:`AgentWorkspace.aget_with_revision`,
    rebase its changes onto the fresh document, and retry. The
    :py:meth:`AgentWorkspace.apatch_with_retry` helper does this loop
    automatically up to :data:`DEFAULT_RETRY_BUDGET` attempts.
    """


class AgentWorkspace(Generic[T]):
    """Typed shared document for multi-agent coordination.

    Each workspace stores a single Pydantic document under the namespace
    ``("workspace", workspace_id)`` at key ``"doc"``. The Store handles
    persistence; this class adds:

    1. Pydantic schema enforcement on read and write.
    2. A monotonically-incrementing ``_revision`` counter on the
       wire-format dict (not on the model class itself, so callers
       don't have to thread it through their schema).
    3. Optimistic-concurrency ``apatch`` that refuses stale writes.

    Workspaces are cheap; create one per coordination context (a
    shared task board, a shared draft document, etc.). The schema
    type is parameterized so callers get typed reads — e.g.
    ``AgentWorkspace[TaskBoard]``.
    """

    def __init__(
        self,
        store: Any,
        workspace_id: str,
        schema: type[T],
    ) -> None:
        super().__init__()
        self._store = store
        self._workspace_id = workspace_id
        self._schema = schema
        self._ns = _workspace_namespace(workspace_id)

    @property
    def workspace_id(self) -> str:
        return self._workspace_id

    async def aget(self) -> T | None:
        """Return the document, or ``None`` if the workspace doesn't exist yet.

        Use :py:meth:`aget_with_revision` instead when you intend to
        write back — the revision is required for safe concurrent
        updates.
        """
        result = await self.aget_with_revision()
        return result[0] if result is not None else None

    async def aget_with_revision(self) -> tuple[T, int] | None:
        """Read the document and its current revision number.

        Returns ``None`` if the workspace doesn't exist (no
        :py:meth:`aput` has run yet). Use the returned revision in the
        next :py:meth:`apatch` call to detect concurrent writes.
        """
        raw = await self._store.aget(self._ns, _DOC_KEY)
        if raw is None:
            return None
        value = raw.value if hasattr(raw, "value") else raw
        if not isinstance(value, dict):
            msg = (
                f"Workspace {self._workspace_id!r} contains non-dict "
                f"payload (got {type(value).__name__}); cannot decode"
            )
            raise TypeError(msg)
        revision = int(value.get("_revision", 0))
        # Strip the metadata before validating so the user's schema
        # doesn't have to declare ``_revision``.
        payload = {k: v for k, v in value.items() if k != "_revision"}
        return self._schema.model_validate(payload), revision

    async def aput(self, doc: T) -> int:
        """Overwrite the workspace document. Returns the new revision number.

        Use this for the initial write (when no document exists) or
        when you explicitly want to discard concurrent writes (rare —
        prefer :py:meth:`apatch` for normal updates so concurrent
        edits surface as conflicts instead of silently dropping).
        """
        existing = await self.aget_with_revision()
        next_revision = (existing[1] + 1) if existing is not None else 1
        await self._write(doc, next_revision)
        return next_revision

    async def apatch(self, doc: T, *, expected_revision: int) -> int:
        """Replace the document iff the live revision matches *expected_revision*.

        Raises :class:`WorkspaceConflict` if the revision moved on
        between the caller's read and this write. Returns the new
        revision number on success.
        """
        existing = await self.aget_with_revision()
        live_revision = existing[1] if existing is not None else 0
        if live_revision != expected_revision:
            msg = (
                f"Workspace {self._workspace_id!r} revision moved "
                f"from {expected_revision} to {live_revision} "
                f"between read and patch"
            )
            raise WorkspaceConflict(msg)
        next_revision = live_revision + 1
        await self._write(doc, next_revision)
        return next_revision

    async def apatch_with_retry(
        self,
        mutate: Any,
        *,
        retries: int = DEFAULT_RETRY_BUDGET,
    ) -> tuple[T, int]:
        """Read-modify-write loop with automatic retries on conflict.

        ``mutate`` is a callable receiving the current document and
        returning the modified document. The callable runs at most
        ``retries`` times; on persistent conflict raises the last
        :class:`WorkspaceConflict`.

        Use this for "increment a counter," "append to a list," and
        similar idempotent patches. Don't use it for patches with
        side effects — the mutate callable may run multiple times.
        """
        last_error: WorkspaceConflict | None = None
        for _ in range(retries):
            current = await self.aget_with_revision()
            if current is None:
                msg = (
                    f"Workspace {self._workspace_id!r} doesn't exist; "
                    f"call aput() before apatch_with_retry()"
                )
                raise WorkspaceConflict(msg)
            doc, revision = current
            new_doc = mutate(doc)
            try:
                new_revision = await self.apatch(
                    new_doc, expected_revision=revision
                )
            except WorkspaceConflict as exc:
                last_error = exc
                continue
            return new_doc, new_revision
        # Reaching here means every attempt collided — surface the last
        # conflict so the caller can decide whether to escalate or
        # widen the retry budget.
        assert last_error is not None  # noqa: S101 - loop can't exit cleanly otherwise
        raise last_error

    async def _write(self, doc: T, revision: int) -> None:
        payload = doc.model_dump(mode="json")
        # Keep ``_revision`` out of the user schema so callers don't
        # have to model it; merge it in only on the wire format.
        payload["_revision"] = revision
        await self._store.aput(self._ns, _DOC_KEY, payload)
        logger.debug(
            "Workspace %s written (revision=%d)",
            self._workspace_id,
            revision,
        )

    async def adelete(self) -> None:
        """Delete the workspace entirely. The document is gone after this."""
        await self._store.adelete(self._ns, _DOC_KEY)


__all__ = [
    "DEFAULT_RETRY_BUDGET",
    "AgentWorkspace",
    "WorkspaceConflict",
]
