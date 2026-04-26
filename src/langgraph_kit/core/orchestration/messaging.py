"""Per-agent message queue for multi-agent communication.

A :class:`AgentMailbox` lets one agent drop a message into another
agent's inbox without a direct call — the recipient picks it up the
next time it polls. Messages are typed (:class:`AgentMessage`),
FIFO-ordered per recipient, and persist via the same Store the rest
of the kit uses, so message delivery survives restarts and
multi-worker setups (subject to the Store backend's own consistency).

Usage::

    mailbox = AgentMailbox(store)

    # Agent A:
    await mailbox.asend(
        AgentMessage(
            sender="agent-a",
            recipient="agent-b",
            kind="info",
            payload={"observation": "task X is complete"},
        )
    )

    # Agent B (later, in its own run):
    inbox = await mailbox.arecv("agent-b")
    for msg in inbox:
        ...handle...
        await mailbox.amark_read("agent-b", msg.id)

The "negotiation" kinds (``propose`` / ``accept`` / ``reject``) are
declared on :class:`AgentMessage` so callers can build cooperative
protocols on top, but the helper sugar around them (proposal state
machine, expiry, etc.) is deferred to a follow-up issue — pure
message passing is the primitive everyone needs.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


INBOX_NAMESPACE_PREFIX = ("agent_inbox",)
"""Top-level Store-namespace prefix; keyed by recipient agent_id."""


def _inbox_namespace(agent_id: str) -> tuple[str, ...]:
    return (*INBOX_NAMESPACE_PREFIX, agent_id)


def _message_key(message: AgentMessage) -> str:
    """Time-prefixed key so :py:meth:`AgentMailbox.arecv` can sort FIFO.

    Microsecond resolution so two messages dropped in the same call
    can't collide; the message id (a uuid4) is the tie-breaker if
    two messages somehow share the exact same microsecond stamp.
    """
    return f"{message.created_at.timestamp():.6f}_{message.id}"


MessageKind = Literal["info", "propose", "accept", "reject"]
"""Valid values for :pyattr:`AgentMessage.kind`.

Negotiation kinds are declared up-front so message recipients can
pattern-match on them, but only ``info`` has fully-specified
semantics in v1 — the propose/accept/reject state machine is
deferred. Senders may use the negotiation kinds today; recipients
should treat unknown ``kind`` interactions defensively.
"""


class AgentMessage(BaseModel):
    """A single inter-agent message.

    Frozen so a message handed off via mailbox is the exact message
    the recipient sees — sender-side mutation after ``asend`` would
    be a footgun (the wire format is set at send time, but the
    in-process object would diverge).
    """

    model_config = {"frozen": True}

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Stable identifier; used as the :pyattr:`in_reply_to` target by
    follow-up messages and as the :py:meth:`AgentMailbox.amark_read`
    key."""

    sender: str
    """Sending agent's id. Free-form; the mailbox doesn't validate
    that the sender exists. Use any stable identifier scheme that
    makes sense for your deployment."""

    recipient: str
    """Receiving agent's id. The message lives in the inbox keyed by
    this value until :py:meth:`AgentMailbox.amark_read` removes it."""

    kind: MessageKind = "info"

    payload: dict[str, Any] = Field(default_factory=dict)
    """Free-form JSON-serializable payload. The mailbox doesn't
    interpret it — handlers do. For typed payloads, validate against
    a Pydantic model on receipt."""

    in_reply_to: str | None = None
    """Optional :pyattr:`AgentMessage.id` of the message this one is
    responding to. Used by callers to thread conversations; the
    mailbox itself doesn't enforce reply-chain integrity."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    """Send-side wall-clock timestamp. The mailbox uses this for FIFO
    ordering on the recipient side."""


class AgentMailbox:
    """Store-backed per-agent inbox primitive.

    One mailbox instance can serve all agents in the same Store —
    it's a thin wrapper over namespaced reads/writes. Construct with
    the same ``store`` the rest of the kit uses; the namespace
    layout is internal.
    """

    _PAGE_SIZE = 100
    """Maximum items pulled per ``asearch`` call. Recipients with
    deeper inboxes drain across multiple pages."""

    def __init__(self, store: Any) -> None:
        super().__init__()
        self._store = store

    async def asend(self, message: AgentMessage) -> str:
        """Drop *message* into ``message.recipient``'s inbox.

        Returns the message id. The recipient sees the message on the
        next :py:meth:`arecv` call; there's no push notification
        (recipients are responsible for polling at whatever cadence
        their workflow needs).
        """
        ns = _inbox_namespace(message.recipient)
        key = _message_key(message)
        await self._store.aput(ns, key, message.model_dump(mode="json"))
        logger.debug(
            "Message %s sent: %s -> %s (kind=%s)",
            message.id,
            message.sender,
            message.recipient,
            message.kind,
        )
        return message.id

    async def arecv(
        self,
        agent_id: str,
        *,
        limit: int | None = None,
        mark_read: bool = False,
    ) -> list[AgentMessage]:
        """Read *agent_id*'s inbox in FIFO order.

        ``limit`` caps the returned count (the rest stay queued).
        ``mark_read=True`` deletes returned messages from the inbox
        atomically per-message (set False to peek; useful for
        idempotent handlers that want to ack only after side effects).

        Pages through the Store so deep inboxes drain fully instead
        of silently truncating at ``_PAGE_SIZE``.
        """
        ns = _inbox_namespace(agent_id)
        result: list[AgentMessage] = []
        while True:
            batch = await self._store.asearch(ns, limit=self._PAGE_SIZE)
            if not batch:
                break
            batch.sort(key=lambda x: x.key)  # pyright: ignore[reportUnknownLambdaType,reportUnknownMemberType]
            for raw in batch:
                if limit is not None and len(result) >= limit:
                    break
                try:
                    msg = AgentMessage.model_validate(raw.value)
                except Exception:
                    logger.warning(
                        "Skipping malformed inbox message for %s: %s",
                        agent_id,
                        raw.key,
                    )
                    continue
                result.append(msg)
                if mark_read:
                    await self._store.adelete(ns, raw.key)
            if limit is not None and len(result) >= limit:
                break
            if len(batch) < self._PAGE_SIZE:
                break
        logger.debug(
            "Drained %d messages for agent %s (mark_read=%s)",
            len(result),
            agent_id,
            mark_read,
        )
        return result

    async def amark_read(self, agent_id: str, message_id: str) -> bool:
        """Delete one message from *agent_id*'s inbox by id.

        Returns ``True`` if a message was found and deleted, ``False``
        if no such message exists in the inbox (already-read,
        wrong recipient, or never sent). The caller decides whether a
        ``False`` return is interesting (idempotent ack) or a bug
        (double-delivery detection).
        """
        ns = _inbox_namespace(agent_id)
        # The store key embeds the message timestamp + id; we have to
        # search to find it because the timestamp prefix isn't known
        # to the caller.
        batch = await self._store.asearch(ns, limit=self._PAGE_SIZE)
        for raw in batch:
            value = raw.value if hasattr(raw, "value") else raw
            if isinstance(value, dict) and value.get("id") == message_id:
                await self._store.adelete(ns, raw.key)
                return True
        return False

    async def acount(self, agent_id: str) -> int:
        """Return the count of messages currently in *agent_id*'s inbox.

        Useful for "should I poll?" decisions and metrics. Issues a
        single wide ``asearch`` call rather than paging — we expect
        inboxes to be O(tens) in normal operation; multi-thousand
        inboxes signal upstream backpressure issues anyway.
        """
        ns = _inbox_namespace(agent_id)
        # ``asearch`` accepts a large limit fine on the in-memory and
        # postgres backends in the kit's current dependency set.
        items = await self._store.asearch(ns, limit=10_000)
        return len(items)


__all__ = [
    "INBOX_NAMESPACE_PREFIX",
    "AgentMailbox",
    "AgentMessage",
    "MessageKind",
]
