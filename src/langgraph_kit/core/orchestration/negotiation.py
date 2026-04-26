"""Negotiation primitives over :class:`AgentMailbox`.

Three thin helpers — :func:`propose`, :func:`accept`,
:func:`reject` — that wrap the existing mailbox so a pair of agents
can run a propose/accept-reject conversation without re-deriving the
``AgentMessage`` shape per caller. The actual delivery, FIFO order,
and persistence semantics belong to ``AgentMailbox``; this module
only fixes the message ``kind`` and the ``in_reply_to`` chain.

Usage::

    from langgraph_kit.core.orchestration.messaging import AgentMailbox
    from langgraph_kit.core.orchestration.negotiation import (
        propose, accept, reject,
    )

    mailbox = AgentMailbox(store)

    # Agent A initiates.
    proposal_id = await propose(
        mailbox,
        sender="agent-a",
        recipient="agent-b",
        action="merge_branch",
        terms={"branch": "feature/x", "into": "main"},
    )

    # Agent B receives the propose, decides, replies.
    inbox = await mailbox.arecv("agent-b")
    for msg in inbox:
        if msg.kind == "propose":
            await accept(mailbox, msg, replier="agent-b")
        await mailbox.amark_read("agent-b", msg.id)

    # Agent A reads the reply later.
    replies = await mailbox.arecv("agent-a")

The state machine (``pending`` → ``accepted | rejected``) is
implicit in the mailbox traffic — there's no separate proposal
store. Callers wanting durable proposal state should layer it on top
(e.g. via :class:`AgentWorkspace`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph_kit.core.orchestration.messaging import AgentMessage

if TYPE_CHECKING:
    from langgraph_kit.core.orchestration.messaging import AgentMailbox


async def propose(
    mailbox: AgentMailbox,
    *,
    sender: str,
    recipient: str,
    action: str,
    terms: dict[str, Any] | None = None,
    in_reply_to: str | None = None,
) -> str:
    """Send a ``propose`` message and return the proposal id.

    The proposal id IS the message id — replies thread through
    ``in_reply_to`` so receivers can match accept/reject back to the
    original propose. ``action`` is a free-form string identifying
    what's being proposed; ``terms`` is the payload the recipient
    inspects to decide.
    """
    message = AgentMessage(
        sender=sender,
        recipient=recipient,
        kind="propose",
        payload={"action": action, "terms": terms or {}},
        in_reply_to=in_reply_to,
    )
    await mailbox.asend(message)
    return message.id


async def accept(
    mailbox: AgentMailbox,
    proposal: AgentMessage,
    *,
    replier: str,
    notes: str | None = None,
) -> str:
    """Reply to *proposal* with an ``accept`` message; return its id.

    ``proposal.kind`` must be ``"propose"``; the reply's
    ``in_reply_to`` is set to ``proposal.id`` and the recipient is
    set to ``proposal.sender`` so the conversation threads back to
    the original proposer.
    """
    if proposal.kind != "propose":
        msg = (
            f"accept() expects a propose message; got kind={proposal.kind!r} "
            f"id={proposal.id!r}"
        )
        raise ValueError(msg)
    payload: dict[str, Any] = {"action": proposal.payload.get("action")}
    if notes is not None:
        payload["notes"] = notes
    reply = AgentMessage(
        sender=replier,
        recipient=proposal.sender,
        kind="accept",
        payload=payload,
        in_reply_to=proposal.id,
    )
    await mailbox.asend(reply)
    return reply.id


async def reject(
    mailbox: AgentMailbox,
    proposal: AgentMessage,
    *,
    replier: str,
    reason: str,
) -> str:
    """Reply to *proposal* with a ``reject`` message; return its id.

    ``reason`` is required — a reject without a reason wastes the
    proposer's debug-time. ``in_reply_to`` is set to ``proposal.id``
    so the proposer can match the rejection back.
    """
    if proposal.kind != "propose":
        msg = (
            f"reject() expects a propose message; got kind={proposal.kind!r} "
            f"id={proposal.id!r}"
        )
        raise ValueError(msg)
    reply = AgentMessage(
        sender=replier,
        recipient=proposal.sender,
        kind="reject",
        payload={
            "action": proposal.payload.get("action"),
            "reason": reason,
        },
        in_reply_to=proposal.id,
    )
    await mailbox.asend(reply)
    return reply.id


__all__ = [
    "accept",
    "propose",
    "reject",
]
