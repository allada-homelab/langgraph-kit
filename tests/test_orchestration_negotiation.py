"""Tests for ``langgraph_kit.core.orchestration.negotiation`` (issue #20)."""

from __future__ import annotations

from typing import Any

import pytest

from langgraph_kit.core.orchestration.messaging import AgentMailbox
from langgraph_kit.core.orchestration.negotiation import accept, propose, reject


@pytest.mark.asyncio
async def test_propose_threads_through_mailbox(mock_store: Any) -> None:
    """A propose() lands in the recipient's inbox with the right shape."""
    mailbox = AgentMailbox(mock_store)
    proposal_id = await propose(
        mailbox,
        sender="agent-a",
        recipient="agent-b",
        action="merge_branch",
        terms={"branch": "feature/x"},
    )

    inbox = await mailbox.arecv("agent-b")
    assert len(inbox) == 1
    msg = inbox[0]
    assert msg.id == proposal_id
    assert msg.kind == "propose"
    assert msg.sender == "agent-a"
    assert msg.payload["action"] == "merge_branch"
    assert msg.payload["terms"] == {"branch": "feature/x"}


@pytest.mark.asyncio
async def test_accept_threads_back_to_proposer(mock_store: Any) -> None:
    """accept() replies to the proposer with kind=accept and in_reply_to set."""
    mailbox = AgentMailbox(mock_store)
    proposal_id = await propose(
        mailbox,
        sender="agent-a",
        recipient="agent-b",
        action="merge_branch",
    )
    inbox = await mailbox.arecv("agent-b")
    proposal = inbox[0]

    reply_id = await accept(mailbox, proposal, replier="agent-b", notes="LGTM")

    a_inbox = await mailbox.arecv("agent-a")
    assert len(a_inbox) == 1
    reply = a_inbox[0]
    assert reply.id == reply_id
    assert reply.kind == "accept"
    assert reply.in_reply_to == proposal_id
    assert reply.recipient == "agent-a"
    assert reply.payload["action"] == "merge_branch"
    assert reply.payload["notes"] == "LGTM"


@pytest.mark.asyncio
async def test_reject_requires_reason_and_threads_back(mock_store: Any) -> None:
    """reject() lands a reject message with a reason in the proposer's inbox."""
    mailbox = AgentMailbox(mock_store)
    proposal_id = await propose(
        mailbox,
        sender="agent-a",
        recipient="agent-b",
        action="merge_branch",
    )
    proposal = (await mailbox.arecv("agent-b"))[0]

    await reject(mailbox, proposal, replier="agent-b", reason="conflicts on file X")

    a_inbox = await mailbox.arecv("agent-a")
    assert len(a_inbox) == 1
    reply = a_inbox[0]
    assert reply.kind == "reject"
    assert reply.in_reply_to == proposal_id
    assert reply.payload["reason"] == "conflicts on file X"


@pytest.mark.asyncio
async def test_accept_rejects_non_propose_input(mock_store: Any) -> None:
    """accept() guards against being handed a non-propose message."""
    mailbox = AgentMailbox(mock_store)
    await propose(mailbox, sender="agent-a", recipient="agent-b", action="x")
    proposal = (await mailbox.arecv("agent-b"))[0]
    await accept(mailbox, proposal, replier="agent-b")
    accept_msg = (await mailbox.arecv("agent-a"))[0]

    # Passing the accept reply back into accept() must fail loudly.
    with pytest.raises(ValueError, match="expects a propose message"):
        await accept(mailbox, accept_msg, replier="agent-c")


@pytest.mark.asyncio
async def test_reject_rejects_non_propose_input(mock_store: Any) -> None:
    """reject() guards against being handed a non-propose message."""
    mailbox = AgentMailbox(mock_store)
    await propose(mailbox, sender="agent-a", recipient="agent-b", action="x")
    proposal = (await mailbox.arecv("agent-b"))[0]
    await accept(mailbox, proposal, replier="agent-b")
    accept_msg = (await mailbox.arecv("agent-a"))[0]

    with pytest.raises(ValueError, match="expects a propose message"):
        await reject(mailbox, accept_msg, replier="agent-c", reason="nope")
