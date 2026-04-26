"""Tests for ``langgraph_kit.core.orchestration.messaging`` (issue #20 v1)."""

from __future__ import annotations

import asyncio

import pytest

from langgraph_kit.core.orchestration.messaging import AgentMailbox, AgentMessage
from tests.conftest import MockStore


@pytest.fixture
def mock_store() -> MockStore:
    return MockStore()


# ---------------------------------------------------------------------------
# AgentMessage shape
# ---------------------------------------------------------------------------


class TestAgentMessage:
    def test_default_fields_populated(self) -> None:
        msg = AgentMessage(sender="a", recipient="b")
        assert msg.id  # uuid populated
        assert msg.kind == "info"
        assert msg.payload == {}
        assert msg.in_reply_to is None
        assert msg.created_at is not None

    def test_message_is_frozen(self) -> None:
        msg = AgentMessage(sender="a", recipient="b")
        with pytest.raises(Exception):  # noqa: B017,PT011 - Pydantic raises ValidationError on frozen
            msg.payload = {"changed": True}  # type: ignore[misc]

    def test_kind_validation_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            AgentMessage(sender="a", recipient="b", kind="bogus")  # type: ignore[arg-type]

    @pytest.mark.parametrize("kind", ["info", "propose", "accept", "reject"])
    def test_all_declared_kinds_accepted(self, kind: str) -> None:
        msg = AgentMessage(sender="a", recipient="b", kind=kind)  # type: ignore[arg-type]
        assert msg.kind == kind


# ---------------------------------------------------------------------------
# Mailbox round-trip
# ---------------------------------------------------------------------------


class TestAgentMailboxBasics:
    @pytest.mark.asyncio
    async def test_send_then_receive_round_trip(self, mock_store: MockStore) -> None:
        mailbox = AgentMailbox(mock_store)
        sent = AgentMessage(
            sender="agent-a",
            recipient="agent-b",
            payload={"observation": "x"},
        )
        sent_id = await mailbox.asend(sent)
        assert sent_id == sent.id

        received = await mailbox.arecv("agent-b")
        assert len(received) == 1
        assert received[0].id == sent.id
        assert received[0].payload == {"observation": "x"}

    @pytest.mark.asyncio
    async def test_recv_with_mark_read_drains_inbox(
        self, mock_store: MockStore
    ) -> None:
        mailbox = AgentMailbox(mock_store)
        await mailbox.asend(AgentMessage(sender="a", recipient="b"))
        first = await mailbox.arecv("b", mark_read=True)
        assert len(first) == 1
        # Second call returns empty — message was deleted.
        second = await mailbox.arecv("b")
        assert second == []

    @pytest.mark.asyncio
    async def test_recv_without_mark_read_is_a_peek(
        self, mock_store: MockStore
    ) -> None:
        mailbox = AgentMailbox(mock_store)
        await mailbox.asend(AgentMessage(sender="a", recipient="b"))
        first = await mailbox.arecv("b", mark_read=False)
        second = await mailbox.arecv("b", mark_read=False)
        assert len(first) == len(second) == 1
        assert first[0].id == second[0].id

    @pytest.mark.asyncio
    async def test_recv_for_unknown_recipient_returns_empty(
        self, mock_store: MockStore
    ) -> None:
        mailbox = AgentMailbox(mock_store)
        assert await mailbox.arecv("never-registered") == []

    @pytest.mark.asyncio
    async def test_messages_isolated_per_recipient(self, mock_store: MockStore) -> None:
        """A message to ``agent-b`` doesn't appear in ``agent-c``'s inbox."""
        mailbox = AgentMailbox(mock_store)
        await mailbox.asend(AgentMessage(sender="a", recipient="b"))
        await mailbox.asend(AgentMessage(sender="a", recipient="c"))
        b_inbox = await mailbox.arecv("b")
        c_inbox = await mailbox.arecv("c")
        assert len(b_inbox) == 1
        assert len(c_inbox) == 1
        assert b_inbox[0].recipient == "b"
        assert c_inbox[0].recipient == "c"


# ---------------------------------------------------------------------------
# Ordering + limits
# ---------------------------------------------------------------------------


class TestAgentMailboxOrdering:
    @pytest.mark.asyncio
    async def test_recv_returns_messages_in_send_order(
        self, mock_store: MockStore
    ) -> None:
        mailbox = AgentMailbox(mock_store)
        # Use distinct ids and small async gaps to ensure timestamps differ.
        for i in range(5):
            await mailbox.asend(
                AgentMessage(sender="a", recipient="b", payload={"i": i})
            )
            # Yield to let asyncio advance the clock between sends — not
            # strictly needed at microsecond resolution but cheap and
            # explicit about the intent.
            await asyncio.sleep(0)
        received = await mailbox.arecv("b")
        order = [msg.payload["i"] for msg in received]
        assert order == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_recv_limit_caps_count_and_leaves_remainder(
        self, mock_store: MockStore
    ) -> None:
        mailbox = AgentMailbox(mock_store)
        for i in range(5):
            await mailbox.asend(
                AgentMessage(sender="a", recipient="b", payload={"i": i})
            )
            await asyncio.sleep(0)
        first_two = await mailbox.arecv("b", limit=2, mark_read=True)
        assert [msg.payload["i"] for msg in first_two] == [0, 1]
        # The other three are still in the inbox.
        rest = await mailbox.arecv("b")
        assert [msg.payload["i"] for msg in rest] == [2, 3, 4]


# ---------------------------------------------------------------------------
# amark_read + acount
# ---------------------------------------------------------------------------


class TestAgentMailboxAck:
    @pytest.mark.asyncio
    async def test_mark_read_known_message_returns_true(
        self, mock_store: MockStore
    ) -> None:
        mailbox = AgentMailbox(mock_store)
        msg = AgentMessage(sender="a", recipient="b")
        await mailbox.asend(msg)
        assert await mailbox.amark_read("b", msg.id) is True
        assert await mailbox.arecv("b") == []

    @pytest.mark.asyncio
    async def test_mark_read_unknown_message_returns_false(
        self, mock_store: MockStore
    ) -> None:
        mailbox = AgentMailbox(mock_store)
        await mailbox.asend(AgentMessage(sender="a", recipient="b"))
        assert await mailbox.amark_read("b", "no-such-id") is False
        # Real message still present.
        assert len(await mailbox.arecv("b")) == 1

    @pytest.mark.asyncio
    async def test_mark_read_only_removes_target_message(
        self, mock_store: MockStore
    ) -> None:
        mailbox = AgentMailbox(mock_store)
        msg1 = AgentMessage(sender="a", recipient="b", payload={"x": 1})
        msg2 = AgentMessage(sender="a", recipient="b", payload={"x": 2})
        await mailbox.asend(msg1)
        await mailbox.asend(msg2)
        assert await mailbox.amark_read("b", msg1.id) is True
        remaining = await mailbox.arecv("b")
        assert len(remaining) == 1
        assert remaining[0].id == msg2.id

    @pytest.mark.asyncio
    async def test_acount_reflects_inbox_size(self, mock_store: MockStore) -> None:
        mailbox = AgentMailbox(mock_store)
        assert await mailbox.acount("b") == 0
        for i in range(3):
            await mailbox.asend(
                AgentMessage(sender="a", recipient="b", payload={"i": i})
            )
        assert await mailbox.acount("b") == 3
        # Drain — count drops.
        await mailbox.arecv("b", mark_read=True)
        assert await mailbox.acount("b") == 0
