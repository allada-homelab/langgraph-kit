"""Tests for the per-process thread cancellation primitive."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from langgraph_kit.cancellation import (
    ThreadCancellationRegistry,
    cancel_thread,
    get_cancellation_registry,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class TestRegistryRegisterCancel:
    """Direct exercise of ``register`` / ``cancel`` / ``unregister``."""

    async def test_cancel_unknown_returns_false(self) -> None:
        registry = ThreadCancellationRegistry()
        assert registry.cancel("nonexistent") is False

    async def test_cancel_running_task_returns_true(self) -> None:
        registry = ThreadCancellationRegistry()

        async def _runner() -> None:
            await asyncio.sleep(60)

        task = asyncio.create_task(_runner())
        registry.register("t1", task)
        try:
            assert registry.cancel("t1") is True
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            if not task.done():
                task.cancel()

    async def test_cancel_completed_task_returns_false(self) -> None:
        """Re-cancellation after the task has finished is a no-op."""
        registry = ThreadCancellationRegistry()

        async def _quick() -> str:
            return "done"

        task = asyncio.create_task(_quick())
        registry.register("t1", task)
        await task  # let it finish

        assert registry.cancel("t1") is False

    async def test_unregister_is_idempotent(self) -> None:
        registry = ThreadCancellationRegistry()
        registry.unregister("never-registered")  # no-op, no raise

        async def _runner() -> None:
            await asyncio.sleep(60)

        task = asyncio.create_task(_runner())
        registry.register("t1", task)
        registry.unregister("t1")
        registry.unregister("t1")  # second call is also a no-op

        # After unregister the cancel API can't reach the task.
        assert registry.cancel("t1") is False
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_is_running_reports_state(self) -> None:
        registry = ThreadCancellationRegistry()
        assert registry.is_running("t1") is False

        async def _runner() -> None:
            await asyncio.sleep(60)

        task = asyncio.create_task(_runner())
        registry.register("t1", task)
        assert registry.is_running("t1") is True

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Task is now done; is_running should reflect that.
        assert registry.is_running("t1") is False

    async def test_register_overwrite_logs_and_replaces(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Concurrent register on the same thread_id loses the prior task."""
        import logging

        registry = ThreadCancellationRegistry()

        async def _runner() -> None:
            await asyncio.sleep(60)

        first = asyncio.create_task(_runner())
        second = asyncio.create_task(_runner())
        registry.register("t1", first)
        with caplog.at_level(logging.WARNING, logger="langgraph_kit.cancellation"):
            registry.register("t1", second)

        # The warning fires only when the prior task is still alive.
        assert any("already has a running task" in r.message for r in caplog.records)

        # Cancel reaches second, not first.
        registry.cancel("t1")
        with pytest.raises(asyncio.CancelledError):
            await second
        assert first.done() is False  # first is still running, untouched
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first


class TestTrackContextManager:
    """The ``track`` async context manager registers ``current_task``."""

    async def test_track_registers_and_unregisters(self) -> None:
        registry = ThreadCancellationRegistry()
        assert registry.is_running("t1") is False

        async with registry.track("t1"):
            assert registry.is_running("t1") is True

        assert registry.is_running("t1") is False

    async def test_track_unregisters_on_exception(self) -> None:
        registry = ThreadCancellationRegistry()

        class _Boom(RuntimeError):
            pass

        with pytest.raises(_Boom):
            async with registry.track("t1"):
                raise _Boom

        assert registry.is_running("t1") is False

    async def test_track_cancellation_propagates_to_caller(self) -> None:
        """A cancel issued from outside ``track`` raises in the tracked task."""
        registry = ThreadCancellationRegistry()
        entered = asyncio.Event()
        exited = asyncio.Event()

        async def _worker() -> None:
            try:
                async with registry.track("t1"):
                    entered.set()
                    await asyncio.sleep(60)
            except asyncio.CancelledError:
                exited.set()
                raise

        task = asyncio.create_task(_worker())
        await entered.wait()

        # Cancel via the registry — the worker's CancelledError handler fires.
        assert registry.cancel("t1") is True
        with pytest.raises(asyncio.CancelledError):
            await task
        assert exited.is_set()
        # And the registry is clean afterwards.
        assert registry.is_running("t1") is False


class TestModuleSingleton:
    """``get_cancellation_registry`` and ``cancel_thread`` use the singleton."""

    async def test_singleton_identity(self) -> None:
        a = get_cancellation_registry()
        b = get_cancellation_registry()
        assert a is b

    async def test_cancel_thread_helper_uses_singleton(self) -> None:
        registry = get_cancellation_registry()

        async def _runner() -> None:
            await asyncio.sleep(60)

        task = asyncio.create_task(_runner())
        try:
            registry.register("singleton-t", task)
            assert cancel_thread("singleton-t") is True
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            registry.unregister("singleton-t")

    async def test_cancel_thread_unknown_returns_false_via_singleton(self) -> None:
        # Use a clearly unique thread_id to avoid collision with any
        # other test that may have registered something on the singleton.
        assert cancel_thread("nonexistent-zzz-001") is False


class TestCancellationDuringInvocation:
    """End-to-end: register a long-running task, cancel it, observe rollback."""

    async def test_simulated_invoke_cancellation(self) -> None:
        registry = ThreadCancellationRegistry()
        ran_to_completion = asyncio.Event()
        observed_cancel = asyncio.Event()

        async def _simulated_graph_ainvoke() -> None:
            try:
                await asyncio.sleep(60)
                ran_to_completion.set()
            except asyncio.CancelledError:
                observed_cancel.set()
                raise

        async def _invoke_handler(thread_id: str) -> None:
            async with registry.track(thread_id):
                await _simulated_graph_ainvoke()

        task = asyncio.create_task(_invoke_handler("t-slow"))
        # Yield so the handler enters the ``track`` block.
        await asyncio.sleep(0)
        # Round-trip several times to let the inner sleep actually start.
        for _ in range(3):
            await asyncio.sleep(0)

        assert registry.cancel("t-slow") is True
        with pytest.raises(asyncio.CancelledError):
            await task

        assert observed_cancel.is_set()
        assert not ran_to_completion.is_set()
        assert registry.is_running("t-slow") is False


class TestStreamWrapperPattern:
    """Mirrors the pattern :func:`_track_stream` uses in contrib.fastapi.

    A separate test (rather than spinning up the FastAPI integration)
    so the contract is pinned at the registry level: registering
    ``current_task`` from inside an async generator's iteration loop
    captures the *iterator's* task, which is what we want to cancel.
    """

    async def test_async_generator_iterator_is_what_gets_cancelled(self) -> None:
        registry = ThreadCancellationRegistry()
        observed_cancel = asyncio.Event()

        async def _producer() -> AsyncGenerator[str, None]:
            for i in range(1000):
                await asyncio.sleep(0.01)
                yield f"chunk-{i}"

        async def _wrapped() -> AsyncGenerator[str, None]:
            try:
                async with registry.track("t-stream"):
                    async for chunk in _producer():
                        yield chunk
            except asyncio.CancelledError:
                observed_cancel.set()
                raise

        async def _consumer() -> list[str]:
            chunks: list[str] = []
            async for chunk in _wrapped():
                chunks.append(chunk)
            return chunks

        task = asyncio.create_task(_consumer())
        # Let the producer get going.
        for _ in range(5):
            await asyncio.sleep(0.01)

        assert registry.cancel("t-stream") is True
        with pytest.raises(asyncio.CancelledError):
            await task

        assert observed_cancel.is_set()
        assert registry.is_running("t-stream") is False
