"""Store-condition watcher triggers.

A watcher polls a Store namespace at a configurable interval and
fires an agent invocation when a caller-supplied predicate goes
from false to true. Edge-triggered: a stable "predicate stays true"
state doesn't re-fire on every poll — the watcher tracks the
"already fired for this rising edge" state internally and only
re-arms once the predicate flips back to false.

Usage::

    from langgraph_kit.contrib.watchers import (
        StoreWatcherRegistry, StoreWatcherSpec, StoreWatcherRunner,
    )

    registry = StoreWatcherRegistry()
    registry.register(StoreWatcherSpec(
        id="alert-batcher",
        agent_id="batcher-agent",
        namespace=("alerts", "unhandled"),
        predicate=lambda items: len(items) >= 10,
        poll_interval_seconds=60.0,
        payload_template="Process the unhandled alerts queue.",
    ))

    async with StoreWatcherRunner(
        registry, store=my_store, graph_resolver=...
    ) as runner:
        await runner.start()
        # ... lifespan continues; runner.stop() called on __aexit__

The runner spawns one asyncio task per registered watcher; each
task loops ``await asyncio.sleep(interval)`` → poll → fire if
edge-triggered. ``poll_now(spec_id)`` triggers a single poll out of
band for tests.

## Scope (issue #82 v1)

- One watcher surface backed by stdlib polling (no LISTEN/NOTIFY,
  no Postgres-specific push). Works against any Store backend that
  honors ``asearch``.
- Edge-triggered firing: predicate-met → fire once → don't refire
  until predicate flips back to false then true again.
- Configurable poll interval per spec; no global default — callers
  choose what trades off freshness vs Store load.
- ``graph_resolver`` callable matches the webhook router and
  scheduled-trigger surfaces so callers share one resolver across
  all three trigger types.

## Deferred

- Push-based watchers via Postgres ``LISTEN/NOTIFY`` for hot-path
  use cases (current API stays compatible).
- Persistence of fire history. Useful for ops dashboards; defer
  until someone asks.
- Coalescing across multi-worker deployments. Until #27 lands the
  Postgres advisory-lock primitives, run one watcher process per
  deployment (or accept up-to-N duplicate fires for a given rising
  edge).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

logger = logging.getLogger(__name__)


# ``Predicate`` operates on the list returned by ``store.asearch(namespace)``.
# Type loose because Store backends differ slightly in item shape; the
# predicate sees what asearch returns, not a normalized form.
Predicate = "Callable[[list[Any]], bool]"


class StoreWatcherSpec(BaseModel):
    """Configuration for one store-condition watcher.

    Frozen because the registry uses ``id`` as a stable key — a spec
    that mutates after registration could silently re-target the
    watch with no audit trail.

    The ``predicate`` callable can't be JSON-serialized; this is a
    runtime-configured spec, not a persistable one. Use
    ``arbitrary_types_allowed`` to let Pydantic accept a callable
    field.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    id: str
    """Stable identifier; used as the watcher's task name and the
    key in :class:`StoreWatcherRegistry`."""

    agent_id: str
    """Which registered agent to fire when the predicate flips
    true. Resolved via ``graph_resolver`` passed to
    :class:`StoreWatcherRunner`."""

    namespace: tuple[str, ...]
    """Store namespace to poll. Same shape the kit's Store-protocol
    callers use elsewhere (e.g. ``("workspace", "task-board-1")``)."""

    predicate: Any
    """``Callable[[list[Item]], bool]``. Receives the raw
    ``asearch(namespace)`` result. Returning ``True`` arms a
    fire; the watcher refuses to re-fire until a subsequent poll
    sees ``False`` (edge-triggered).

    Typed as ``Any`` to accommodate Pydantic's reluctance to
    validate callables; the callable contract is documented but not
    enforced at registration."""

    poll_interval_seconds: float = 60.0
    """How often to poll. Per-spec because the right cadence
    depends on the predicate's volatility — an "alerts >= 10"
    watcher might poll every 10s; a "memory store has > 1000
    entries" watcher every hour."""

    payload_template: str = ""
    """Plain string handed to the agent as the user message when
    the watcher fires. Like the schedule trigger, parameterization
    happens at registration via Python f-strings."""

    payload_data: dict[str, Any] = Field(default_factory=dict)
    """Free-form metadata stored on the spec; for ops/log context.
    The agent itself only sees ``payload_template``."""


class StoreWatcherRegistry:
    """In-process registry of :class:`StoreWatcherSpec` instances.

    Persistence is intentionally out of scope for v1 — initialize
    at app startup with whatever specs you care about.
    Re-registering an existing id replaces it; the runner picks up
    the new spec on the next ``start()`` (already-running watchers
    keep using their original spec until the runner restarts).
    """

    def __init__(self) -> None:
        super().__init__()
        self._specs: dict[str, StoreWatcherSpec] = {}

    def register(self, spec: StoreWatcherSpec) -> None:
        """Register *spec*. Validates per-spec invariants."""
        if spec.poll_interval_seconds <= 0:
            msg = f"poll_interval_seconds must be > 0; got {spec.poll_interval_seconds}"
            raise ValueError(msg)
        if not callable(spec.predicate):
            msg = "predicate must be callable"
            raise TypeError(msg)
        self._specs[spec.id] = spec

    def get(self, spec_id: str) -> StoreWatcherSpec | None:
        return self._specs.get(spec_id)

    def remove(self, spec_id: str) -> None:
        self._specs.pop(spec_id, None)

    def list_ids(self) -> list[str]:
        return list(self._specs.keys())

    def all_specs(self) -> list[StoreWatcherSpec]:
        """All registered specs in insertion order."""
        return list(self._specs.values())


class StoreWatcherRunner:
    """Lifecycle wrapper that drives polling tasks for each watcher.

    One asyncio task per spec, each looping
    ``await asyncio.sleep(interval)`` → poll → fire if edge-triggered.
    Designed as an async context manager so the FastAPI lifespan
    pattern is the natural shape::

        async with StoreWatcherRunner(
            registry, store=my_store, graph_resolver=...
        ) as runner:
            await runner.start()
            yield
            # __aexit__ stops the watcher tasks cleanly

    For tests: ``poll_now(spec_id)`` runs one poll cycle for the
    named spec out of band so coverage doesn't depend on real
    wall-clock advancement.
    """

    def __init__(
        self,
        registry: StoreWatcherRegistry,
        *,
        store: Any,
        graph_resolver: Callable[[str], Any],
    ) -> None:
        super().__init__()
        self._registry = registry
        self._store = store
        # graph_resolver mirrors the webhook + scheduled-trigger
        # surfaces: ``(agent_id) -> compiled graph``.
        self._graph_resolver = graph_resolver
        self._tasks: dict[str, asyncio.Task[None]] = {}
        # Edge-trigger state: ``True`` means the predicate fired
        # last poll and we're waiting for it to flip back to
        # ``False`` before re-arming.
        self._armed: dict[str, bool] = {}

    async def __aenter__(self) -> StoreWatcherRunner:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()

    async def start(self) -> None:
        """Spawn one async task per registered watcher.

        Idempotent — calling ``start()`` twice is a no-op on the
        second call. Tasks set up here are stopped via
        :py:meth:`stop` (or the ``__aexit__`` boundary).
        """
        if self._tasks:
            return
        for spec in self._registry.all_specs():
            self._armed[spec.id] = False
            task = asyncio.create_task(
                self._watch_loop(spec.id), name=f"watcher-{spec.id}"
            )
            self._tasks[spec.id] = task
        logger.info("StoreWatcherRunner started %d watcher task(s)", len(self._tasks))

    async def stop(self) -> None:
        """Cancel every watcher task. Idempotent."""
        if not self._tasks:
            return
        for task in self._tasks.values():
            task.cancel()
        # Wait for cancellations to propagate so callers can rely
        # on "stop returns => no more fires".
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()
        self._armed.clear()

    async def poll_now(self, spec_id: str) -> bool:
        """Run one poll cycle for *spec_id*; return True if it fired.

        Bypasses the sleep loop so tests don't need real wall-clock
        time. Maintains the same edge-trigger state as the loop, so
        a test can verify "first call fires, second call (still
        true) doesn't, third call (after predicate flipped false
        and back) fires again."

        Raises ``KeyError`` if *spec_id* isn't registered.
        """
        spec = self._registry.get(spec_id)
        if spec is None:
            msg = f"Unknown watcher spec id: {spec_id!r}"
            raise KeyError(msg)
        # Initialize armed state if this is the first poll for an
        # unstarted runner; preserves the same first-poll semantics
        # as the loop.
        self._armed.setdefault(spec_id, False)
        return await self._poll_once(spec)

    async def _watch_loop(self, spec_id: str) -> None:
        """Per-spec polling loop. Cancellation-aware."""
        spec = self._registry.get(spec_id)
        if spec is None:
            return
        try:
            while True:
                await asyncio.sleep(spec.poll_interval_seconds)
                # Re-fetch spec on each iteration in case a caller
                # swapped it via ``registry.register`` mid-run.
                spec = self._registry.get(spec_id)
                if spec is None:
                    return
                try:
                    await self._poll_once(spec)
                except Exception:
                    logger.exception("StoreWatcher %s poll cycle raised", spec_id)
        except asyncio.CancelledError:
            # Clean exit on stop().
            return

    async def _poll_once(self, spec: StoreWatcherSpec) -> bool:
        """One poll: read the namespace, evaluate the predicate, fire if armed.

        Returns whether the watcher fired this cycle.
        """
        items = await self._store.asearch(spec.namespace, limit=10_000)
        try:
            condition_met = bool(spec.predicate(items))
        except Exception:
            logger.exception(
                "StoreWatcher %s predicate raised; treating as False", spec.id
            )
            condition_met = False

        previously_armed = self._armed.get(spec.id, False)
        self._armed[spec.id] = condition_met

        if condition_met and not previously_armed:
            await self._fire(spec)
            return True
        return False

    async def _fire(self, spec: StoreWatcherSpec) -> str:
        """Invoke the agent for *spec*. Returns the run's thread id."""
        from langchain_core.messages import (
            HumanMessage,
        )

        graph = self._graph_resolver(spec.agent_id)
        thread_id = f"watcher-{spec.id}-{uuid.uuid4().hex[:12]}"
        config = {"configurable": {"thread_id": thread_id}}
        await graph.ainvoke(
            {"messages": [HumanMessage(content=spec.payload_template)]},
            config=config,
        )
        logger.info(
            "StoreWatcher fired",
            extra={
                "spec_id": spec.id,
                "agent_id": spec.agent_id,
                "thread_id": thread_id,
            },
        )
        return thread_id


__all__ = [
    "Predicate",
    "StoreWatcherRegistry",
    "StoreWatcherRunner",
    "StoreWatcherSpec",
]
