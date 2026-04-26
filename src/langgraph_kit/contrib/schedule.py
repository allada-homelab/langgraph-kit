"""Cron-style scheduled triggers for agent invocation.

A scheduled spec ties a cron expression to an agent and a payload
template that renders into the user message the agent sees on each
fire. Built on apscheduler's ``AsyncIOScheduler`` so it integrates
cleanly with the FastAPI lifespan / any asyncio app.

Usage::

    from langgraph_kit.contrib.schedule import (
        ScheduledRegistry, ScheduledSpec, ScheduledTriggerRunner,
    )

    registry = ScheduledRegistry()
    registry.register(ScheduledSpec(
        id="weekly-summary",
        agent_id="reports-agent",
        cron="0 9 * * MON",  # UTC — see below
        payload_template="Generate this week's summary report.",
    ))

    async with ScheduledTriggerRunner(registry, graph_resolver=...) as runner:
        await runner.start()
        # ... lifespan continues; runner.stop() called on __aexit__

The runner exposes ``fire_now(spec_id)`` for testing and manual
invocation; production fires happen via the apscheduler loop.

## Scope (issue #81 v1)

- One scheduled-trigger surface backed by ``apscheduler``.
- Cron expressions validated at registration (apscheduler raises
  ``ValueError`` on bad syntax).
- **UTC-only.** Local-time cron is a footgun across deployments —
  ``"0 9 * * MON"`` always means 09:00 UTC, never local time.
- ``graph_resolver`` callable matches the webhook-router signature
  (``(agent_id) -> compiled graph``) so callers can share one
  resolver across multiple trigger surfaces.

## Deferred

- Multi-worker advisory lock. Issue spec calls for Postgres
  ``pg_try_advisory_lock`` to prevent duplicate fires when N
  workers run the same scheduler. That depends on the wider
  Postgres-coordination work in #27. Until then, run one scheduler
  per deployment (or cope with at-most-N duplicate fires).
- FastAPI lifespan integration as a one-call helper. Today the
  caller wires ``ScheduledTriggerRunner`` into their lifespan
  manually — small enough that a helper is premature.
- Persistence of fire history / next-run timestamps to the Store.
  Useful for ops dashboards; defer until someone asks.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

logger = logging.getLogger(__name__)


class ScheduledSpec(BaseModel):
    """Configuration for one cron-fired agent invocation.

    Frozen because the registry uses ``id`` as a stable key — a spec
    that mutates after it's registered would silently re-target
    the schedule with no audit trail.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    """Stable identifier; used as the apscheduler job id and the
    key in :class:`ScheduledRegistry`."""

    agent_id: str
    """Which registered agent the schedule fires against. Resolved
    via the ``graph_resolver`` callable passed to
    :class:`ScheduledTriggerRunner`."""

    cron: str
    """Five-field cron expression (``minute hour day month
    day-of-week``). Validated at registration via apscheduler's
    parser — bad syntax raises ``ValueError`` immediately rather
    than at first fire.

    UTC-only by design — see module docstring. The ``timezone``
    field on the underlying ``CronTrigger`` is hardcoded to
    ``"UTC"`` because every other choice is a footgun across
    multi-region deployments.
    """

    payload_template: str = ""
    """Plain string handed to the agent as the user message at fire
    time. ``str.format``-style templating against the spec's
    ``payload_data`` would let callers parameterize fires at
    registration time but adds complexity — for v1 the template is
    a literal string. Use Python f-strings at registration for any
    parameterization you need."""

    payload_data: dict[str, Any] = Field(default_factory=dict)
    """Free-form metadata stored on the spec; surfaced to caller
    code via :py:meth:`ScheduledRegistry.get` (e.g. for logging the
    fire context). The agent itself only sees ``payload_template``
    as the user message."""


class ScheduledRegistry:
    """In-process registry of :class:`ScheduledSpec` instances.

    Persistence is intentionally out of scope for v1 — initialize
    at app startup with whatever specs you care about.
    Re-registering an existing id replaces it; the runner picks up
    the new schedule on its next ``start()``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._specs: dict[str, ScheduledSpec] = {}

    def register(self, spec: ScheduledSpec) -> None:
        """Register *spec*. Validates the cron expression up front.

        Raises ``ValueError`` (from apscheduler) if the cron string
        is malformed — catches typos at deploy time instead of at
        first fire.
        """
        # Validate by attempting to construct the apscheduler
        # CronTrigger; we discard the result and let the runner
        # build its own at start time.
        from apscheduler.triggers.cron import (
            CronTrigger,
        )

        CronTrigger.from_crontab(spec.cron, timezone="UTC")
        self._specs[spec.id] = spec

    def get(self, spec_id: str) -> ScheduledSpec | None:
        return self._specs.get(spec_id)

    def remove(self, spec_id: str) -> None:
        self._specs.pop(spec_id, None)

    def list_ids(self) -> list[str]:
        return list(self._specs.keys())

    def all_specs(self) -> list[ScheduledSpec]:
        """All registered specs in insertion order. Used by the runner."""
        return list(self._specs.values())


class ScheduledTriggerRunner:
    """apscheduler lifecycle wrapper for a :class:`ScheduledRegistry`.

    Built as an async context manager so the FastAPI lifespan
    pattern is the natural shape::

        async with ScheduledTriggerRunner(registry, graph_resolver=...) as runner:
            await runner.start()
            yield  # FastAPI lifespan continues
            # __aexit__ stops the scheduler cleanly

    For tests: ``fire_now(spec_id)`` triggers the agent invocation
    out of band so coverage doesn't depend on real wall-clock
    advancement.
    """

    def __init__(
        self,
        registry: ScheduledRegistry,
        *,
        graph_resolver: Callable[[str], Any],
    ) -> None:
        super().__init__()
        # graph_resolver mirrors the webhook router's signature:
        # ``(agent_id) -> compiled graph``. Anything that quacks
        # like a LangGraph compiled graph works (real graph in
        # production, mock in tests).
        self._registry = registry
        self._graph_resolver = graph_resolver
        self._scheduler: Any = None

    async def __aenter__(self) -> ScheduledTriggerRunner:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()

    async def start(self) -> None:
        """Build the scheduler, register every spec's job, start the loop.

        Idempotent — calling ``start()`` twice is a no-op on the
        second call (apscheduler raises if you try to start a
        running scheduler, so we guard).
        """
        if self._scheduler is not None and self._scheduler.running:
            return
        from apscheduler.schedulers.asyncio import (
            AsyncIOScheduler,
        )
        from apscheduler.triggers.cron import (
            CronTrigger,
        )

        scheduler = AsyncIOScheduler(timezone="UTC")
        for spec in self._registry.all_specs():
            scheduler.add_job(
                self._fire,
                trigger=CronTrigger.from_crontab(spec.cron, timezone="UTC"),
                id=spec.id,
                args=[spec.id],
                replace_existing=True,
                misfire_grace_time=60,
            )
        scheduler.start()
        self._scheduler = scheduler
        logger.info(
            "ScheduledTriggerRunner started with %d job(s)",
            len(self._registry.all_specs()),
        )

    async def stop(self) -> None:
        """Stop the scheduler. Idempotent."""
        if self._scheduler is None:
            return
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        self._scheduler = None

    async def fire_now(self, spec_id: str) -> str:
        """Fire *spec_id* immediately, bypassing the cron schedule.

        Returns the thread id of the resulting agent run. Useful for
        tests (no wall-clock dependency) and manual ops invocations.
        Raises ``KeyError`` if *spec_id* isn't registered.
        """
        spec = self._registry.get(spec_id)
        if spec is None:
            msg = f"Unknown scheduled spec id: {spec_id!r}"
            raise KeyError(msg)
        return await self._fire(spec_id)

    async def _fire(self, spec_id: str) -> str:
        """Internal fire path used by both apscheduler and ``fire_now``."""
        from langchain_core.messages import (
            HumanMessage,
        )

        spec = self._registry.get(spec_id)
        if spec is None:
            # Spec was removed between schedule registration and
            # this fire — surface as a warning rather than blowing
            # up the scheduler thread.
            logger.warning(
                "ScheduledTrigger %s fired but spec is no longer registered",
                spec_id,
            )
            return ""

        graph = self._graph_resolver(spec.agent_id)
        thread_id = f"scheduled-{spec.id}-{uuid.uuid4().hex[:12]}"
        config = {"configurable": {"thread_id": thread_id}}
        await graph.ainvoke(
            {"messages": [HumanMessage(content=spec.payload_template)]},
            config=config,
        )
        logger.info(
            "ScheduledTrigger fired",
            extra={
                "spec_id": spec.id,
                "agent_id": spec.agent_id,
                "thread_id": thread_id,
            },
        )
        return thread_id


__all__ = [
    "ScheduledRegistry",
    "ScheduledSpec",
    "ScheduledTriggerRunner",
]
