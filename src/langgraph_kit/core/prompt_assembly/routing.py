"""A/B routing primitive for ``PromptSection`` versions (#18).

Pairs with the multi-version :class:`SectionRegistry` already shipped
in :mod:`langgraph_kit.core.prompt_assembly.sections`. Use a router
when you want to A/B test prompt revisions, run a canary rollout, or
shadow-test a candidate version against the current one.

Design — keep the router boring. The mapping from
*run context* → *per-section version* is a callable the caller
provides; this module just encodes:

- a stable hash-bucketing strategy (so a given user always lands in
  the same arm);
- a percentage-rollout strategy that delegates the bucketing to it;
- the integration point: :py:meth:`PromptVersionRouter.snapshot`
  returns ``{section_id: version}`` for one run, suitable to thread
  into :func:`build_agent_run_config(prompt_versions=...)`.

Persistence and feature-flag wiring are intentionally out of scope —
they're the responsibility of the canary-rollout follow-up. This
file ships the primitives needed to build either.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgraph_kit.core.prompt_assembly.sections import SectionRegistry


@dataclass(frozen=True)
class RunContext:
    """Per-run inputs available to a routing strategy.

    Strategies typically bucket on :attr:`user_id` so a single user
    sees a stable assignment across runs (avoids "user gets two
    different prompt versions in one conversation"). :attr:`extra`
    is a free-form bag for callers that want to bucket on something
    custom (cohort id, tenant id, region) without subclassing.
    """

    user_id: str | None = None
    thread_id: str | None = None
    agent_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# Strategy callable: takes a RunContext, returns {section_id: version}.
# An empty dict means "use the registry's current version for every
# section." A dict that omits some section_ids leaves those alone.
RoutingStrategy = Callable[[RunContext], dict[str, str]]


def stable_bucket(key: str, num_buckets: int) -> int:
    """Return a deterministic bucket index in ``[0, num_buckets)``.

    Uses BLAKE2b-128 of the key — hash collisions don't matter for
    bucketing, only the integer modulo. ``num_buckets`` must be ``>= 1``;
    otherwise the function raises :class:`ValueError`.

    Same input always yields the same bucket: the property A/B routing
    relies on so a user doesn't flip arms run-to-run.
    """
    if num_buckets < 1:
        msg = f"num_buckets must be >= 1, got {num_buckets!r}"
        raise ValueError(msg)
    digest = hashlib.blake2b(key.encode(), digest_size=16).digest()
    return int.from_bytes(digest, "big") % num_buckets


def percentage_rollout(
    section_id: str,
    *,
    new_version: str,
    base_version: str,
    percent_new: float,
    bucket_key: Callable[[RunContext], str | None] | None = None,
) -> RoutingStrategy:
    """Build a routing strategy that sends *percent_new* of users to *new_version*.

    Bucketing is stable per *bucket_key* — by default
    :attr:`RunContext.user_id`. When *bucket_key* returns ``None``
    (e.g. anonymous run), the strategy falls back to *base_version*
    so anonymous traffic never lands in a canary.

    *percent_new* is a fraction in ``[0.0, 1.0]``. ``0.0`` keeps every
    bucket on *base_version*; ``1.0`` flips every bucket to
    *new_version*. Values outside that range raise :class:`ValueError`.

    Returns a strategy callable suitable to pass to
    :class:`PromptVersionRouter`. The strategy only sets the version
    for *section_id* — other sections use whatever the registry
    points at currently.
    """
    if not 0.0 <= percent_new <= 1.0:
        msg = f"percent_new must be in [0.0, 1.0], got {percent_new!r}"
        raise ValueError(msg)
    keyfn = bucket_key or (lambda ctx: ctx.user_id)
    # 10000-bucket grid → 0.01% resolution. Matches the granularity
    # most rollout tooling exposes; finer than that and the bucketing
    # noise dominates.
    n_buckets = 10_000
    cutoff = int(percent_new * n_buckets)

    def _strategy(ctx: RunContext) -> dict[str, str]:
        key = keyfn(ctx)
        if key is None:
            return {section_id: base_version}
        bucket = stable_bucket(f"{section_id}:{key}", n_buckets)
        return {section_id: new_version if bucket < cutoff else base_version}

    return _strategy


class PromptVersionRouter:
    """Resolve per-run section versions from a strategy callable.

    Given a :class:`SectionRegistry` and a :data:`RoutingStrategy`,
    :py:meth:`snapshot` produces ``{section_id: version}`` for one
    run. The result is suitable to:

    - thread into ``build_agent_run_config(prompt_versions=...)`` so
      Langfuse / tracing tags the run with its arm;
    - feed back into :py:meth:`SectionRegistry.set_current` per-build
      if you want the new version live for a specific compiled graph
      (out of scope here — composing with the builder is the canary
      follow-up's problem).

    The router never mutates the registry. Strategies that name
    versions which aren't registered would silently produce stale
    metadata; :py:meth:`snapshot` validates against
    :py:meth:`SectionRegistry.list_versions` and raises
    :class:`KeyError` for unknown ``(id, version)`` pairs so a typo
    in the strategy fails loudly.
    """

    def __init__(
        self,
        registry: SectionRegistry,
        strategy: RoutingStrategy,
    ) -> None:
        super().__init__()
        self._registry = registry
        self._strategy = strategy

    def snapshot(self, ctx: RunContext) -> dict[str, str]:
        """Return ``{section_id: version}`` for the run described by *ctx*.

        Section ids the strategy doesn't mention default to the
        registry's :py:meth:`current_version`. Section ids the
        strategy names but the registry doesn't recognize raise
        :class:`KeyError`; section ids that exist but with an
        unregistered version do too.
        """
        result: dict[str, str] = dict(self._registry.current_versions())
        overrides = self._strategy(ctx)
        for section_id, version in overrides.items():
            known_versions = self._registry.list_versions(section_id)
            if not known_versions:
                msg = f"Strategy referenced unknown section id: {section_id!r}"
                raise KeyError(msg)
            if version not in known_versions:
                msg = (
                    f"Strategy referenced unknown version {version!r} for "
                    f"section {section_id!r}; known: {sorted(known_versions)}"
                )
                raise KeyError(msg)
            result[section_id] = version
        return result


__all__ = [
    "PromptVersionRouter",
    "RoutingStrategy",
    "RunContext",
    "percentage_rollout",
    "stable_bucket",
]
