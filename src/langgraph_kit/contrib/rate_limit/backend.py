"""Token-bucket rate-limit primitives.

Two pieces:

- :class:`TokenBucket` — a single bucket with a configurable rate
  (tokens / second) and capacity (max burst). Pure data structure;
  no I/O, async-safe under a single-thread event loop.
- :class:`RateLimitBackend` — the protocol the middleware speaks to.
  In-memory backend ships in this module; a multi-process Redis
  backend can implement the same protocol later (#27 cross-process
  consistency).

The bucket uses a continuous-refill model: every check first refills
based on wall-clock time since the last touch, then attempts to
spend ``n`` tokens. If the bucket lacks the tokens, ``take`` returns
:class:`RateLimitDecision` with ``allowed=False`` and a
``retry_after_seconds`` that says how long until the requested
amount would be available.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class RateLimitDecision:
    """Result of a single :meth:`RateLimitBackend.take` call."""

    allowed: bool
    retry_after_seconds: float = 0.0
    remaining: float = 0.0


@dataclass
class TokenBucket:
    """Continuous-refill token bucket.

    ``rate_per_second`` controls the steady-state refill rate;
    ``capacity`` is the max burst the bucket can hold. The bucket
    starts full so first-request latency isn't spent waiting on
    a fill. Internal state is wall-clock based (``time.monotonic``)
    so freezing/thawing the process doesn't grant retroactive
    capacity.
    """

    capacity: float
    rate_per_second: float
    tokens: float = field(init=False)
    _last_refill: float = field(init=False, default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if self.capacity <= 0 or self.rate_per_second <= 0:
            msg = "capacity and rate_per_second must be positive"
            raise ValueError(msg)
        self.tokens = float(self.capacity)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self.tokens = min(
                self.capacity, self.tokens + elapsed * self.rate_per_second
            )
            self._last_refill = now

    def take(self, n: float = 1.0) -> RateLimitDecision:
        """Attempt to spend ``n`` tokens. Refills first.

        Returns ``allowed=False`` when the bucket lacks tokens, with
        ``retry_after_seconds`` set to the wait that would make the
        request fit. Does not deduct tokens on a denied take.
        """
        if n <= 0:
            return RateLimitDecision(allowed=True, remaining=self.tokens)
        self._refill()
        if self.tokens >= n:
            self.tokens -= n
            return RateLimitDecision(allowed=True, remaining=self.tokens)
        deficit = n - self.tokens
        wait = deficit / self.rate_per_second if self.rate_per_second > 0 else 0.0
        return RateLimitDecision(
            allowed=False,
            retry_after_seconds=wait,
            remaining=self.tokens,
        )


class RateLimitBackend(Protocol):
    """Pluggable backing store for per-key rate limits.

    The middleware only depends on ``take``; new backends (Redis,
    Postgres advisory locks, etc.) implement this Protocol without
    needing to be subclasses of :class:`InMemoryRateLimitBackend`.
    """

    async def take(
        self, key: str, n: float = 1.0
    ) -> RateLimitDecision:  # pragma: no cover — interface only
        ...


class InMemoryRateLimitBackend:
    """Token-bucket store keyed by string. Single-process only.

    Multi-worker deployments need a cross-process backend (Redis is
    the obvious choice; #27 covers the broader cross-process story).
    """

    def __init__(
        self,
        *,
        capacity: float,
        rate_per_second: float,
    ) -> None:
        super().__init__()
        if capacity <= 0 or rate_per_second <= 0:
            msg = "capacity and rate_per_second must be positive"
            raise ValueError(msg)
        self._capacity = float(capacity)
        self._rate = float(rate_per_second)
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    def _bucket(self, key: str) -> TokenBucket:
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = TokenBucket(capacity=self._capacity, rate_per_second=self._rate)
            self._buckets[key] = bucket
        return bucket

    async def take(self, key: str, n: float = 1.0) -> RateLimitDecision:
        # asyncio.Lock guards the dict-mutation + bucket take so two
        # concurrent requests for the same key don't double-spend.
        async with self._lock:
            return self._bucket(key).take(n)
