"""Rate limiting: per-key token bucket with continuous refill.

What this shows
---------------
- Acquiring tokens via :class:`InMemoryRateLimitBackend.take` (async,
  per-key, asyncio-Lock guarded)
- Burst behaviour (capacity > rate * interval) and refusal once the
  bucket runs dry, with ``retry_after_seconds`` populated for the
  middleware's HTTP 429 response
- The bucket auto-refills with wall-clock elapsed time, so freezing the
  process doesn't grant retroactive capacity

The same backend powers ``RateLimitMiddleware``, which sits in front of
the FastAPI router. Multi-worker deployments need a cross-process
backend (Redis, etc.) — the in-memory backend is right for single-
process dev / preview environments.

How to run
----------
    uv run python -m examples.rate_limit_token_bucket

Expected output
---------------
    Burst of 5 takes against a capacity-3 / rate-1 bucket:
      take 1: allowed=True  remaining=2.0
      take 2: allowed=True  remaining=1.0
      take 3: allowed=True  remaining=0.0
      take 4: allowed=False retry_after=1.00s
      take 5: allowed=False retry_after=1.00s
    Sleeping 1.1s for the bucket to refill...
    take 6: allowed=True  remaining=0.10
"""

from __future__ import annotations

import asyncio

from examples._lib import banner, line


async def main() -> None:
    banner("rate_limit_token_bucket")

    from langgraph_kit.contrib.rate_limit.backend import InMemoryRateLimitBackend

    # 3 tokens of burst, 1 refilled per second. The first three takes
    # succeed instantly; the next two get 429-style decisions; one full
    # second of sleep is enough to refill 1 token.
    backend = InMemoryRateLimitBackend(capacity=3, rate_per_second=1)
    key = "user:demo"

    line("Burst of 5 takes against a capacity-3 / rate-1 bucket:")
    for i in range(1, 6):
        decision = await backend.take(key)
        if decision.allowed:
            line(f"  take {i}: allowed=True  remaining={decision.remaining:.1f}")
        else:
            wait = decision.retry_after_seconds or 0.0
            line(f"  take {i}: allowed=False retry_after={wait:.2f}s")

    line("Sleeping 1.1s for the bucket to refill...")
    await asyncio.sleep(1.1)
    decision = await backend.take(key)
    line(f"take 6: allowed={decision.allowed}  remaining={decision.remaining:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
