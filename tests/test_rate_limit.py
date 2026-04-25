"""Coverage — per-user rate limit middleware + token bucket added by issue #25.

Tests exercise the bucket primitive, the in-memory backend's
multi-key isolation + concurrency safety, and the ASGI middleware's
allow / deny / excluded-path / anonymous-key behaviour.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from langgraph_kit.contrib.rate_limit import (
    InMemoryRateLimitBackend,
    RateLimitMiddleware,
    TokenBucket,
    default_user_key,
)

# ---------------------------------------------------------------------------
# TokenBucket
# ---------------------------------------------------------------------------


def test_token_bucket_starts_full() -> None:
    bucket = TokenBucket(capacity=10.0, rate_per_second=1.0)
    assert bucket.tokens == 10.0


def test_token_bucket_take_succeeds_until_empty() -> None:
    bucket = TokenBucket(capacity=3.0, rate_per_second=1.0)
    for _ in range(3):
        assert bucket.take(1.0).allowed is True
    decision = bucket.take(1.0)
    assert decision.allowed is False
    assert decision.retry_after_seconds > 0


def test_token_bucket_does_not_deduct_on_denial() -> None:
    bucket = TokenBucket(capacity=1.0, rate_per_second=1.0)
    assert bucket.take(1.0).allowed is True
    assert bucket.tokens < 1.0
    pre = bucket.tokens
    decision = bucket.take(1.0)
    assert decision.allowed is False
    # Deduction did not happen on the denied take (tokens may have
    # refilled slightly between checks but never go down by 1.0).
    assert bucket.tokens >= pre - 1e-6


def test_token_bucket_refills_over_time() -> None:
    bucket = TokenBucket(capacity=1.0, rate_per_second=100.0)
    bucket.take(1.0)
    assert bucket.take(1.0).allowed is False
    time.sleep(0.05)  # ~5 tokens worth
    assert bucket.take(1.0).allowed is True


def test_token_bucket_caps_at_capacity_after_long_idle() -> None:
    bucket = TokenBucket(capacity=2.0, rate_per_second=1000.0)
    bucket.take(2.0)
    time.sleep(0.05)
    bucket._refill()  # explicit, deterministic
    assert bucket.tokens == 2.0


def test_token_bucket_rejects_invalid_init() -> None:
    with pytest.raises(ValueError, match="positive"):
        TokenBucket(capacity=0, rate_per_second=1.0)
    with pytest.raises(ValueError, match="positive"):
        TokenBucket(capacity=1.0, rate_per_second=-1.0)


def test_token_bucket_take_zero_is_always_allowed() -> None:
    bucket = TokenBucket(capacity=1.0, rate_per_second=1.0)
    bucket.take(1.0)  # drain
    assert bucket.take(0.0).allowed is True


# ---------------------------------------------------------------------------
# InMemoryRateLimitBackend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backend_separate_buckets_per_key() -> None:
    backend = InMemoryRateLimitBackend(capacity=2.0, rate_per_second=0.001)
    # Drain user A
    assert (await backend.take("user:a")).allowed is True
    assert (await backend.take("user:a")).allowed is True
    assert (await backend.take("user:a")).allowed is False
    # User B is unaffected
    assert (await backend.take("user:b")).allowed is True


@pytest.mark.asyncio
async def test_backend_concurrent_takes_dont_oversubscribe() -> None:
    """asyncio.Lock guards the bucket — N concurrent takes against
    capacity=N must all succeed exactly N times."""
    backend = InMemoryRateLimitBackend(capacity=10.0, rate_per_second=0.001)
    decisions = await asyncio.gather(*(backend.take("k") for _ in range(20)))
    allowed = sum(1 for d in decisions if d.allowed)
    assert allowed == 10  # exactly the capacity, no over-take


# ---------------------------------------------------------------------------
# default_user_key
# ---------------------------------------------------------------------------


def test_default_user_key_falls_back_to_anonymous_when_no_user() -> None:
    assert default_user_key({}) == "anonymous"
    assert default_user_key({"state": {}}) == "anonymous"
    assert default_user_key({"state": {"current_user": None}}) == "anonymous"


def test_default_user_key_extracts_user_id() -> None:
    class _User:
        id = "alice"

    assert default_user_key({"state": {"current_user": _User()}}) == "user:alice"


def test_default_user_key_falls_back_when_user_has_no_id() -> None:
    class _User:
        pass

    assert default_user_key({"state": {"current_user": _User()}}) == "anonymous"


# ---------------------------------------------------------------------------
# Middleware (ASGI-level smoke test)
# ---------------------------------------------------------------------------


class _DummyApp:
    """Trivial ASGI app that always returns 200 OK with empty body."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        self.calls += 1
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})


async def _drive(
    middleware: RateLimitMiddleware, scope: dict[str, Any]
) -> tuple[int, dict[bytes, bytes], bytes]:
    """Run a request through the middleware and capture the response."""
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await middleware(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    headers = dict(start.get("headers", []))
    body_message = next(
        (m for m in sent if m["type"] == "http.response.body"), {"body": b""}
    )
    return start["status"], headers, body_message.get("body", b"")


@pytest.mark.asyncio
async def test_middleware_allows_under_limit() -> None:
    app = _DummyApp()
    backend = InMemoryRateLimitBackend(capacity=2.0, rate_per_second=0.001)
    mw = RateLimitMiddleware(app, backend)
    scope = {"type": "http", "path": "/agents", "state": {}}

    status, _, _ = await _drive(mw, scope)
    assert status == 200
    assert app.calls == 1


@pytest.mark.asyncio
async def test_middleware_blocks_over_limit_with_429() -> None:
    app = _DummyApp()
    backend = InMemoryRateLimitBackend(capacity=1.0, rate_per_second=0.001)
    mw = RateLimitMiddleware(app, backend)
    scope = {"type": "http", "path": "/agents", "state": {}}

    # First request consumes the only token.
    s1, _, _ = await _drive(mw, scope)
    assert s1 == 200
    # Second request is denied.
    s2, headers, body = await _drive(mw, scope)
    assert s2 == 429
    assert b"retry-after" in headers
    parsed = json.loads(body)
    assert parsed["error"] == "rate_limited"
    assert parsed["retry_after_seconds"] >= 1
    # Underlying app was NOT called the second time.
    assert app.calls == 1


@pytest.mark.asyncio
async def test_middleware_excludes_health_paths_from_limiting() -> None:
    app = _DummyApp()
    backend = InMemoryRateLimitBackend(capacity=1.0, rate_per_second=0.001)
    mw = RateLimitMiddleware(app, backend)

    # Pre-drain the anonymous bucket.
    await backend.take("anonymous")

    for path in ("/healthz", "/readyz"):
        status, _, _ = await _drive(mw, {"type": "http", "path": path, "state": {}})
        assert status == 200, f"{path} should bypass rate limit"


@pytest.mark.asyncio
async def test_middleware_passes_non_http_through() -> None:
    """Lifespan / websocket scopes shouldn't go through the rate
    limiter — they aren't user-driven requests."""
    app = _DummyApp()
    backend = InMemoryRateLimitBackend(
        capacity=0.0001, rate_per_second=0.001
    )  # super-tight
    # Backend init forbids 0; use a bigger lifespan-shaped scope and
    # confirm the middleware short-circuits.
    backend = InMemoryRateLimitBackend(capacity=1.0, rate_per_second=0.001)
    mw = RateLimitMiddleware(app, backend)

    # Drain so HTTP would be denied; lifespan should still pass through.
    await backend.take("anonymous")

    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "lifespan.startup"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await mw({"type": "lifespan"}, receive, send)
    # The dummy app responds with http.response.start for any scope
    # — confirms our pass-through reached it.
    assert any(m["type"] == "http.response.start" for m in sent)


@pytest.mark.asyncio
async def test_middleware_buckets_anonymous_users_together() -> None:
    """No user_id in scope → all anonymous traffic shares one bucket."""
    app = _DummyApp()
    backend = InMemoryRateLimitBackend(capacity=1.0, rate_per_second=0.001)
    mw = RateLimitMiddleware(app, backend)
    scope = {"type": "http", "path": "/agents", "state": {}}

    s1, _, _ = await _drive(mw, scope)
    s2, _, _ = await _drive(mw, scope)
    assert s1 == 200
    assert s2 == 429


@pytest.mark.asyncio
async def test_middleware_separates_buckets_per_user() -> None:
    class _User:
        def __init__(self, uid: str) -> None:
            self.id = uid

    app = _DummyApp()
    backend = InMemoryRateLimitBackend(capacity=1.0, rate_per_second=0.001)
    mw = RateLimitMiddleware(app, backend)

    alice_scope = {
        "type": "http",
        "path": "/agents",
        "state": {"current_user": _User("alice")},
    }
    bob_scope = {
        "type": "http",
        "path": "/agents",
        "state": {"current_user": _User("bob")},
    }

    s1, _, _ = await _drive(mw, alice_scope)
    s2, _, _ = await _drive(mw, bob_scope)
    assert s1 == 200
    assert s2 == 200  # different bucket, also allowed
    s3, _, _ = await _drive(mw, alice_scope)
    assert s3 == 429  # alice ran out
