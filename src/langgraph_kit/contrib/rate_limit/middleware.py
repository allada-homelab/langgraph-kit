"""Starlette/FastAPI middleware that enforces per-user rate limits.

Drops in via :meth:`fastapi.FastAPI.add_middleware` (or directly on
a Starlette app). When a request would exceed the configured rate
the middleware returns ``HTTP 429 Too Many Requests`` with a
``Retry-After`` header and a small JSON body describing the limit
that fired.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .backend import RateLimitBackend

_ANONYMOUS_KEY: str = "anonymous"


def default_user_key(scope: dict[str, Any]) -> str:
    """Default key extractor: ASGI scope -> rate-limit bucket key.

    Looks at ``scope["state"]["current_user"]`` (set by FastAPI's
    dependency-injection pipeline at request time) and falls back to
    a single shared ``"anonymous"`` bucket so unauthenticated traffic
    is still bounded.
    """
    state = scope.get("state") or {}
    user = state.get("current_user")
    if user is None:
        return _ANONYMOUS_KEY
    user_id = getattr(user, "id", None) or getattr(user, "user_id", None)
    if user_id is None:
        return _ANONYMOUS_KEY
    return f"user:{user_id}"


class RateLimitMiddleware:
    """ASGI middleware enforcing a per-user request rate.

    Uses a :class:`RateLimitBackend` (in-memory by default) to take
    one token per HTTP request. Excludes the configured set of paths
    (default ``/healthz``, ``/readyz``) so probes never get throttled.
    """

    def __init__(
        self,
        app: Any,
        backend: RateLimitBackend,
        *,
        key_fn: Callable[[dict[str, Any]], str] = default_user_key,
        excluded_paths: tuple[str, ...] = ("/healthz", "/readyz"),
        cost_per_request: float = 1.0,
    ) -> None:
        super().__init__()
        self._app = app
        self._backend = backend
        self._key_fn = key_fn
        self._excluded = tuple(excluded_paths)
        self._cost = float(cost_per_request)

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return

        path = scope.get("path", "")
        if any(
            path == ex or path.startswith(ex.rstrip("/") + "/") for ex in self._excluded
        ):
            await self._app(scope, receive, send)
            return

        key = self._key_fn(scope)
        decision = await self._backend.take(key, n=self._cost)
        if decision.allowed:
            await self._app(scope, receive, send)
            return

        # Limit hit — short-circuit with 429 + Retry-After.
        retry_after = max(1, math.ceil(decision.retry_after_seconds))
        body = json.dumps(
            {
                "error": "rate_limited",
                "message": (
                    f"Rate limit exceeded for {key}. Retry after {retry_after}s."
                ),
                "retry_after_seconds": retry_after,
            }
        ).encode()

        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"retry-after", str(retry_after).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
