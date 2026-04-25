"""Per-user rate limiting for the FastAPI surface.

Issue #25 lands the foundation: a token-bucket implementation, an
in-memory single-process backend, a generic backend Protocol so a
multi-process Redis backend can drop in later, and a Starlette /
FastAPI middleware that enforces requests-per-minute keyed by the
``current_user`` dependency.

Tokens-per-day enforcement (the second limit in the issue) is
deferred — it integrates with the existing :class:`BudgetManager`
and is materially separate work.
"""

from .backend import (
    InMemoryRateLimitBackend,
    RateLimitBackend,
    RateLimitDecision,
    TokenBucket,
)
from .middleware import RateLimitMiddleware, default_user_key

__all__ = [
    "InMemoryRateLimitBackend",
    "RateLimitBackend",
    "RateLimitDecision",
    "RateLimitMiddleware",
    "TokenBucket",
    "default_user_key",
]
