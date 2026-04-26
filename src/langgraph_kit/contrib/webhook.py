"""Webhook trigger surface — invoke an agent from an HTTP webhook payload.

A webhook spec ties a public webhook ID to an agent, a shared secret
(for HMAC validation), and a payload template that renders the
incoming JSON body into the user-message text the agent will see.

Usage::

    from langgraph_kit.contrib.webhook import (
        WebhookSpec, WebhookRegistry, create_webhook_router,
    )

    registry = WebhookRegistry()
    registry.register(WebhookSpec(
        id="stripe-payment",
        agent_id="payments-agent",
        secret="whsec_xyz",
        payload_template="A new payment arrived: {amount} {currency} from {customer_email}",
    ))

    app.include_router(
        create_webhook_router(registry, graph_resolver=my_graph_lookup),
        prefix="/api/v1",
    )

The router exposes ``POST /webhooks/{webhook_id}`` and:

1. Looks up the spec; 404 if unknown.
2. Validates the HMAC-SHA256 signature in ``signature_header`` against
   the raw request body; 401 on mismatch or missing header.
3. Renders ``payload_template`` against the parsed JSON body; 422 if a
   placeholder isn't present in the payload.
4. Invokes the resolved agent via ``graph.ainvoke`` with the rendered
   text as the single user message and ``configurable.thread_id`` set
   to a webhook-prefixed UUID.
5. Returns ``{"thread_id": ..., "agent_id": ...}`` on success.

Scope (issue #19 v1):

- Webhook trigger only. Cron / Store-condition watchers are tracked
  separately (see issue body for the full plan).
- HMAC validation is mandatory — there's no "no-signature" mode by
  design. A webhook that anyone on the internet can fire is a
  trivially-abusable agent invocation surface.
- Idempotency / dedup is the caller's responsibility; document a
  request-id header convention in your webhook spec.
"""

# NOTE: intentionally does NOT use ``from __future__ import annotations`` —
# FastAPI inspects route signatures via ``typing.get_type_hints()``,
# which fails on string annotations whose names are only resolvable in
# the route factory's closure (``Request`` is imported lazily inside
# ``create_webhook_router``). See ``contrib/fastapi.py`` for the same
# constraint elsewhere in the kit.

import hashlib
import hmac
import logging
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from fastapi import APIRouter

logger = logging.getLogger(__name__)


_DEFAULT_SIGNATURE_HEADER = "X-Hub-Signature-256"
"""Default header name for the HMAC signature.

Matches GitHub's webhook convention so existing webhook-sender libraries
work out of the box. Override per spec when integrating a service that
uses a different header (Stripe → ``Stripe-Signature``, etc.).
"""

_SIGNATURE_PREFIX = "sha256="
"""Expected prefix on the signature header value (GitHub convention).

The header value is ``sha256=<hex digest>``. We strip the prefix before
constant-time-compare; missing prefix → mismatch (don't try to be
clever about format detection).
"""


class WebhookSpec(BaseModel):
    """Configuration for one webhook endpoint.

    Frozen because the registry uses (id) as a stable key — mutating an
    in-flight spec would let a webhook fire against a different agent
    than the one registered, with no audit trail.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    """Public path component for the webhook URL (``/webhooks/{id}``).

    Treat as semi-secret — a leaked id plus a leaked secret allows
    arbitrary agent invocations. Don't put PII or anything guessable
    here.
    """

    agent_id: str
    """Which registered agent the webhook fires against."""

    secret: str
    """Shared secret used to compute the expected HMAC-SHA256 over the
    raw request body. The router compares this against the value in
    ``signature_header`` using :func:`hmac.compare_digest`.

    Rotate by re-registering the spec with a new secret; old senders
    will start failing immediately (by design — silent grace periods
    are how secret leaks become persistent).
    """

    payload_template: str = "{event}"
    """``str.format``-style template rendered against the parsed JSON
    body. The rendered string becomes the user message handed to the
    agent. Default ``"{event}"`` expects a top-level ``event`` field;
    override per-spec for richer rendering.

    Placeholders that aren't present in the payload yield HTTP 422.
    """

    signature_header: str = _DEFAULT_SIGNATURE_HEADER
    """HTTP header name carrying the HMAC signature.

    Defaults to GitHub's convention; override per-service. Header
    matching is case-insensitive (FastAPI normalizes headers).
    """


class WebhookRegistry:
    """In-process registry of :class:`WebhookSpec` instances.

    Persistence is intentionally out of scope for v1 — the registry is
    initialized at FastAPI startup with whatever specs the app cares
    about. Multi-worker deployments should construct the same set in
    each worker (or shard by webhook id).
    """

    def __init__(self) -> None:
        super().__init__()
        self._specs: dict[str, WebhookSpec] = {}

    def register(self, spec: WebhookSpec) -> None:
        """Register *spec*. Re-registering an existing id replaces it."""
        self._specs[spec.id] = spec

    def get(self, webhook_id: str) -> WebhookSpec | None:
        return self._specs.get(webhook_id)

    def remove(self, webhook_id: str) -> None:
        self._specs.pop(webhook_id, None)

    def list_ids(self) -> list[str]:
        return list(self._specs.keys())


# ---------------------------------------------------------------------------
# Signature + templating primitives (broken out so tests can exercise them
# without spinning up a FastAPI app).
# ---------------------------------------------------------------------------


def compute_signature(secret: str, body: bytes) -> str:
    """Return the canonical ``sha256=<hex>`` signature for *body*.

    Pure helper — no I/O, no FastAPI deps. Useful for senders writing
    test fixtures and for the router's verify path.
    """
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"{_SIGNATURE_PREFIX}{digest}"


def verify_signature(secret: str, body: bytes, header_value: str | None) -> bool:
    """Constant-time-verify *header_value* against the expected HMAC of *body*.

    Returns ``False`` for any of: missing header, missing prefix,
    digest mismatch. Never raises — callers translate the boolean to
    HTTP status.
    """
    if not header_value or not header_value.startswith(_SIGNATURE_PREFIX):
        return False
    expected = compute_signature(secret, body)
    return hmac.compare_digest(expected, header_value)


def render_payload(template: str, payload: dict[str, Any]) -> str:
    """Render *template* with *payload* via ``str.format``.

    Raises ``KeyError`` if a placeholder isn't present in the payload —
    the router catches this and returns HTTP 422 with the missing key
    in the detail. Avoid f-string-style nested attribute access in
    templates; if the payload is nested, flatten it before passing in.
    """
    return template.format(**payload)


# ---------------------------------------------------------------------------
# FastAPI router factory.
# ---------------------------------------------------------------------------


def create_webhook_router(
    registry: WebhookRegistry,
    *,
    graph_resolver: Callable[[str], Any],
    prefix: str = "/webhooks",
    audit_store: Any = None,
) -> "APIRouter":
    # graph_resolver is ``(agent_id: str) -> compiled graph``. The
    # router calls ``graph.ainvoke({"messages": [HumanMessage(...)]})``
    # against the returned object, so anything that quacks like a
    # LangGraph compiled graph works (real graph, mock for tests, etc.).

    """Build a FastAPI router that turns webhook hits into agent runs.

    Imports FastAPI lazily so the wider kit can be imported without it
    (the rest of ``contrib/`` follows the same pattern).
    """
    from fastapi import APIRouter, HTTPException, Request, status
    from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
        HumanMessage,
    )

    router = APIRouter(prefix=prefix, tags=["webhooks"])

    @router.post("/{webhook_id}")
    async def fire_webhook(webhook_id: str, request: Request) -> dict[str, str]:
        spec = registry.get(webhook_id)
        if spec is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown webhook id: {webhook_id!r}",
            )

        body = await request.body()
        provided_signature = request.headers.get(spec.signature_header)
        if not verify_signature(spec.secret, body, provided_signature):
            # Don't echo the expected signature in the error — that'd
            # turn the 401 path into an oracle for forging signatures.
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing webhook signature",
            )

        try:
            payload = await request.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Webhook body is not valid JSON: {exc}",
            ) from exc

        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Webhook body must be a JSON object at the top level",
            )

        try:
            rendered = render_payload(spec.payload_template, payload)
        except KeyError as exc:
            raise HTTPException(
                # ``HTTP_422_UNPROCESSABLE_ENTITY`` was renamed in
                # Starlette to ``HTTP_422_UNPROCESSABLE_CONTENT``; use
                # the literal so we work across versions without
                # tripping the deprecation warning under
                # ``filterwarnings = ["error"]`` in pyproject.
                status_code=422,
                detail=(
                    f"Webhook payload missing template placeholder: {exc.args[0]!r}"
                ),
            ) from exc

        graph = graph_resolver(spec.agent_id)
        thread_id = f"webhook-{spec.id}-{uuid.uuid4().hex[:12]}"
        config = {"configurable": {"thread_id": thread_id}}
        await graph.ainvoke(
            {"messages": [HumanMessage(content=rendered)]},
            config=config,
        )
        logger.info(
            "Webhook fired",
            extra={
                "webhook_id": spec.id,
                "agent_id": spec.agent_id,
                "thread_id": thread_id,
            },
        )
        from langgraph_kit.contrib.schedule import emit_trigger_audit

        await emit_trigger_audit(
            audit_store,
            source="webhook",
            spec_id=spec.id,
            agent_id=spec.agent_id,
            thread_id=thread_id,
        )
        return {"thread_id": thread_id, "agent_id": spec.agent_id}

    return router


__all__ = [
    "WebhookRegistry",
    "WebhookSpec",
    "compute_signature",
    "create_webhook_router",
    "render_payload",
    "verify_signature",
]
