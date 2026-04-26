"""Tests for ``langgraph_kit.contrib.webhook`` — issue #19 v1.

Covers the pure helpers (``compute_signature`` / ``verify_signature`` /
``render_payload``) and the FastAPI router end-to-end via TestClient
with a mock graph in place of a real compiled LangGraph.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from langgraph_kit.contrib.webhook import (
    WebhookRegistry,
    WebhookSpec,
    compute_signature,
    create_webhook_router,
    render_payload,
    verify_signature,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeGraph:
    """Records each ``ainvoke`` call so tests can assert on what was sent."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, Any] | None]] = []

    async def ainvoke(
        self, input_data: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.calls.append((input_data, config))
        return {"messages": []}


def _build_app(
    registry: WebhookRegistry,
    graphs: dict[str, _FakeGraph],
    *,
    audit_store: object = None,
) -> tuple[FastAPI, dict[str, _FakeGraph]]:
    app = FastAPI()
    app.include_router(
        create_webhook_router(
            registry,
            graph_resolver=lambda agent_id: graphs[agent_id],
            audit_store=audit_store,
        )
    )
    return app, graphs


# ---------------------------------------------------------------------------
# Pure-helper coverage (no FastAPI required).
# ---------------------------------------------------------------------------


class TestSignatureHelpers:
    def test_compute_signature_round_trip(self) -> None:
        sig = compute_signature("secret", b'{"event":"x"}')
        assert sig.startswith("sha256=")
        # Same body → same signature (HMAC determinism).
        assert sig == compute_signature("secret", b'{"event":"x"}')

    def test_verify_signature_happy_path(self) -> None:
        body = b'{"event":"x"}'
        sig = compute_signature("secret", body)
        assert verify_signature("secret", body, sig) is True

    def test_verify_signature_wrong_secret_rejected(self) -> None:
        body = b'{"event":"x"}'
        sig = compute_signature("right", body)
        assert verify_signature("wrong", body, sig) is False

    def test_verify_signature_tampered_body_rejected(self) -> None:
        sig = compute_signature("secret", b'{"event":"original"}')
        assert verify_signature("secret", b'{"event":"tampered"}', sig) is False

    def test_verify_signature_missing_header_rejected(self) -> None:
        assert verify_signature("secret", b"body", None) is False
        assert verify_signature("secret", b"body", "") is False

    def test_verify_signature_missing_prefix_rejected(self) -> None:
        body = b"body"
        digest = compute_signature("secret", body).removeprefix("sha256=")
        # Bare hex without the ``sha256=`` prefix should fail — we don't
        # try to be clever about format detection.
        assert verify_signature("secret", body, digest) is False


class TestPayloadTemplating:
    def test_render_payload_simple_substitution(self) -> None:
        rendered = render_payload("Hello, {name}!", {"name": "Alice"})
        assert rendered == "Hello, Alice!"

    def test_render_payload_multiple_placeholders(self) -> None:
        rendered = render_payload(
            "{amount} {currency} from {customer_email}",
            {"amount": 42, "currency": "USD", "customer_email": "a@b.co"},
        )
        assert rendered == "42 USD from a@b.co"

    def test_render_payload_missing_key_raises(self) -> None:
        with pytest.raises(KeyError):
            render_payload("{missing}", {"present": "x"})

    def test_render_payload_no_placeholders_passes_through(self) -> None:
        # Templates that don't reference the payload at all are valid —
        # useful for "fire on any event with this fixed message" specs.
        assert render_payload("ping", {"unrelated": "x"}) == "ping"


# ---------------------------------------------------------------------------
# WebhookRegistry coverage.
# ---------------------------------------------------------------------------


class TestWebhookRegistry:
    def test_register_then_get_returns_spec(self) -> None:
        registry = WebhookRegistry()
        spec = WebhookSpec(id="x", agent_id="agent", secret="s")
        registry.register(spec)
        assert registry.get("x") is spec

    def test_get_unknown_id_returns_none(self) -> None:
        assert WebhookRegistry().get("missing") is None

    def test_register_replaces_existing_id(self) -> None:
        registry = WebhookRegistry()
        registry.register(WebhookSpec(id="x", agent_id="a", secret="old"))
        registry.register(WebhookSpec(id="x", agent_id="a", secret="new"))
        spec = registry.get("x")
        assert spec is not None
        assert spec.secret == "new"  # noqa: S105 - test fixture, not a real secret

    def test_remove_drops_spec(self) -> None:
        registry = WebhookRegistry()
        registry.register(WebhookSpec(id="x", agent_id="a", secret="s"))
        registry.remove("x")
        assert registry.get("x") is None

    def test_list_ids(self) -> None:
        registry = WebhookRegistry()
        registry.register(WebhookSpec(id="a", agent_id="x", secret="s"))
        registry.register(WebhookSpec(id="b", agent_id="x", secret="s"))
        assert sorted(registry.list_ids()) == ["a", "b"]

    def test_webhook_spec_is_frozen(self) -> None:
        spec = WebhookSpec(id="x", agent_id="a", secret="s")
        with pytest.raises(Exception):  # noqa: B017,PT011 - Pydantic raises ValidationError on frozen-set
            spec.secret = "changed"  # type: ignore[misc]  # noqa: S105 - test fixture


# ---------------------------------------------------------------------------
# Router behavior end-to-end.
# ---------------------------------------------------------------------------


class TestWebhookRouter:
    @staticmethod
    def _setup() -> tuple[TestClient, _FakeGraph, WebhookSpec]:
        spec = WebhookSpec(
            id="hook1",
            agent_id="payment-agent",
            secret="whsec_top_secret",
            payload_template="payment {amount} {currency}",
        )
        registry = WebhookRegistry()
        registry.register(spec)
        graph = _FakeGraph()
        app, _ = _build_app(registry, {"payment-agent": graph})
        return TestClient(app), graph, spec

    def test_valid_signature_invokes_agent_and_returns_thread_id(self) -> None:
        client, graph, spec = self._setup()
        body = json.dumps({"amount": 42, "currency": "USD"}).encode()
        sig = compute_signature(spec.secret, body)

        resp = client.post(
            f"/webhooks/{spec.id}",
            content=body,
            headers={
                spec.signature_header: sig,
                "content-type": "application/json",
            },
        )
        assert resp.status_code == 200
        body_resp = resp.json()
        assert body_resp["agent_id"] == "payment-agent"
        assert body_resp["thread_id"].startswith(f"webhook-{spec.id}-")

        # Agent received the rendered template, not the raw payload.
        assert len(graph.calls) == 1
        input_data, config = graph.calls[0]
        msgs = input_data["messages"]
        assert len(msgs) == 1
        assert msgs[0].content == "payment 42 USD"
        assert config is not None
        assert config["configurable"]["thread_id"] == body_resp["thread_id"]

    def test_invalid_signature_rejected_without_invoking_agent(self) -> None:
        client, graph, spec = self._setup()
        body = json.dumps({"amount": 42, "currency": "USD"}).encode()

        resp = client.post(
            f"/webhooks/{spec.id}",
            content=body,
            headers={
                spec.signature_header: "sha256=deadbeef",
                "content-type": "application/json",
            },
        )
        assert resp.status_code == 401
        # Critical: agent was NOT invoked on auth failure.
        assert graph.calls == []

    def test_missing_signature_header_rejected(self) -> None:
        client, graph, spec = self._setup()
        body = json.dumps({"amount": 42, "currency": "USD"}).encode()
        resp = client.post(
            f"/webhooks/{spec.id}",
            content=body,
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 401
        assert graph.calls == []

    def test_unknown_webhook_id_returns_404(self) -> None:
        client, _, spec = self._setup()
        body = b"{}"
        resp = client.post(
            "/webhooks/nonexistent",
            content=body,
            headers={spec.signature_header: compute_signature(spec.secret, body)},
        )
        assert resp.status_code == 404

    def test_invalid_json_body_returns_400(self) -> None:
        client, graph, spec = self._setup()
        body = b"not json at all"
        sig = compute_signature(spec.secret, body)
        resp = client.post(
            f"/webhooks/{spec.id}",
            content=body,
            headers={
                spec.signature_header: sig,
                "content-type": "application/json",
            },
        )
        assert resp.status_code == 400
        assert graph.calls == []

    def test_non_object_json_body_returns_400(self) -> None:
        """Top-level array / string / number payloads aren't supported."""
        client, graph, spec = self._setup()
        body = b'["not", "an", "object"]'
        sig = compute_signature(spec.secret, body)
        resp = client.post(
            f"/webhooks/{spec.id}",
            content=body,
            headers={
                spec.signature_header: sig,
                "content-type": "application/json",
            },
        )
        assert resp.status_code == 400
        assert graph.calls == []

    def test_template_missing_placeholder_returns_422(self) -> None:
        client, graph, spec = self._setup()
        # Missing 'amount' — template references {amount} and {currency}.
        body = json.dumps({"currency": "USD"}).encode()
        sig = compute_signature(spec.secret, body)
        resp = client.post(
            f"/webhooks/{spec.id}",
            content=body,
            headers={
                spec.signature_header: sig,
                "content-type": "application/json",
            },
        )
        assert resp.status_code == 422
        assert "amount" in resp.json()["detail"]
        assert graph.calls == []

    def test_custom_signature_header_honored(self) -> None:
        """Specs can override ``signature_header`` for non-GitHub services."""
        spec = WebhookSpec(
            id="stripe-hook",
            agent_id="agent",
            secret="s",
            payload_template="{event}",
            signature_header="X-Stripe-Signature",
        )
        registry = WebhookRegistry()
        registry.register(spec)
        graph = _FakeGraph()
        app, _ = _build_app(registry, {"agent": graph})
        client = TestClient(app)

        body = json.dumps({"event": "charge.succeeded"}).encode()
        # Sending under the default header should fail (auth missing
        # under the configured header name).
        resp = client.post(
            f"/webhooks/{spec.id}",
            content=body,
            headers={
                "X-Hub-Signature-256": compute_signature(spec.secret, body),
                "content-type": "application/json",
            },
        )
        assert resp.status_code == 401

        # Under the configured header name, it succeeds.
        resp = client.post(
            f"/webhooks/{spec.id}",
            content=body,
            headers={
                spec.signature_header: compute_signature(spec.secret, body),
                "content-type": "application/json",
            },
        )
        assert resp.status_code == 200
        assert len(graph.calls) == 1
        assert graph.calls[0][0]["messages"][0].content == "charge.succeeded"

    def test_thread_ids_are_unique_per_fire(self) -> None:
        """Two consecutive fires get distinct thread ids."""
        client, _graph, spec = self._setup()
        body = json.dumps({"amount": 1, "currency": "USD"}).encode()
        sig = compute_signature(spec.secret, body)
        resp1 = client.post(
            f"/webhooks/{spec.id}",
            content=body,
            headers={spec.signature_header: sig, "content-type": "application/json"},
        )
        resp2 = client.post(
            f"/webhooks/{spec.id}",
            content=body,
            headers={spec.signature_header: sig, "content-type": "application/json"},
        )
        assert resp1.status_code == resp2.status_code == 200
        assert resp1.json()["thread_id"] != resp2.json()["thread_id"]


class TestWebhookAudit:
    def test_fire_emits_audit_entry_when_audit_store_set(self, mock_store: Any) -> None:
        """A successful webhook fire writes an AGENT_INVOKE audit entry."""
        import asyncio

        from langgraph_kit.core.audit import AuditAction, AuditStore

        audit_store = AuditStore(mock_store)
        spec = WebhookSpec(
            id="github-push",
            agent_id="reviewer",
            secret="whsec_audit",
            payload_template="A new event arrived: {event}",
        )
        registry = WebhookRegistry()
        registry.register(spec)
        app, _ = _build_app(
            registry, {"reviewer": _FakeGraph()}, audit_store=audit_store
        )
        client = TestClient(app)

        body = b'{"event":"push"}'
        sig = compute_signature(spec.secret, body)
        resp = client.post(
            f"/webhooks/{spec.id}",
            content=body,
            headers={spec.signature_header: sig, "content-type": "application/json"},
        )
        assert resp.status_code == 200

        entries = asyncio.run(
            audit_store.query(action=AuditAction.AGENT_INVOKE, limit=10)
        )
        assert len(entries) == 1
        entry = entries[0]
        assert entry.actor == "trigger:webhook"
        assert entry.metadata["trigger_source"] == "webhook"
        assert entry.metadata["trigger_spec_id"] == "github-push"
        assert entry.metadata["agent_id"] == "reviewer"
