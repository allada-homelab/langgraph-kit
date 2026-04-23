"""Cluster J — FastAPI contrib end-to-end HTTP round-trip.

``create_agent_router`` exposes registered kit graphs as HTTP routes.
Existing unit coverage at ``test_contrib_fastapi.py`` verifies the
OpenAPI schema renders and the routes are registered. These e2e tests
take the next step: drive a real HTTP request through FastAPI's
TestClient, invoke a registered kit graph via the ``/invoke`` endpoint,
and assert the HTTP response carries the agent's final message.

This catches regressions where the kit's invoke shape changes (for
example the result dict key structure) but the HTTP adapter isn't
updated in lockstep.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest
from fastapi import Depends, FastAPI

from langgraph_kit.contrib.fastapi import (
    _checkpointer_var,
    _store_var,
    create_agent_router,
)
from langgraph_kit.graphs._builder import build_deep_agent
from langgraph_kit.registry import AgentMetadata, register
from tests.e2e.helpers import answer, scripted_llm

pytestmark = pytest.mark.e2e


class _StubUser:
    """Minimal user object that satisfies observability.UserInfo protocol."""

    id = "test-user"
    email = "test@example.test"


def _get_user() -> _StubUser:
    return _StubUser()


@pytest.fixture
def registered_agent(
    checkpointer: Any,
    e2e_store: Any,
    patched_build_llm: Any,
) -> Iterator[str]:
    """Build + register a minimal kit agent scoped to this test."""
    agent_id = "fastapi-e2e-agent"
    scripted = scripted_llm([answer("fastapi-response")])
    with patched_build_llm(scripted):
        graph, dispatcher = build_deep_agent(
            agent_name=agent_id,
            core_sections=[],
            subagents=[],
            checkpointer=checkpointer,
            store=e2e_store,
        )
    register(
        agent_id,
        graph,
        command_dispatcher=dispatcher,
        metadata=AgentMetadata(description="fastapi e2e test agent"),
    )
    try:
        yield agent_id
    finally:
        from langgraph_kit import registry as registry_mod

        registry_mod._registry.pop(agent_id, None)
        registry_mod._dispatchers.pop(agent_id, None)
        registry_mod._metadata.pop(agent_id, None)


@pytest.fixture
def fastapi_app(
    registered_agent: str,
    checkpointer: Any,
    e2e_store: Any,
) -> Iterator[FastAPI]:
    """Build a FastAPI app with the agent router mounted.

    Manually seeds the store/checkpointer contextvars rather than using
    the full ``create_app_lifespan`` so the test doesn't depend on the
    configure_from_settings path.
    """
    current_user: Any = Annotated[Any, Depends(_get_user)]
    app = FastAPI()
    app.include_router(create_agent_router(get_current_user=current_user))
    app.state.store = e2e_store
    app.state.checkpointer = checkpointer

    ckpt_token = _checkpointer_var.set(checkpointer)
    store_token = _store_var.set(e2e_store)
    try:
        yield app
    finally:
        _store_var.reset(store_token)
        _checkpointer_var.reset(ckpt_token)


@pytest.mark.asyncio
async def test_invoke_endpoint_returns_agent_response(
    fastapi_app: FastAPI,
    registered_agent: str,
) -> None:
    """POST /agents/{id}/invoke returns the agent's final message.

    Uses TestClient for a real HTTP round-trip. The graph runs inside
    the app's request handler, reaches the scripted LLM, and returns
    its response as the ``content`` field of ``InvokeResponse``.
    """
    from fastapi.testclient import TestClient

    # Patch observability so we don't need a real Langfuse.
    with patch(
        "langgraph_kit.contrib.fastapi.build_agent_run_config",
        return_value={"configurable": {"thread_id": "fastapi-invoke-1"}},
    ):
        client = TestClient(fastapi_app)
        response = client.post(
            f"/agents/{registered_agent}/invoke",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "thread_id": "fastapi-invoke-1",
            },
        )

    assert response.status_code == 200, response.text
    data = response.json()
    assert "fastapi-response" in data["content"], (
        f"Invoke endpoint didn't carry the scripted final message. Got: {data!r}"
    )
    assert data["thread_id"] == "fastapi-invoke-1"


def test_list_agents_endpoint_includes_registered_agent(
    fastapi_app: FastAPI,
    registered_agent: str,
) -> None:
    """GET /agents/ lists every registered agent including ours."""
    from fastapi.testclient import TestClient

    client = TestClient(fastapi_app)
    response = client.get("/agents/")
    assert response.status_code == 200, response.text
    agents = response.json().get("agents", [])
    ids = [a["id"] for a in agents]
    assert registered_agent in ids, (
        f"Registered agent should appear in /agents/; got ids {ids!r}"
    )


def test_invoke_endpoint_returns_404_for_unknown_agent(
    fastapi_app: FastAPI,
) -> None:
    """Invoking an unregistered agent returns a 404, not a server error."""
    from fastapi.testclient import TestClient

    client = TestClient(fastapi_app)
    response = client.post(
        "/agents/ghost-agent/invoke",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404, response.text
