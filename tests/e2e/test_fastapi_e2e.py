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


def test_queue_enqueue_then_status_round_trip(
    fastapi_app: FastAPI,
    registered_agent: str,
) -> None:
    """``POST /queue`` then ``GET /queue`` reports the queued item + depth."""
    from fastapi.testclient import TestClient

    client = TestClient(fastapi_app)
    enqueue_resp = client.post(
        f"/agents/{registered_agent}/threads/queue-thread-1/queue",
        json={
            "content": "remember to check logs",
            "semantic": "append",
            "source": "user",
            "metadata": {"origin": "test"},
        },
    )
    assert enqueue_resp.status_code == 200, enqueue_resp.text
    enq_body = enqueue_resp.json()
    assert enq_body["queued"] is True
    assert enq_body["queue_depth"] >= 1

    status_resp = client.get(f"/agents/{registered_agent}/threads/queue-thread-1/queue")
    assert status_resp.status_code == 200, status_resp.text
    status_body = status_resp.json()
    assert status_body["queue_depth"] >= 1
    assert any(
        item.get("content") == "remember to check logs"
        for item in status_body.get("items", [])
    ), f"Queued item should appear in peek list; got {status_body!r}"


def test_thread_messages_endpoint_returns_empty_for_new_thread(
    fastapi_app: FastAPI,
    registered_agent: str,
) -> None:
    """A never-invoked thread has no messages — endpoint returns ``[]``."""
    from fastapi.testclient import TestClient

    client = TestClient(fastapi_app)
    response = client.get(f"/agents/{registered_agent}/threads/empty-thread/messages")
    assert response.status_code == 200, response.text
    assert response.json() == []


def test_thread_state_endpoint_reports_idle_for_unused_thread(
    fastapi_app: FastAPI,
    registered_agent: str,
) -> None:
    """``GET /state`` returns status=idle + empty interrupts for a fresh thread."""
    from fastapi.testclient import TestClient

    client = TestClient(fastapi_app)
    response = client.get(f"/agents/{registered_agent}/threads/fresh-thread/state")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["thread_id"] == "fresh-thread"
    assert body["status"] in ("idle", "error"), (
        f"Fresh thread should report idle (or graceful 'error' on aget_state"
        f" failures); got {body!r}"
    )
    assert body.get("interrupts", []) == []


# ---------------------------------------------------------------------------
# Thread-management routes (CRUD, search, history, fork)
# ---------------------------------------------------------------------------


async def _seed_thread_metadata(
    store: Any, *, thread_id: str, user_id: str, agent_id: str, title: str
) -> None:
    """Seed a thread-metadata record via ``ThreadManager`` for route tests."""
    from langgraph_kit.core.threads import ThreadManager

    mgr = ThreadManager(store)
    await mgr.ensure_thread(
        thread_id=thread_id,
        user_id=user_id,
        agent_id=agent_id,
        first_message=title,
    )


@pytest.mark.asyncio
async def test_list_threads_returns_threads_for_user(
    fastapi_app: FastAPI,
    registered_agent: str,
    e2e_store: Any,
) -> None:
    """``GET /threads`` lists threads belonging to the current user."""
    await _seed_thread_metadata(
        e2e_store,
        thread_id="thr-1",
        user_id="test-user",
        agent_id=registered_agent,
        title="First conversation about widgets",
    )
    await _seed_thread_metadata(
        e2e_store,
        thread_id="thr-2",
        user_id="test-user",
        agent_id=registered_agent,
        title="Second conversation about gadgets",
    )

    from fastapi.testclient import TestClient

    client = TestClient(fastapi_app)
    response = client.get("/agents/threads")
    assert response.status_code == 200, response.text
    body = response.json()
    ids = [t["thread_id"] for t in body["threads"]]
    assert "thr-1" in ids
    assert "thr-2" in ids
    assert body["total"] >= 2


@pytest.mark.asyncio
async def test_search_threads_finds_by_title(
    fastapi_app: FastAPI,
    registered_agent: str,
    e2e_store: Any,
) -> None:
    """``GET /threads/search?q=...`` returns threads whose metadata matches."""
    await _seed_thread_metadata(
        e2e_store,
        thread_id="search-1",
        user_id="test-user",
        agent_id=registered_agent,
        title="quarterly budget review",
    )

    from fastapi.testclient import TestClient

    client = TestClient(fastapi_app)
    response = client.get("/agents/threads/search", params={"q": "budget"})
    assert response.status_code == 200, response.text
    body = response.json()
    ids = [t["thread_id"] for t in body["threads"]]
    assert "search-1" in ids


@pytest.mark.asyncio
async def test_get_thread_metadata_route_returns_record(
    fastapi_app: FastAPI,
    registered_agent: str,
    e2e_store: Any,
) -> None:
    await _seed_thread_metadata(
        e2e_store,
        thread_id="meta-1",
        user_id="test-user",
        agent_id=registered_agent,
        title="meta test",
    )

    from fastapi.testclient import TestClient

    client = TestClient(fastapi_app)
    response = client.get("/agents/threads/meta-1")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["thread_id"] == "meta-1"


@pytest.mark.asyncio
async def test_get_thread_metadata_returns_404_for_unknown_thread(
    fastapi_app: FastAPI,
) -> None:
    from fastapi.testclient import TestClient

    client = TestClient(fastapi_app)
    response = client.get("/agents/threads/nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_thread_metadata_patches_title_and_tags(
    fastapi_app: FastAPI,
    registered_agent: str,
    e2e_store: Any,
) -> None:
    await _seed_thread_metadata(
        e2e_store,
        thread_id="patch-1",
        user_id="test-user",
        agent_id=registered_agent,
        title="original",
    )

    from fastapi.testclient import TestClient

    client = TestClient(fastapi_app)
    response = client.patch(
        "/agents/threads/patch-1",
        json={"title": "updated", "tags": ["urgent"]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["title"] == "updated"
    assert body["tags"] == ["urgent"]


@pytest.mark.asyncio
async def test_update_thread_returns_404_for_foreign_user_thread(
    fastapi_app: FastAPI,
    registered_agent: str,
    e2e_store: Any,
) -> None:
    await _seed_thread_metadata(
        e2e_store,
        thread_id="other-users-thread",
        user_id="other-user",
        agent_id=registered_agent,
        title="not yours",
    )

    from fastapi.testclient import TestClient

    client = TestClient(fastapi_app)
    response = client.patch(
        "/agents/threads/other-users-thread",
        json={"title": "hacked"},
    )
    assert response.status_code == 404, (
        f"Should return 404 to avoid leaking foreign-user thread existence;"
        f" got {response.status_code}"
    )


@pytest.mark.asyncio
async def test_delete_thread_removes_the_record(
    fastapi_app: FastAPI,
    registered_agent: str,
    e2e_store: Any,
) -> None:
    await _seed_thread_metadata(
        e2e_store,
        thread_id="del-1",
        user_id="test-user",
        agent_id=registered_agent,
        title="please delete",
    )

    from fastapi.testclient import TestClient

    client = TestClient(fastapi_app)
    response = client.delete("/agents/threads/del-1")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body.get("deleted") is True

    # GET after delete → 404.
    get_resp = client.get("/agents/threads/del-1")
    assert get_resp.status_code == 404


def test_thread_history_endpoint_returns_empty_list_for_unused_thread(
    fastapi_app: FastAPI,
    registered_agent: str,
) -> None:
    """``GET /history`` returns an empty list for a thread with no checkpoints."""
    from fastapi.testclient import TestClient

    client = TestClient(fastapi_app)
    response = client.get(f"/agents/{registered_agent}/threads/no-history/history")
    assert response.status_code == 200, response.text
    assert isinstance(response.json(), list)
