"""FastAPI: mount the kit's agent router and exercise it in-process.

What this shows
---------------
- Wiring :func:`create_agent_router` onto a FastAPI app
- Providing the ``get_current_user`` dependency the router requires
- Registering an agent via :func:`langgraph_kit.register` so the
  router's ``/agents/`` endpoint discovers it
- Calling the router via an ASGI in-process client — no real socket,
  no port to manage, fully hermetic

The full router exposes 11 endpoints (stream / invoke / state / resume /
branch / queue / etc.); this demo hits the discovery endpoint
(``GET /agents/``) and the simple ``invoke`` endpoint to keep output
predictable. See ``tests/e2e/test_fastapi_e2e.py`` for the wider surface.

How to run
----------
    uv run python -m examples.fastapi_minimal_router

Expected output
---------------
    GET /agents/ -> 200
    Registered agents: ['echo']
    POST /agents/echo/invoke -> 200
    assistant: I am the FastAPI demo agent.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from examples._lib import (
    answer,
    banner,
    line,
    make_in_memory_persistence,
    patch_build_llm,
    scripted_llm,
    tmp_workspace,
)


class _DemoUser:
    """Minimal :class:`UserInfo`-compatible object the router will accept."""

    id: str = "demo-user-id"
    email: str = "demo@example.com"


def _resolve_demo_user() -> _DemoUser:
    """FastAPI dependency that returns a fixed user — auth-stub for the demo."""
    return _DemoUser()


async def main() -> None:
    banner("fastapi_minimal_router")

    import httpx  # pyright: ignore[reportMissingImports]
    from fastapi import Depends, FastAPI  # pyright: ignore[reportMissingImports]

    from langgraph_kit import register
    from langgraph_kit.contrib.fastapi import create_agent_router
    from langgraph_kit.graphs.echo_agent import build_graph

    with tmp_workspace() as workspace:
        _ = workspace

        # 1. Register a hermetic echo agent so the router can discover it.
        with patch_build_llm(scripted_llm([answer("I am the FastAPI demo agent.")])):
            checkpointer, store = make_in_memory_persistence()
            graph = build_graph(checkpointer, store)
            register("echo", graph)

            # 2. Build the FastAPI app and mount the router.
            app = FastAPI()
            app.state.store = store
            app.include_router(
                create_agent_router(
                    get_current_user=Annotated[_DemoUser, Depends(_resolve_demo_user)]
                )
            )

            # 3. Drive the app in-process (no port, no socket).
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://demo"
            ) as client:
                # Discovery endpoint
                resp = await client.get("/agents/")
                line(f"GET /agents/ -> {resp.status_code}")
                payload: dict[str, Any] = resp.json()
                line(f"Registered agents: {[a['id'] for a in payload['agents']]}")

                # Invoke endpoint
                body = {
                    "messages": [{"role": "user", "content": "Say hello."}],
                }
                resp = await client.post("/agents/echo/invoke", json=body)
                line(f"POST /agents/echo/invoke -> {resp.status_code}")
                if resp.status_code == 200:
                    msg = resp.json()
                    content = msg.get("response", "") or msg.get("content", "")
                    line(f"assistant: {content}")


if __name__ == "__main__":
    asyncio.run(main())
