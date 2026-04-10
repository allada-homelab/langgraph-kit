"""A2A (Agent-to-Agent) protocol integration for langgraph-kit.

Exposes registered agents as A2A endpoints with auto-generated Agent Cards.

Usage::

    from langgraph_kit.contrib.a2a import create_a2a_router, build_agent_card
    app.include_router(create_a2a_router())
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from langgraph_kit.registry import get, get_metadata, list_agents

logger = logging.getLogger(__name__)



def build_agent_card(agent_id: str, base_url: str) -> dict[str, Any]:
    """Build an A2A Agent Card JSON from registry metadata.

    Parameters
    ----------
    agent_id:
        The registered agent ID.
    base_url:
        The base URL where the A2A endpoint is hosted.
    """
    meta = get_metadata(agent_id)
    return {
        "name": agent_id.replace("-", " ").title(),
        "description": meta.description or f"Agent: {agent_id}",
        "url": f"{base_url}/a2a/{agent_id}",
        "version": meta.version,
        "capabilities": {
            "streaming": "streaming" in meta.capabilities,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "skills": [
            {
                "id": agent_id,
                "name": agent_id.replace("-", " ").title(),
                "description": meta.description or f"Invoke {agent_id}",
                "tags": meta.tags,
            }
        ],
        "defaultInputModes": meta.input_modes,
        "defaultOutputModes": meta.output_modes,
    }


def build_aggregated_card(base_url: str) -> dict[str, Any]:
    """Build a single Agent Card aggregating all registered agents as skills."""
    agents = list_agents()
    skills = []
    for agent in agents:
        agent_id = agent["id"]
        meta = get_metadata(agent_id)
        skills.append({
            "id": agent_id,
            "name": agent["name"],
            "description": meta.description or f"Agent: {agent_id}",
            "tags": meta.tags,
        })

    return {
        "name": "LangGraph Kit Agent Hub",
        "description": "Multi-agent platform with specialized AI assistants",
        "url": f"{base_url}/a2a",
        "version": "1.0.0",
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "skills": skills,
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
    }


async def invoke_agent_a2a(
    agent_id: str,
    message_text: str,
    *,
    thread_id: str = "",
) -> dict[str, Any]:
    """Invoke a registered agent with a text message and return A2A Task result.

    Returns a simplified A2A Task dict with status and artifacts.
    """
    from langchain_core.messages import (
        HumanMessage,  # pyright: ignore[reportMissingModuleSource]
    )

    graph = get(agent_id)
    tid = thread_id or str(uuid4())
    config: dict[str, Any] = {"configurable": {"thread_id": tid}}
    input_data = {"messages": [HumanMessage(content=message_text)]}

    task_id = str(uuid4())

    try:
        result = await graph.ainvoke(input_data, config=config)
        last = result["messages"][-1]
        content = last.content if hasattr(last, "content") else str(last)

        return {
            "id": task_id,
            "contextId": tid,
            "status": {"state": "completed"},
            "artifacts": [
                {
                    "artifactId": str(uuid4()),
                    "parts": [{"kind": "text", "text": content}],
                }
            ],
        }
    except Exception as exc:
        return {
            "id": task_id,
            "contextId": tid,
            "status": {
                "state": "failed",
                "message": {
                    "messageId": str(uuid4()),
                    "role": "agent",
                    "parts": [{"kind": "text", "text": str(exc)}],
                },
            },
        }


def create_a2a_router(*, get_current_user: Any = None) -> Any:  # noqa: ARG001
    """Create a FastAPI router for A2A protocol endpoints.

    Adds:
    - ``GET /.well-known/agent.json`` — aggregated Agent Card
    - ``POST /a2a/{agent_id}`` — invoke agent via A2A message
    """
    from fastapi import APIRouter, HTTPException, Request, status
    from fastapi.responses import JSONResponse

    router = APIRouter(tags=["a2a"])

    @router.get("/.well-known/agent.json")
    async def get_agent_card(request: Request) -> JSONResponse:
        """Return the aggregated A2A Agent Card."""
        base_url = str(request.base_url).rstrip("/")
        card = build_aggregated_card(base_url)
        return JSONResponse(content=card)

    @router.post("/a2a/{agent_id}")
    async def a2a_invoke(agent_id: str, request: Request) -> JSONResponse:
        """Handle an A2A message send request (simplified JSON-RPC)."""
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON body",
            ) from None

        # Extract text from message parts
        message_text = ""
        if "params" in body:
            # JSON-RPC format
            params = body["params"]
            msg = params.get("message", {})
            parts = msg.get("parts", [])
            for part in parts:
                if part.get("kind") == "text":
                    message_text += part.get("text", "")
        elif "message" in body:
            # Direct format
            parts = body["message"].get("parts", [])
            for part in parts:
                if part.get("kind") == "text":
                    message_text += part.get("text", "")
        elif "content" in body:
            # Simple format
            message_text = body["content"]

        if not message_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text content found in message",
            ) from None

        try:
            get(agent_id)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{agent_id}' not found",
            ) from None

        context_id = body.get("params", {}).get("message", {}).get("contextId", "")
        result = await invoke_agent_a2a(agent_id, message_text, thread_id=context_id)
        return JSONResponse(content=result)

    return router
