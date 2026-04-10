"""Optional FastAPI integration for langgraph-kit.

Provides:
- ``create_agent_router()`` — route factory for agent endpoints
- ``create_app_lifespan()`` — one-call FastAPI lifespan handler
- ``StoreDep`` / ``CheckpointerDep`` — FastAPI dependency injection

Usage::

    from langgraph_kit import configure_from_settings
    from langgraph_kit.contrib.fastapi import create_agent_router, create_app_lifespan

    configure_from_settings(settings, field_map={"database_url": "SQLALCHEMY_DATABASE_URI"})
    app = FastAPI(lifespan=create_app_lifespan())
    app.include_router(create_agent_router(get_current_user=CurrentUser), prefix="/api/v1")
"""

from __future__ import annotations

import contextvars
import json
import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from langgraph_kit._config import get_config
from langgraph_kit.core.hitl.models import (
    ResumeRequest,
    ThreadStateResponse,
)
from langgraph_kit.models import (
    AgentInfo,
    AgentListResponse,
    ChatMessage,
    CheckpointInfo,
    ForkRequest,
    ForkResponse,
    InvokeRequest,
    InvokeResponse,
    QueueMessageRequest,
    QueueMessageResponse,
    QueueStatusResponse,
    ThreadListResponse,
    ThreadMetadataResponse,
    ThreadUpdateRequest,
)
from langgraph_kit.observability import build_agent_run_config, flush_langfuse
from langgraph_kit.registry import get, get_dispatcher, list_agents
from langgraph_kit.streaming import stream_agent_events

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context vars — set by create_app_lifespan(), read by DI deps
# ---------------------------------------------------------------------------

_store_var: contextvars.ContextVar[Any] = contextvars.ContextVar("langgraph_store")
_checkpointer_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "langgraph_checkpointer"
)


# ---------------------------------------------------------------------------
# FastAPI dependency injection
# ---------------------------------------------------------------------------


def get_store() -> Any:
    """FastAPI dependency that returns the LangGraph store."""
    try:
        return _store_var.get()
    except LookupError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Store not available — agent lifespan not initialized",
        ) from None


def get_checkpointer() -> Any:
    """FastAPI dependency that returns the LangGraph checkpointer."""
    try:
        return _checkpointer_var.get()
    except LookupError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Checkpointer not available — agent lifespan not initialized",
        ) from None


StoreDep = Annotated[Any, Depends(get_store)]
CheckpointerDep = Annotated[Any, Depends(get_checkpointer)]


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------


def create_app_lifespan(
    *,
    register_agents: Callable[..., Any] | None = None,
) -> Any:
    """Return a FastAPI lifespan that initialises the full agent stack.

    The lifespan handles:
    1. ``init_langfuse()``
    2. ``create_persistence()`` → (checkpointer, store)
    3. MCP server connections (if configured)
    4. Agent graph registration
    5. yield (app serves requests)
    6. MCP close + ``shutdown_langfuse()``

    Parameters
    ----------
    register_agents:
        Optional ``(checkpointer, store, mcp_tools) -> None`` callback.
        If omitted, calls ``langgraph_kit.graphs.register_all()``.
    """
    from fastapi import FastAPI

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        from langgraph_kit.observability import init_langfuse, shutdown_langfuse
        from langgraph_kit.persistence import create_persistence

        init_langfuse()
        mcp_mgr = None
        try:
            async with create_persistence() as (checkpointer, store):
                # Set context vars (for DI deps)
                _checkpointer_var.set(checkpointer)
                _store_var.set(store)
                # Set app.state (backwards compat)
                app.state.checkpointer = checkpointer
                app.state.store = store

                # MCP servers
                mcp_tools: list[Any] = []
                config = get_config()
                if config.mcp_servers:
                    from langgraph_kit.core.plugins.mcp_client import (
                        MCPClientManager,
                    )

                    mcp_mgr = MCPClientManager(config.mcp_servers)
                    contribution = await mcp_mgr.connect_all()
                    mcp_tools = contribution.tools
                app.state.mcp_tools = mcp_tools

                # Register agents
                if register_agents is not None:
                    register_agents(checkpointer, store, mcp_tools)
                else:
                    from langgraph_kit.graphs import register_all

                    register_all(checkpointer, store, mcp_tools=mcp_tools)

                yield
        finally:
            if mcp_mgr is not None:
                await mcp_mgr.close()
            shutdown_langfuse()

    return lifespan


def _to_lc_messages(messages: list[ChatMessage]) -> list[Any]:
    """Convert Pydantic ChatMessage list to LangChain message objects."""
    from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
        AIMessage,
        HumanMessage,
        SystemMessage,
    )

    result: list[Any] = []
    for m in messages:
        if m.role == "user":
            if m.attachments:
                content = _build_multipart_content(m)
                result.append(HumanMessage(content=content))
            else:
                result.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            result.append(AIMessage(content=m.content))
        elif m.role == "system":
            result.append(SystemMessage(content=m.content))
    return result


def _build_multipart_content(m: ChatMessage) -> list[dict[str, Any]]:
    """Build a LangChain multi-part content list from a message with attachments."""
    parts: list[dict[str, Any]] = []
    if m.content.strip():
        parts.append({"type": "text", "text": m.content})
    for att in m.attachments:
        if att.type.startswith("image/"):
            parts.append({"type": "image_url", "image_url": {"url": att.data_url}})
        elif att.type == "application/pdf":
            # Many LLMs support PDF via the image_url path with data URLs
            parts.append({"type": "image_url", "image_url": {"url": att.data_url}})
        else:
            # Text-based files: decode base64 and include as text
            text_content = _decode_data_url_text(att.data_url)
            parts.append({"type": "text", "text": f"[File: {att.name}]\n{text_content}"})
    return parts


def _extract_text_content(content: Any) -> str:
    """Extract text from LangChain message content (str or multi-part list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    parts.append("[image]")
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(parts)
    return str(content)


def _decode_data_url_text(data_url: str) -> str:
    """Decode a base64 data URL to a UTF-8 string.

    Falls back to the raw data URL if decoding fails.
    """
    import base64

    try:
        # data:text/plain;base64,SGVsbG8=
        if ";base64," in data_url:
            encoded = data_url.split(";base64,", 1)[1]
            return base64.b64decode(encoded).decode("utf-8", errors="replace")
    except Exception:
        logger.debug("Failed to decode data URL", exc_info=True)
    return data_url


def _get_graph(agent_id: str) -> Any:
    """Resolve agent_id to a compiled graph, or raise 404."""
    try:
        return get(agent_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_id}' not found",
        ) from None


async def _try_command(agent_id: str, messages: list[ChatMessage]) -> str | None:
    """If the last message is a slash command, dispatch it and return SSE output."""
    if not messages:
        return None
    last = messages[-1]
    if last.role != "user" or not last.content.strip().startswith("/"):
        return None
    dispatcher = get_dispatcher(agent_id)
    if dispatcher is None or not dispatcher.is_command(last.content.strip()):
        return None

    lc_messages = _to_lc_messages(messages)
    result = await dispatcher.dispatch(
        last.content.strip(),
        context={"messages": lc_messages, "state": {}},
    )
    if not result.handled:
        return None

    return f"data: {json.dumps({'command_result': {'output': result.output}})}\n\ndata: [DONE]\n\n"


def _get_store(request: Request) -> Any:
    """Get the LangGraph Store from contextvar or app state (fallback)."""
    try:
        return _store_var.get()
    except LookupError:
        store = getattr(request.app.state, "store", None)
        if store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Store not available",
            )
        return store


def create_agent_router(*, get_current_user: Any) -> APIRouter:
    """Create a FastAPI router for agent endpoints.

    Parameters
    ----------
    get_current_user:
        A FastAPI dependency (e.g. an ``Annotated`` type alias) that
        resolves the current authenticated user.  The resolved user
        object must satisfy the :class:`langgraph_kit.observability.UserInfo`
        protocol (i.e. have ``id`` and ``email`` attributes).
    """
    router = APIRouter(prefix="/agents", tags=["agents"])
    CurrentUser = get_current_user

    # ------------------------------------------------------------------
    # List agents
    # ------------------------------------------------------------------

    @router.get("/", response_model=AgentListResponse)
    async def get_agents(_current_user: CurrentUser) -> AgentListResponse:  # type: ignore[valid-type]
        """Return all registered agents."""
        return AgentListResponse(agents=[AgentInfo(**a) for a in list_agents()])

    # ------------------------------------------------------------------
    # Stream
    # ------------------------------------------------------------------

    @router.post("/{agent_id}/stream", response_class=StreamingResponse)
    async def stream(
        agent_id: str,
        request: InvokeRequest,
        http_request: Request,
        current_user: CurrentUser,  # type: ignore[valid-type]
    ) -> StreamingResponse:
        """Stream LLM tokens as Server-Sent Events."""
        cmd_output = await _try_command(agent_id, request.messages)
        if cmd_output is not None:

            async def _cmd_stream() -> Any:
                yield cmd_output

            return StreamingResponse(
                _cmd_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        graph = _get_graph(agent_id)
        tid = request.thread_id or str(uuid4())
        input_data: dict[str, Any] = {"messages": _to_lc_messages(request.messages)}
        config = build_agent_run_config(
            agent_id=agent_id,
            thread_id=tid,
            current_user=current_user,
            endpoint="stream",
        )
        if request.checkpoint_id:
            config["configurable"]["checkpoint_id"] = request.checkpoint_id
        store = _store_var.get(None) or getattr(http_request.app.state, "store", None)

        # Track thread metadata
        if store is not None:
            first_msg = request.messages[-1].content if request.messages else None
            await _ensure_thread(store, tid, current_user, agent_id, first_msg)

        return StreamingResponse(
            stream_agent_events(graph, input_data, config, store=store),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------
    # Invoke
    # ------------------------------------------------------------------

    @router.post("/{agent_id}/invoke", response_model=InvokeResponse)
    async def invoke(
        agent_id: str,
        request: InvokeRequest,
        current_user: CurrentUser,  # type: ignore[valid-type]
    ) -> InvokeResponse:
        """Invoke the agent and return the full response."""
        graph = _get_graph(agent_id)
        tid = request.thread_id or str(uuid4())
        config = build_agent_run_config(
            agent_id=agent_id,
            thread_id=tid,
            current_user=current_user,
            endpoint="invoke",
        )
        input_data: dict[str, Any] = {"messages": _to_lc_messages(request.messages)}

        # Track thread metadata
        store = _store_var.get(None)
        if store is not None:
            first_msg = request.messages[-1].content if request.messages else None
            await _ensure_thread(store, tid, current_user, agent_id, first_msg)

        try:
            result = await graph.ainvoke(input_data, config=config)
            last = result["messages"][-1]
            content: str = last.content if hasattr(last, "content") else str(last)
            return InvokeResponse(content=content, thread_id=tid)
        finally:
            flush_langfuse()

    # ------------------------------------------------------------------
    # Queue endpoints
    # ------------------------------------------------------------------

    @router.post(
        "/{agent_id}/threads/{thread_id}/queue", response_model=QueueMessageResponse
    )
    async def enqueue_message(
        agent_id: str,
        thread_id: str,
        request: QueueMessageRequest,
        http_request: Request,
        _current_user: CurrentUser,  # type: ignore[valid-type]
    ) -> QueueMessageResponse:
        """Enqueue a message to a thread."""
        _get_graph(agent_id)
        store = _get_store(http_request)

        from langgraph_kit.core.orchestration.queue import (
            QueuedItem,
            QueueSemantic,
            ThreadBusyTracker,
            ThreadQueue,
        )

        tracker = ThreadBusyTracker(store)
        busy = await tracker.is_busy(thread_id)

        queue = ThreadQueue(store, thread_id)
        item = QueuedItem(
            content=request.content,
            semantic=QueueSemantic(request.semantic),
            source=request.source,
            metadata=request.metadata,
        )
        item_id = await queue.enqueue(item)
        depth = await queue.depth()

        return QueueMessageResponse(
            item_id=item_id,
            thread_id=thread_id,
            queued=True,
            thread_busy=busy,
            queue_depth=depth,
        )

    @router.get(
        "/{agent_id}/threads/{thread_id}/queue", response_model=QueueStatusResponse
    )
    async def get_queue_status(
        agent_id: str,
        thread_id: str,
        http_request: Request,
        _current_user: CurrentUser,  # type: ignore[valid-type]
    ) -> QueueStatusResponse:
        """Check queue status for a thread."""
        _get_graph(agent_id)
        store = _get_store(http_request)

        from langgraph_kit.core.orchestration.queue import (
            ThreadBusyTracker,
            ThreadQueue,
        )

        tracker = ThreadBusyTracker(store)
        queue = ThreadQueue(store, thread_id)

        return QueueStatusResponse(
            thread_id=thread_id,
            thread_busy=await tracker.is_busy(thread_id),
            queue_depth=await queue.depth(),
            items=[item.model_dump(mode="json") for item in await queue.peek()],
        )

    # ------------------------------------------------------------------
    # Thread messages
    # ------------------------------------------------------------------

    @router.get("/{agent_id}/threads/{thread_id}/messages")
    async def get_thread_messages(
        agent_id: str,
        thread_id: str,
        _current_user: CurrentUser,  # type: ignore[valid-type]
    ) -> list[dict[str, str]]:
        """Return messages for a thread by reading the latest checkpoint."""
        graph = _get_graph(agent_id)
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

        try:
            state = await graph.aget_state(config)
            if not state or not state.values:
                return []
            messages = state.values.get("messages", [])
            result: list[dict[str, str]] = []
            for msg in messages:
                role = "assistant"
                if hasattr(msg, "type"):
                    if msg.type == "human":
                        role = "user"
                    elif msg.type == "system":
                        role = "system"
                raw_content = msg.content if hasattr(msg, "content") else str(msg)
                content = _extract_text_content(raw_content)
                if content.strip():
                    result.append({"role": role, "content": content})
            return result
        except Exception:
            logger.debug(
                "Could not load messages for thread %s", thread_id, exc_info=True
            )
            return []

    # ------------------------------------------------------------------
    # HITL endpoints
    # ------------------------------------------------------------------

    @router.get(
        "/{agent_id}/threads/{thread_id}/state", response_model=ThreadStateResponse
    )
    async def get_thread_state(
        agent_id: str,
        thread_id: str,
        _current_user: CurrentUser,  # type: ignore[valid-type]
    ) -> ThreadStateResponse:
        """Return thread state including any pending interrupts."""
        graph = _get_graph(agent_id)
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

        try:
            state = await graph.aget_state(config)
        except Exception:
            return ThreadStateResponse(thread_id=thread_id, status="error")

        interrupts: list[dict[str, Any]] = []
        if hasattr(state, "tasks") and state.tasks:
            for task in state.tasks:
                for intr in getattr(task, "interrupts", []):
                    value = intr.value if hasattr(intr, "value") else intr
                    if isinstance(value, dict):
                        interrupts.append(dict(value))  # pyright: ignore[reportUnknownArgumentType]
                    elif isinstance(value, list):
                        interrupts.extend(  # pyright: ignore[reportUnknownArgumentType]
                            dict(v)
                            for v in value
                            if isinstance(v, dict)  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]
                        )

        thread_status: str = "interrupted" if interrupts else "idle"
        return ThreadStateResponse(
            thread_id=thread_id,
            status=thread_status,
            interrupts=interrupts,  # type: ignore[arg-type]
        )

    @router.post(
        "/{agent_id}/threads/{thread_id}/resume", response_model=InvokeResponse
    )
    async def resume_thread(
        agent_id: str,
        thread_id: str,
        body: ResumeRequest,
        current_user: CurrentUser,  # type: ignore[valid-type]
    ) -> InvokeResponse:
        """Resume an interrupted thread with human responses."""
        from langgraph.types import (
            Command,  # pyright: ignore[reportMissingModuleSource]
        )

        graph = _get_graph(agent_id)
        config = build_agent_run_config(
            agent_id=agent_id,
            thread_id=thread_id,
            current_user=current_user,
            endpoint="resume",
        )

        responses = [r.model_dump(mode="json") for r in body.responses]

        try:
            result = await graph.ainvoke(
                Command(resume=responses),
                config=config,
            )
            last = result["messages"][-1]
            content: str = last.content if hasattr(last, "content") else str(last)
            return InvokeResponse(content=content, thread_id=thread_id)
        finally:
            flush_langfuse()

    @router.post("/{agent_id}/threads/{thread_id}/resume/stream")
    async def resume_thread_stream(
        agent_id: str,
        thread_id: str,
        body: ResumeRequest,
        http_request: Request,
        current_user: CurrentUser,  # type: ignore[valid-type]
    ) -> StreamingResponse:
        """Resume an interrupted thread with streaming response."""
        from langgraph.types import (
            Command,  # pyright: ignore[reportMissingModuleSource]
        )

        graph = _get_graph(agent_id)
        config = build_agent_run_config(
            agent_id=agent_id,
            thread_id=thread_id,
            current_user=current_user,
            endpoint="resume_stream",
        )
        store = _store_var.get(None) or getattr(http_request.app.state, "store", None)

        responses = [r.model_dump(mode="json") for r in body.responses]

        return StreamingResponse(
            stream_agent_events(graph, Command(resume=responses), config, store=store),  # type: ignore[arg-type]
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------
    # Branching endpoints
    # ------------------------------------------------------------------

    @router.get("/{agent_id}/threads/{thread_id}/history")
    async def get_thread_history(
        agent_id: str,
        thread_id: str,
        _current_user: CurrentUser,  # type: ignore[valid-type]
        limit: int = 50,
    ) -> list[CheckpointInfo]:
        """Return checkpoint history for branch tree construction."""
        graph = _get_graph(agent_id)
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

        snapshots: list[CheckpointInfo] = []
        try:
            for snapshot in graph.get_state_history(config, limit=limit):
                cp = snapshot.config.get("configurable", {})
                parent_cfg: dict[str, Any] = (
                    snapshot.parent_config.get("configurable", {})
                    if snapshot.parent_config
                    else {}
                )
                messages = snapshot.values.get("messages", [])
                snapshots.append(
                    CheckpointInfo(
                        checkpoint_id=cp.get("checkpoint_id"),
                        parent_checkpoint_id=parent_cfg.get("checkpoint_id"),
                        created_at=snapshot.created_at,
                        metadata=snapshot.metadata or {},
                        next_nodes=list(snapshot.next),
                        message_count=len(messages),
                    )
                )
        except Exception:
            logger.debug(
                "Could not fetch history for thread %s", thread_id, exc_info=True
            )

        return snapshots

    @router.post(
        "/{agent_id}/threads/{thread_id}/fork",
        response_model=ForkResponse,
    )
    async def fork_thread(
        agent_id: str,
        thread_id: str,
        body: ForkRequest,
        _current_user: CurrentUser,  # type: ignore[valid-type]
    ) -> ForkResponse:
        """Fork a conversation at a specific checkpoint."""
        graph = _get_graph(agent_id)
        config: dict[str, Any] = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": body.checkpoint_id,
            }
        }

        try:
            new_config = await graph.aupdate_state(
                config, values=None, as_node="__copy__"
            )
            new_cp_id = new_config["configurable"].get("checkpoint_id", "")
            return ForkResponse(thread_id=thread_id, new_checkpoint_id=new_cp_id)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Fork failed: {exc}",
            ) from exc

    # ------------------------------------------------------------------
    # Thread management endpoints
    # ------------------------------------------------------------------

    @router.get("/threads", response_model=ThreadListResponse)
    async def list_threads(
        current_user: CurrentUser,  # type: ignore[valid-type]
        http_request: Request,
        agent_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ThreadListResponse:
        """List threads for the current user."""
        store = _get_store(http_request)
        from langgraph_kit.core.threads import ThreadManager

        mgr = ThreadManager(store)
        threads, total = await mgr.list_for_user(
            str(current_user.id), agent_id=agent_id, limit=limit, offset=offset
        )
        return ThreadListResponse(
            threads=[
                ThreadMetadataResponse(**t.model_dump(mode="json")) for t in threads
            ],
            total=total,
            limit=limit,
            offset=offset,
        )

    @router.get("/threads/search", response_model=ThreadListResponse)
    async def search_threads(
        q: str,
        current_user: CurrentUser,  # type: ignore[valid-type]
        http_request: Request,
        limit: int = 20,
    ) -> ThreadListResponse:
        """Search threads by title or content."""
        store = _get_store(http_request)
        from langgraph_kit.core.threads import ThreadManager

        mgr = ThreadManager(store)
        threads = await mgr.search(str(current_user.id), q, limit=limit)
        return ThreadListResponse(
            threads=[
                ThreadMetadataResponse(**t.model_dump(mode="json")) for t in threads
            ],
            total=len(threads),
            limit=limit,
            offset=0,
        )

    @router.get("/threads/{thread_id}", response_model=ThreadMetadataResponse)
    async def get_thread_metadata(
        thread_id: str,
        current_user: CurrentUser,  # type: ignore[valid-type]
        http_request: Request,
    ) -> ThreadMetadataResponse:
        """Get metadata for a single thread."""
        store = _get_store(http_request)
        from langgraph_kit.core.threads import ThreadManager

        mgr = ThreadManager(store)
        meta = await mgr.get(thread_id)
        if meta is None or meta.user_id != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Thread not found",
            )
        return ThreadMetadataResponse(**meta.model_dump(mode="json"))

    @router.patch("/threads/{thread_id}", response_model=ThreadMetadataResponse)
    async def update_thread_metadata(
        thread_id: str,
        body: ThreadUpdateRequest,
        current_user: CurrentUser,  # type: ignore[valid-type]
        http_request: Request,
    ) -> ThreadMetadataResponse:
        """Update thread title or tags."""
        store = _get_store(http_request)
        from langgraph_kit.core.threads import ThreadManager

        mgr = ThreadManager(store)
        existing = await mgr.get(thread_id)
        if existing is None or existing.user_id != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Thread not found",
            )

        updated = await mgr.update(thread_id, title=body.title, tags=body.tags)
        if updated is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update thread",
            )
        return ThreadMetadataResponse(**updated.model_dump(mode="json"))

    @router.delete("/threads/{thread_id}")
    async def delete_thread(
        thread_id: str,
        current_user: CurrentUser,  # type: ignore[valid-type]
        http_request: Request,
    ) -> dict[str, bool]:
        """Delete a thread's metadata."""
        store = _get_store(http_request)
        from langgraph_kit.core.threads import ThreadManager

        mgr = ThreadManager(store)
        existing = await mgr.get(thread_id)
        if existing is None or existing.user_id != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Thread not found",
            )
        deleted = await mgr.delete(thread_id)
        return {"deleted": deleted}

    return router


async def _ensure_thread(
    store: Any,
    thread_id: str,
    current_user: Any,
    agent_id: str,
    first_message: str | None,
) -> None:
    """Create/update thread metadata as a fire-and-forget side effect."""
    try:
        from langgraph_kit.core.threads import ThreadManager

        mgr = ThreadManager(store)
        await mgr.ensure_thread(
            thread_id=thread_id,
            user_id=str(current_user.id),
            agent_id=agent_id,
            first_message=first_message,
        )
    except Exception:
        logger.debug("Failed to ensure thread metadata", exc_info=True)
