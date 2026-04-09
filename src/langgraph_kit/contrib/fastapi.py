"""Optional FastAPI route factory for langgraph-kit.

Usage::

    from langgraph_kit.contrib.fastapi import create_agent_router

    router = create_agent_router(get_current_user=CurrentUser)
    app.include_router(router, prefix="/api/v1")
"""

import json
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

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
)
from langgraph_kit.observability import build_agent_run_config, flush_langfuse
from langgraph_kit.registry import get, get_dispatcher, list_agents
from langgraph_kit.streaming import stream_agent_events

logger = logging.getLogger(__name__)


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
            result.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            result.append(AIMessage(content=m.content))
        elif m.role == "system":
            result.append(SystemMessage(content=m.content))
    return result


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
    """Get the LangGraph Store from app state."""
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
        store = getattr(http_request.app.state, "store", None)  # optional for streaming

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
                content = msg.content if hasattr(msg, "content") else str(msg)
                if isinstance(content, str) and content.strip():
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
        store = getattr(http_request.app.state, "store", None)  # optional for streaming

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

    return router
