"""AG-UI protocol adapter for langgraph-kit streaming.

Provides two adapters that emit AG-UI-compatible Server-Sent Events:

1. ``stream_agui_events`` — wraps the existing ``stream_agent_events()`` SSE output
2. ``stream_agui_native`` — consumes LangGraph's native ``astream()`` directly

Both use the shared ``AGUIEncoder`` for state tracking and event serialization.

Usage::

    from langgraph_kit.contrib.agui import create_agui_router
    app.include_router(create_agui_router(get_current_user=CurrentUser))
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class AGUIEncoder:
    """Shared state tracker and event encoder for AG-UI protocol.

    Manages message bracketing (start/content/end), step counting,
    and serialization via ``ag_ui.encoder.EventEncoder``.
    """

    def __init__(self, *, thread_id: str = "", run_id: str = "") -> None:
        self.thread_id = thread_id
        self.run_id = run_id or str(uuid4())
        self._text_started = False
        self._message_id = str(uuid4())
        self._step_counter = 0
        self._encoder: Any = None

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            from ag_ui.encoder import (
                EventEncoder,  # pyright: ignore[reportMissingModuleSource]
            )

            self._encoder = EventEncoder()
        return self._encoder

    def encode_run_started(self) -> str:
        """Emit RUN_STARTED event."""
        return self._encode_event("RUN_STARTED", {
            "thread_id": self.thread_id,
            "run_id": self.run_id,
        })

    def encode_run_finished(self) -> str:
        """Emit RUN_FINISHED event."""
        return self._encode_event("RUN_FINISHED", {
            "thread_id": self.thread_id,
            "run_id": self.run_id,
        })

    def encode_run_error(self, message: str) -> str:
        """Emit RUN_ERROR event."""
        return self._encode_event("RUN_ERROR", {"message": message})

    def encode_text_token(self, token: str) -> list[str]:
        """Encode a text token, emitting TEXT_MESSAGE_START if needed."""
        events: list[str] = []
        if not self._text_started:
            self._text_started = True
            events.append(self._encode_event("TEXT_MESSAGE_START", {
                "message_id": self._message_id,
                "role": "assistant",
            }))
        events.append(self._encode_event("TEXT_MESSAGE_CONTENT", {
            "message_id": self._message_id,
            "delta": token,
        }))
        return events

    def encode_text_end(self) -> str | None:
        """Emit TEXT_MESSAGE_END if text was started."""
        if self._text_started:
            self._text_started = False
            return self._encode_event("TEXT_MESSAGE_END", {
                "message_id": self._message_id,
            })
        return None

    def encode_tool_call_start(self, tool_id: str, name: str) -> list[str]:
        """Emit STEP_STARTED + TOOL_CALL_START."""
        self._step_counter += 1
        return [
            self._encode_event("STEP_STARTED", {
                "step_name": f"step_{self._step_counter}",
            }),
            self._encode_event("TOOL_CALL_START", {
                "tool_call_id": tool_id,
                "tool_call_name": name,
            }),
        ]

    def encode_tool_call_end(self, tool_id: str, output: str) -> list[str]:
        """Emit TOOL_CALL_END + TOOL_CALL_RESULT + STEP_FINISHED."""
        return [
            self._encode_event("TOOL_CALL_END", {
                "tool_call_id": tool_id,
            }),
            self._encode_event("TOOL_CALL_RESULT", {
                "message_id": str(uuid4()),
                "tool_call_id": tool_id,
                "content": output,
            }),
            self._encode_event("STEP_FINISHED", {
                "step_name": f"step_{self._step_counter}",
            }),
        ]

    def encode_custom(self, name: str, value: Any) -> str:
        """Emit a CUSTOM event."""
        return self._encode_event("CUSTOM", {"name": name, "value": value})

    def _encode_event(self, event_type: str, data: dict[str, Any]) -> str:
        """Encode a single AG-UI event as an SSE line."""
        try:
            encoder = self._get_encoder()
            from ag_ui.core import (  # pyright: ignore[reportMissingModuleSource]
                BaseEvent,
                EventType,
            )

            event = BaseEvent(type=EventType(event_type), **data)
            return encoder.encode(event)
        except Exception:
            # Fallback: manual SSE encoding if ag-ui types don't match
            payload = {"type": event_type, **data}
            return f"data: {json.dumps(payload)}\n\n"


async def stream_agui_events(
    graph: Any,
    input_data: dict[str, Any],
    config: dict[str, Any],
    *,
    store: Any | None = None,
    run_id: str = "",
) -> AsyncGenerator[str]:
    """Wrap ``stream_agent_events()`` output and re-emit as AG-UI events.

    This is the primary adapter — works with both astream_events v1 and v2
    since our SSE layer normalizes both formats.
    """
    from langgraph_kit.streaming import stream_agent_events

    thread_id = config.get("configurable", {}).get("thread_id", "")
    encoder = AGUIEncoder(thread_id=thread_id, run_id=run_id)

    yield encoder.encode_run_started()

    try:
        async for sse_line in stream_agent_events(graph, input_data, config, store=store):
            # Parse our SSE format: "data: {...}\n\n"
            if not sse_line.startswith("data: "):
                continue
            raw = sse_line.strip().removeprefix("data: ")
            if raw == "[DONE]":
                break

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue

            for agui_event in _map_sse_to_agui(parsed, encoder):
                yield agui_event

    except Exception as exc:
        yield encoder.encode_run_error(str(exc))

    # Close text message if open
    end_event = encoder.encode_text_end()
    if end_event:
        yield end_event

    yield encoder.encode_run_finished()


async def stream_agui_native(
    graph: Any,
    input_data: dict[str, Any],
    config: dict[str, Any],
    *,
    stream_mode: list[str] | None = None,
    run_id: str = "",
) -> AsyncGenerator[str]:
    """Consume LangGraph's native ``astream()`` and emit AG-UI events.

    This adapter gives richer data (node names, sub-graph namespaces)
    by reading native stream modes directly.
    """
    if stream_mode is None:
        stream_mode = ["messages", "updates"]

    thread_id = config.get("configurable", {}).get("thread_id", "")
    encoder = AGUIEncoder(thread_id=thread_id, run_id=run_id)

    yield encoder.encode_run_started()

    try:
        async for chunk in graph.astream(input_data, config=config, stream_mode=stream_mode):
            if isinstance(chunk, tuple) and len(chunk) == 2:
                mode, data = chunk
            else:
                continue

            for agui_event in _map_native_to_agui(mode, data, encoder):
                yield agui_event

    except Exception as exc:
        yield encoder.encode_run_error(str(exc))

    end_event = encoder.encode_text_end()
    if end_event:
        yield end_event

    yield encoder.encode_run_finished()


def _map_sse_to_agui(parsed: dict[str, Any], encoder: AGUIEncoder) -> list[str]:
    """Map a parsed SSE event dict to AG-UI events."""
    events: list[str] = []

    if "token" in parsed:
        events.extend(encoder.encode_text_token(parsed["token"]))

    elif "tool_call_start" in parsed:
        tc = parsed["tool_call_start"]
        events.extend(encoder.encode_tool_call_start(tc.get("id", ""), tc.get("name", "")))

    elif "tool_call_end" in parsed:
        tc = parsed["tool_call_end"]
        events.extend(encoder.encode_tool_call_end(tc.get("id", ""), tc.get("output", "")))

    elif "command_result" in parsed:
        cr = parsed["command_result"]
        events.extend(encoder.encode_text_token(cr.get("output", "")))

    elif "artifact" in parsed:
        events.append(encoder.encode_custom("artifact", parsed["artifact"]))

    elif "progress" in parsed:
        events.append(encoder.encode_custom("progress", parsed["progress"]))

    elif "suggestions" in parsed:
        events.append(encoder.encode_custom("suggestions", parsed["suggestions"]))

    elif "citation" in parsed:
        events.append(encoder.encode_custom("citation", parsed["citation"]))

    elif "interrupt" in parsed:
        events.append(encoder.encode_custom("interrupt", parsed["interrupt"]))

    elif "budget" in parsed:
        events.append(encoder.encode_custom("budget", parsed["budget"]))

    elif "trace" in parsed:
        events.append(encoder.encode_custom("trace", parsed["trace"]))

    return events


def _map_native_to_agui(mode: str, data: Any, encoder: AGUIEncoder) -> list[str]:
    """Map a native LangGraph stream chunk to AG-UI events."""
    events: list[str] = []

    if mode == "messages":
        # data is (message_chunk, metadata_dict)
        if isinstance(data, tuple) and len(data) == 2:
            msg, _meta = data
            content = getattr(msg, "content", "")
            msg_type = getattr(msg, "type", "")

            if msg_type == "AIMessageChunk" or msg_type == "ai":
                if isinstance(content, str) and content:
                    events.extend(encoder.encode_text_token(content))
            elif msg_type == "tool":
                tool_call_id = getattr(msg, "tool_call_id", "")
                if tool_call_id and isinstance(content, str):
                    events.extend(encoder.encode_tool_call_end(tool_call_id, content))

    elif mode == "updates":
        # data is {node_name: output_dict}
        if isinstance(data, dict):
            for node_name in data:
                encoder._step_counter += 1
                events.append(encoder._encode_event("STEP_STARTED", {
                    "step_name": node_name,
                }))
                events.append(encoder._encode_event("STEP_FINISHED", {
                    "step_name": node_name,
                }))

    elif mode == "custom":
        events.append(encoder.encode_custom("custom", data))

    return events


def create_agui_router(*, get_current_user: Any) -> Any:
    """Create a FastAPI router for AG-UI protocol endpoints.

    Adds ``POST /agents/{agent_id}/agui/stream`` alongside the existing endpoints.
    """
    from fastapi import APIRouter, Request
    from fastapi.responses import StreamingResponse

    from langgraph_kit.contrib.fastapi import _get_store, _to_lc_messages
    from langgraph_kit.models import InvokeRequest
    from langgraph_kit.observability import build_agent_run_config
    from langgraph_kit.registry import get

    router = APIRouter(prefix="/agents", tags=["agents-agui"])
    CurrentUser = get_current_user

    @router.post("/{agent_id}/agui/stream")
    async def agui_stream(
        agent_id: str,
        request: InvokeRequest,
        http_request: Request,
        current_user: CurrentUser,  # type: ignore[valid-type]
        native: bool = False,
    ) -> StreamingResponse:
        """Stream AG-UI protocol events."""
        try:
            graph = get(agent_id)
        except KeyError:
            from fastapi import HTTPException, status

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{agent_id}' not found",
            ) from None

        tid = request.thread_id or str(uuid4())
        config = build_agent_run_config(
            agent_id=agent_id,
            thread_id=tid,
            current_user=current_user,
            endpoint="agui_stream",
        )
        input_data: dict[str, Any] = {"messages": _to_lc_messages(request.messages)}
        run_id = str(uuid4())

        if native:
            generator = stream_agui_native(graph, input_data, config, run_id=run_id)
        else:
            store = _get_store(http_request)
            generator = stream_agui_events(
                graph, input_data, config, store=store, run_id=run_id
            )

        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
