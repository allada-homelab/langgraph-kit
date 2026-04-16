"""LangChain callback handler that collects execution trace spans."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langchain_core.callbacks import (  # pyright: ignore[reportMissingModuleSource]
    AsyncCallbackHandler,
)

from langgraph_kit.core.tracing.models import TraceRecord, TraceSpan


class TraceCallbackHandler(AsyncCallbackHandler):
    """Collects execution spans during a graph run for trace export.

    Attach to ``config["callbacks"]``::

        handler = TraceCallbackHandler(agent_id="my-agent", thread_id="t1")
        config["callbacks"] = [handler]
        await graph.ainvoke(input_data, config=config)
        trace = handler.get_trace()
    """

    def __init__(self, agent_id: str = "", thread_id: str = "") -> None:
        super().__init__()
        self._agent_id = agent_id
        self._thread_id = thread_id
        self._trace_id = str(uuid4())
        self._started_at = datetime.now(UTC).isoformat()
        self._start_mono = time.monotonic()
        self._open_spans: dict[str, TraceSpan] = {}  # run_id -> span
        self._span_start_times: dict[str, float] = {}  # run_id -> monotonic time
        self._parent_map: dict[str, str] = {}  # run_id -> parent_run_id
        self._root_spans: list[TraceSpan] = []

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],  # noqa: ARG002
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Record a chain/node span start."""
        self._start_span(
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            kind="chain",
            name=serialized.get("name", serialized.get("id", ["unknown"])[-1]),
        )

    async def on_chain_end(
        self,
        outputs: dict[str, Any],  # noqa: ARG002
        *,
        run_id: Any,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Close a chain/node span."""
        self._end_span(str(run_id))

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Close a chain/node span with error."""
        self._end_span(str(run_id), error=str(error))

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,  # noqa: ARG002
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Record a tool span start."""
        self._start_span(
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            kind="tool",
            name=serialized.get("name", kwargs.get("name", "unknown")),
        )

    async def on_tool_end(
        self,
        output: str,  # noqa: ARG002
        *,
        run_id: Any,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Close a tool span."""
        self._end_span(str(run_id))

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Close a tool span with error."""
        self._end_span(str(run_id), error=str(error))

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],  # noqa: ARG002
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Record an LLM span start."""
        self._start_span(
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            kind="llm",
            name=serialized.get("kwargs", {}).get("model_name", "llm"),
        )

    async def on_llm_end(
        self,
        response: Any,  # noqa: ARG002
        *,
        run_id: Any,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Close an LLM span."""
        self._end_span(str(run_id))

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Close an LLM span with error."""
        self._end_span(str(run_id), error=str(error))

    def get_trace(self) -> TraceRecord:
        """Finalize and return the collected trace."""
        now = datetime.now(UTC).isoformat()
        duration = (time.monotonic() - self._start_mono) * 1000

        # Close any unclosed spans with their actual start time
        for span in self._open_spans.values():
            if span.ended_at is None:
                span.ended_at = now
                start_time = self._span_start_times.get(span.span_id)
                if start_time is not None:
                    span.duration_ms = round((time.monotonic() - start_time) * 1000, 2)
                else:
                    span.duration_ms = duration

        return TraceRecord(
            trace_id=self._trace_id,
            agent_id=self._agent_id,
            thread_id=self._thread_id,
            started_at=self._started_at,
            ended_at=now,
            duration_ms=round(duration, 2),
            spans=self._root_spans,
        )

    def _start_span(
        self,
        run_id: str,
        parent_run_id: str | None,
        kind: str,
        name: str,
    ) -> None:
        span = TraceSpan(
            span_id=run_id,
            parent_span_id=parent_run_id,
            kind=kind,  # type: ignore[arg-type]
            name=name,
            started_at=datetime.now(UTC).isoformat(),
        )
        self._open_spans[run_id] = span
        self._span_start_times[run_id] = time.monotonic()
        if parent_run_id:
            self._parent_map[run_id] = parent_run_id

        # Attach to parent or root
        if parent_run_id and parent_run_id in self._open_spans:
            self._open_spans[parent_run_id].children.append(span)
        else:
            self._root_spans.append(span)

    def _end_span(self, run_id: str, *, error: str | None = None) -> None:
        span = self._open_spans.get(run_id)
        if span is None:
            return
        span.ended_at = datetime.now(UTC).isoformat()
        start_time = self._span_start_times.get(run_id)
        if start_time is not None:
            span.duration_ms = round((time.monotonic() - start_time) * 1000, 2)
        if error:
            span.metadata["error"] = error
        self._open_spans.pop(run_id, None)
