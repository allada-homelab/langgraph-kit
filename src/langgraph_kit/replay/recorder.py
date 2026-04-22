"""LangChain callback handler that records LLM and tool interactions."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from pathlib import Path

from langchain_core.callbacks import (  # pyright: ignore[reportMissingModuleSource]
    AsyncCallbackHandler,
)

from langgraph_kit.replay.models import (
    ConversationRecording,
    LLMInteraction,
    ToolInteraction,
)


class ConversationRecorder(AsyncCallbackHandler):
    """Records LLM call/response pairs and tool interactions during a graph run.

    Attach to ``config["callbacks"]`` before invoking a graph::

        recorder = ConversationRecorder("my-agent", "thread-1")
        config["callbacks"] = [recorder]
        await graph.ainvoke(input_data, config=config)
        recorder.save(Path("tests/fixtures/my_recording.json"))
    """

    def __init__(self, agent_id: str = "", thread_id: str = "") -> None:
        super().__init__()
        self._agent_id = agent_id
        self._thread_id = thread_id
        self._sequence = 0
        self._interactions: list[LLMInteraction | ToolInteraction] = []
        self._pending_llm: dict[str, dict[str, Any]] = {}  # run_id -> input data
        self._pending_tool: dict[str, dict[str, Any]] = {}  # run_id -> input data
        self._created_at = datetime.now(UTC).isoformat()

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Capture LLM input messages."""
        flat_messages = []
        for msg_list in messages:
            for msg in msg_list:
                flat_messages.append(_serialize_message(msg))

        self._pending_llm[str(run_id)] = {
            "input_messages": flat_messages,
            "model": serialized.get("kwargs", {}).get("model_name", ""),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Capture LLM output and pair with input."""
        key = str(run_id)
        pending = self._pending_llm.pop(key, None)
        if pending is None:
            return

        output_msg: dict[str, Any] = {}
        token_usage: dict[str, Any] | None = None

        if hasattr(response, "generations") and response.generations:
            gen = response.generations[0][0]
            if hasattr(gen, "message"):
                output_msg = _serialize_message(gen.message)
            elif hasattr(gen, "text"):
                output_msg = {"role": "assistant", "content": gen.text}

            # Extract token usage
            if hasattr(response, "llm_output") and response.llm_output:
                usage = response.llm_output.get(
                    "token_usage"
                ) or response.llm_output.get("usage")
                if usage:
                    token_usage = (
                        dict(usage) if hasattr(usage, "__iter__") else {"total": usage}
                    )

        self._sequence += 1
        self._interactions.append(
            LLMInteraction(
                sequence_num=self._sequence,
                model=pending.get("model", ""),
                input_messages=pending["input_messages"],
                output_message=output_msg,
                token_usage=token_usage,
                timestamp=pending.get("timestamp", ""),
            )
        )

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        """Capture tool invocation start."""
        tool_input: dict[str, Any] = {}
        if isinstance(input_str, str):
            try:
                tool_input = json.loads(input_str)
            except (json.JSONDecodeError, TypeError):
                tool_input = {"raw": input_str}
        elif isinstance(input_str, dict):
            tool_input = input_str  # type: ignore[assignment]

        self._pending_tool[str(run_id)] = {
            "tool_name": serialized.get("name", kwargs.get("name", "unknown")),
            "tool_input": tool_input,
            "start_time": time.monotonic(),
        }

    async def on_tool_end(
        self,
        output: str,
        *,
        run_id: Any,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Capture tool output and pair with input."""
        key = str(run_id)
        pending = self._pending_tool.pop(key, None)
        if pending is None:
            return

        duration_ms = (time.monotonic() - pending["start_time"]) * 1000

        output_str = str(output) if not isinstance(output, str) else output
        # Detect error status from ToolErrorMiddleware's structured format
        # or from ToolMessage objects with status="error"
        status: str = "success"
        if (hasattr(output, "status") and output.status == "error") or (
            output_str.startswith("Tool '") and "' failed." in output_str
        ):
            status = "error"

        self._sequence += 1
        self._interactions.append(
            ToolInteraction(
                sequence_num=self._sequence,
                tool_name=pending["tool_name"],
                tool_input=pending["tool_input"],
                tool_output=output_str,
                status=status,  # type: ignore[arg-type]
                duration_ms=round(duration_ms, 2),
            )
        )

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Record tool errors as interactions with error status."""
        key = str(run_id)
        pending = self._pending_tool.pop(key, None)
        if pending is None:
            return

        duration_ms = (time.monotonic() - pending["start_time"]) * 1000

        self._sequence += 1
        self._interactions.append(
            ToolInteraction(
                sequence_num=self._sequence,
                tool_name=pending["tool_name"],
                tool_input=pending["tool_input"],
                tool_output=f"{type(error).__name__}: {error}",
                status="error",
                duration_ms=round(duration_ms, 2),
            )
        )

    def get_recording(self) -> ConversationRecording:
        """Return the complete recording."""
        return ConversationRecording(
            id=str(uuid4()),
            agent_id=self._agent_id,
            thread_id=self._thread_id,
            created_at=self._created_at,
            interactions=list(self._interactions),
        )

    def save(self, path: Path) -> None:
        """Write the recording to a JSON file."""
        recording = self.get_recording()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(recording.model_dump_json(indent=2))

    @staticmethod
    def load(path: Path) -> ConversationRecording:
        """Read a recording from a JSON file."""
        return ConversationRecording.model_validate_json(path.read_text())


def _serialize_message(msg: Any) -> dict[str, Any]:
    """Serialize a LangChain message to a dict."""
    result: dict[str, Any] = {}

    if hasattr(msg, "type"):
        type_to_role = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
        }
        result["role"] = type_to_role.get(msg.type, msg.type)
    elif hasattr(msg, "role"):
        result["role"] = msg.role

    if hasattr(msg, "content"):
        result["content"] = msg.content

    if hasattr(msg, "tool_calls") and msg.tool_calls:
        result["tool_calls"] = [
            {
                "name": tc.get("name", ""),
                "args": tc.get("args", {}),
                "id": tc.get("id", ""),
            }
            for tc in msg.tool_calls
        ]

    if hasattr(msg, "tool_call_id") and msg.tool_call_id:
        result["tool_call_id"] = msg.tool_call_id

    return result
