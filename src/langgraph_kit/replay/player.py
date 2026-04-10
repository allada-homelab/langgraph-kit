"""Replay LLM mock that serves recorded responses for deterministic testing."""

from __future__ import annotations

from typing import Any, ClassVar

from langchain_core.language_models import (  # pyright: ignore[reportMissingModuleSource]
    BaseChatModel,
)
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    AIMessage,
    BaseMessage,
)
from langchain_core.outputs import (  # pyright: ignore[reportMissingModuleSource]
    ChatGeneration,
    ChatResult,
)

from langgraph_kit.replay.models import ConversationRecording, LLMInteraction


class ReplayMismatchError(Exception):
    """Raised when the replay LLM receives input that doesn't match any recording."""


class RecordedChatModel(BaseChatModel):
    """A LangChain chat model that serves recorded responses.

    Matches incoming messages against recorded LLM interactions.
    Uses sequential matching by default, with fuzzy content matching as fallback.

    Usage::

        recording = ConversationRecorder.load(Path("fixture.json"))
        llm = RecordedChatModel(recording=recording)
        # Use llm in place of a real LLM when building the graph
    """

    recording: ConversationRecording
    _call_index: int = 0

    model_config: ClassVar[dict[str, Any]] = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "recorded-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: Any = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        """Serve the next recorded response."""
        llm_interactions = self.recording.llm_interactions

        # Try sequential match first
        if self._call_index < len(llm_interactions):
            interaction = llm_interactions[self._call_index]
            self._call_index += 1
            return _interaction_to_result(interaction)

        # Fallback: fuzzy match on last message content
        last_content = _extract_content(messages[-1]) if messages else ""
        for interaction in llm_interactions:
            if interaction.input_messages and _fuzzy_match(
                last_content, interaction.input_messages
            ):
                return _interaction_to_result(interaction)

        msg = (
            f"No recorded response for call #{self._call_index + 1}. "
            f"Recording has {len(llm_interactions)} LLM interactions. "
            f"Last input: {last_content[:100]!r}"
        )
        raise ReplayMismatchError(msg)

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording.id,
            "agent_id": self.recording.agent_id,
            "total_interactions": len(self.recording.llm_interactions),
        }


def _interaction_to_result(interaction: LLMInteraction) -> ChatResult:
    """Convert a recorded LLM interaction to a ChatResult."""
    output = interaction.output_message
    content = output.get("content", "")
    tool_calls = output.get("tool_calls", [])

    message = AIMessage(
        content=content,
        tool_calls=tool_calls if tool_calls else [],
    )

    return ChatResult(
        generations=[ChatGeneration(message=message)],
        llm_output=interaction.token_usage or {},
    )


def _extract_content(msg: BaseMessage) -> str:
    """Extract text content from a LangChain message."""
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in msg.content]
        return " ".join(parts)
    return str(msg.content)


def _fuzzy_match(content: str, recorded_messages: list[dict[str, Any]]) -> bool:
    """Check if content roughly matches any recorded input message."""
    if not content:
        return False
    for msg in recorded_messages:
        recorded_content = msg.get("content", "")
        if isinstance(recorded_content, str) and recorded_content:
            # Check if significant overlap exists (>50% of shorter string)
            shorter = min(len(content), len(recorded_content))
            if shorter == 0:
                continue
            common = _common_prefix_len(content.lower(), recorded_content.lower())
            if common > shorter * 0.5:
                return True
    return False


def _common_prefix_len(a: str, b: str) -> int:
    """Return the length of the common prefix between two strings."""
    length = min(len(a), len(b))
    for i in range(length):
        if a[i] != b[i]:
            return i
    return length
