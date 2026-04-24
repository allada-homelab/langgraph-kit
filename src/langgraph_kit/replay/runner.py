"""Replay test runner — replays recorded conversations with mocked LLMs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph_kit.replay.assertions import ReplayAssertions
from langgraph_kit.replay.models import ConversationRecording
from langgraph_kit.replay.player import RecordedChatModel
from langgraph_kit.replay.recorder import ConversationRecorder

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class ReplayRunner:
    """Replays a recorded conversation using a mocked LLM for deterministic testing.

    Usage::

        runner = ReplayRunner(
            Path("tests/fixtures/search_flow.json"),
            graph_builder=build_my_agent,
        )
        assertions = await runner.run_and_assert()
        assertions.assert_same_tool_sequence()
        assertions.assert_tool_called("web_search", times=2)
    """

    def __init__(
        self,
        recording_path: Path,
        graph_builder: Callable[..., Any],
        *,
        tool_overrides: dict[str, Callable[..., Any]] | None = None,
        checkpointer: Any = None,
        store: Any = None,
        llm_kwarg: str = "llm",
        fuzzy_match: bool = True,
    ) -> None:
        super().__init__()
        self.recording_path = recording_path
        self.graph_builder = graph_builder
        self.tool_overrides = tool_overrides or {}
        self.checkpointer = checkpointer
        self.store = store
        # Name of the graph_builder kwarg that accepts the mock LLM. Most
        # builders take ``llm=`` but some upstream APIs use ``model=`` —
        # configurable so callers don't have to wrap their builder just to
        # rename a keyword.
        self.llm_kwarg = llm_kwarg
        # Passed through to RecordedChatModel. Set False in CI runs to
        # fail loudly on prompt drift instead of re-serving stale
        # interactions via the fuzzy-content fallback.
        self.fuzzy_match = fuzzy_match
        self._original: ConversationRecording | None = None

    @property
    def original(self) -> ConversationRecording:
        """Lazily load the original recording."""
        if self._original is None:
            self._original = ConversationRecording.model_validate_json(
                self.recording_path.read_text()
            )
        return self._original

    async def run(self) -> ConversationRecording:
        """Replay the conversation and return a new recording of the replay.

        Builds the graph with a ``RecordedChatModel`` that serves recorded
        responses, then feeds each user message through the graph.
        """
        from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
            HumanMessage,
        )

        recording = self.original
        mock_llm = RecordedChatModel(
            recording=recording, fuzzy_match=self.fuzzy_match
        )

        # Build graph with the mock LLM. ``llm_kwarg`` defaults to "llm"
        # but can be set to e.g. "model" for builders that use that name.
        graph = self.graph_builder(
            self.checkpointer,
            self.store,
            **{self.llm_kwarg: mock_llm},
        )

        # Set up recorder to capture the replay
        replay_recorder = ConversationRecorder(
            agent_id=recording.agent_id,
            thread_id=f"replay-{recording.thread_id}",
        )

        config: dict[str, Any] = {
            "configurable": {"thread_id": f"replay-{recording.thread_id}"},
            "callbacks": [replay_recorder],
        }

        # Extract user messages from the original recording's LLM interactions
        user_messages = _extract_user_turns(recording)

        for user_content in user_messages:
            input_data = {"messages": [HumanMessage(content=user_content)]}
            await graph.ainvoke(input_data, config=config)

        return replay_recorder.get_recording()

    async def run_and_assert(
        self,
        *,
        check_tool_args: bool = True,
    ) -> ReplayAssertions:
        """Run the replay and return an assertions object.

        Convenience method that combines ``run()`` with ``ReplayAssertions``.
        """
        replayed = await self.run()
        assertions = ReplayAssertions(self.original, replayed)
        if check_tool_args:
            assertions.assert_same_tool_calls()
        else:
            assertions.assert_same_tool_sequence()
        return assertions


def _extract_user_turns(recording: ConversationRecording) -> list[str]:
    """Extract user message contents from a recording's LLM interactions.

    Looks for 'user' role messages in the input of each LLM interaction
    and deduplicates consecutive user messages.
    """
    seen_contents: set[str] = set()
    user_messages: list[str] = []

    # First check explicit user_messages field
    if recording.user_messages:
        for msg in recording.user_messages:
            content = msg.get("content", "")
            if isinstance(content, str) and content and content not in seen_contents:
                seen_contents.add(content)
                user_messages.append(content)
        return user_messages

    # Fall back to extracting from LLM interaction inputs
    for interaction in recording.llm_interactions:
        for msg in interaction.input_messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if (
                    isinstance(content, str)
                    and content
                    and content not in seen_contents
                ):
                    seen_contents.add(content)
                    user_messages.append(content)

    return user_messages
