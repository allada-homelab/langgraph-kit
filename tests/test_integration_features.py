# pyright: reportPrivateUsage=false
# Strict-mode prep: these tests reach into private helpers (``_metadata``,
# ``_registry``, ``_map_sse_to_agui``, ``_extract_usage``,
# ``_build_multipart_content``, ``_extract_text_content``) by design so
# the private-side contract is covered too.  Keeps the file clean under
# a future strict flip without weakening type safety elsewhere.
"""Integration tests for all v0.5.0 platform features.

Each test class covers one feature with full setup → run → analyze.
All tests are re-runnable, use in-memory mocks, and require no external services.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import MockStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_mock(
    content: str = "Hello!", tool_calls: list[dict[str, Any]] | None = None
) -> AsyncMock:
    """Create an AsyncMock LLM whose ainvoke returns an AIMessage."""
    mock = AsyncMock()
    msg = AIMessage(content=content, tool_calls=tool_calls or [])
    mock.ainvoke.return_value = msg
    mock.bind_tools = MagicMock(return_value=mock)
    return mock


def _make_llm_with_usage(
    content: str = "Done", input_tokens: int = 100, output_tokens: int = 50
) -> AsyncMock:
    """Create an LLM mock that returns token usage data."""
    mock = AsyncMock()
    msg = AIMessage(content=content)
    mock.ainvoke.return_value = msg

    # Simulate LangChain LLMResult for callback
    generation = MagicMock()
    generation.message = msg
    generation.generation_info = {
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "model": "gpt-4o-mini",
    }
    mock._generation_result = MagicMock()
    mock._generation_result.generations = [[generation]]
    mock._generation_result.llm_output = {
        "token_usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
        },
        "model_name": "gpt-4o-mini",
    }
    return mock


# ===========================================================================
# Feature 0: Enriched Agent Registry
# ===========================================================================


class TestEnrichedRegistry:
    """Test that the agent registry supports rich metadata."""

    def test_register_with_metadata(self) -> None:
        """Setup: register an agent with full metadata. Verify all fields persist."""
        from langgraph_kit.registry import (
            AgentMetadata,
            _metadata,
            _registry,
            get_metadata,
            register,
        )

        # Clean state
        _registry.clear()
        _metadata.clear()

        register(
            "test-agent",
            MagicMock(),
            metadata=AgentMetadata(
                description="A test agent",
                version="2.0.0",
                tags=["test", "demo"],
                capabilities=["streaming", "hitl"],
                input_modes=["text/plain", "image/png"],
                output_modes=["text/plain"],
            ),
        )

        meta = get_metadata("test-agent")
        assert meta.description == "A test agent"
        assert meta.version == "2.0.0"
        assert "test" in meta.tags
        assert "streaming" in meta.capabilities
        assert "image/png" in meta.input_modes

    def test_register_without_metadata_uses_defaults(self) -> None:
        from langgraph_kit.registry import (
            _metadata,
            _registry,
            get_metadata,
            register,
        )

        _registry.clear()
        _metadata.clear()

        register("bare-agent", MagicMock())
        meta = get_metadata("bare-agent")
        assert meta.description == ""
        assert meta.version == "1.0.0"
        assert meta.tags == []

    def test_list_agents_includes_metadata_fields(self) -> None:
        from langgraph_kit.registry import (
            AgentMetadata,
            _metadata,
            _registry,
            list_agents,
            register,
        )

        _registry.clear()
        _metadata.clear()

        register(
            "rich-agent",
            MagicMock(),
            metadata=AgentMetadata(description="Rich", tags=["ai"]),
        )
        agents = list_agents()
        assert len(agents) == 1
        assert agents[0]["description"] == "Rich"
        assert agents[0]["tags"] == ["ai"]

    def test_get_all_returns_graphs(self) -> None:
        from langgraph_kit.registry import _metadata, _registry, get_all, register

        _registry.clear()
        _metadata.clear()

        graph_a = MagicMock()
        graph_b = MagicMock()
        register("a", graph_a)
        register("b", graph_b)

        all_graphs = get_all()
        assert all_graphs["a"] is graph_a
        assert all_graphs["b"] is graph_b


# ===========================================================================
# Feature 1: AG-UI Protocol
# ===========================================================================


class TestAGUIProtocol:
    """Test AG-UI event adapter maps our SSE events to AG-UI protocol."""

    def test_encoder_run_lifecycle(self) -> None:
        """Setup: create encoder. Run: emit start/finish. Analyze: correct event types."""
        from langgraph_kit.contrib.agui import AGUIEncoder

        encoder = AGUIEncoder(thread_id="t1", run_id="r1")
        started = encoder.encode_run_started()
        finished = encoder.encode_run_finished()

        # Should be SSE format
        assert "RUN_STARTED" in started
        assert "RUN_FINISHED" in finished

    def test_encoder_text_bracketing(self) -> None:
        """Setup: create encoder. Run: send tokens. Analyze: proper start/content/end."""
        from langgraph_kit.contrib.agui import AGUIEncoder

        encoder = AGUIEncoder(thread_id="t1")

        # First token should emit both TEXT_MESSAGE_START and TEXT_MESSAGE_CONTENT
        events = encoder.encode_text_token("Hello")
        assert len(events) == 2
        assert "TEXT_MESSAGE_START" in events[0]
        assert "TEXT_MESSAGE_CONTENT" in events[1]
        assert "Hello" in events[1]

        # Second token should only emit TEXT_MESSAGE_CONTENT
        events = encoder.encode_text_token(" world")
        assert len(events) == 1
        assert "TEXT_MESSAGE_CONTENT" in events[0]

        # End should close the message
        end = encoder.encode_text_end()
        assert end is not None
        assert "TEXT_MESSAGE_END" in end

    def test_encoder_tool_call_events(self) -> None:
        """Setup: encoder. Run: tool start/end. Analyze: step + tool events."""
        from langgraph_kit.contrib.agui import AGUIEncoder

        encoder = AGUIEncoder(thread_id="t1")

        start_events = encoder.encode_tool_call_start("tc1", "web_search")
        assert len(start_events) == 2
        assert "STEP_STARTED" in start_events[0]
        assert "TOOL_CALL_START" in start_events[1]
        assert "web_search" in start_events[1]

        end_events = encoder.encode_tool_call_end("tc1", "search results here")
        assert len(end_events) == 3
        assert "TOOL_CALL_END" in end_events[0]
        assert "TOOL_CALL_RESULT" in end_events[1]
        assert "STEP_FINISHED" in end_events[2]

    def test_encoder_custom_events(self) -> None:
        """Setup: encoder. Run: custom event. Analyze: CUSTOM type with payload."""
        from langgraph_kit.contrib.agui import AGUIEncoder

        encoder = AGUIEncoder(thread_id="t1")
        event = encoder.encode_custom(
            "artifact", {"type": "code", "content": "print(1)"}
        )
        assert "CUSTOM" in event
        assert "artifact" in event

    def test_sse_to_agui_mapping(self) -> None:
        """Test that our SSE event dicts map correctly to AG-UI events."""
        from langgraph_kit.contrib.agui import AGUIEncoder, _map_sse_to_agui

        encoder = AGUIEncoder(thread_id="t1")

        # Token
        events = _map_sse_to_agui({"token": "hi"}, encoder)
        assert len(events) == 2  # START + CONTENT on first token

        # Tool call start
        events = _map_sse_to_agui(
            {"tool_call_start": {"id": "t1", "name": "search"}}, encoder
        )
        assert len(events) == 2  # STEP_STARTED + TOOL_CALL_START

        # Artifact
        events = _map_sse_to_agui({"artifact": {"type": "code"}}, encoder)
        assert len(events) == 1
        assert "CUSTOM" in events[0]

        # Interrupt
        events = _map_sse_to_agui({"interrupt": {"action": "confirm"}}, encoder)
        assert len(events) == 1
        assert "interrupt" in events[0]


# ===========================================================================
# Feature 2: Token Budget Manager
# ===========================================================================


class TestTokenBudgetManager:
    """Test per-thread token budget tracking and enforcement."""

    @pytest.mark.asyncio
    async def test_budget_check_allows_within_limit(
        self, mock_store: MockStore
    ) -> None:
        """Setup: budget of 10000 tokens. Run: check with no usage. Analyze: allow."""
        from langgraph_kit.core.cost import BudgetConfig, BudgetManager

        config = BudgetConfig(max_tokens_per_thread=10000)
        mgr = BudgetManager(mock_store, config)

        result = await mgr.check_budget("thread-1")
        assert result.action == "allow"
        assert result.budget_consumed_pct == 0.0
        assert result.remaining_tokens == 10000

    @pytest.mark.asyncio
    async def test_budget_records_and_accumulates(self, mock_store: MockStore) -> None:
        """Setup: budget manager. Run: record multiple usages. Analyze: state accumulates."""
        from langgraph_kit.core.cost import BudgetConfig, BudgetManager, TokenUsage

        config = BudgetConfig(max_tokens_per_thread=10000)
        mgr = BudgetManager(mock_store, config)

        await mgr.record_usage(
            "t1", TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        )
        await mgr.record_usage(
            "t1", TokenUsage(input_tokens=200, output_tokens=100, total_tokens=300)
        )

        state = await mgr.load_state("t1")
        assert state.total_input_tokens == 300
        assert state.total_output_tokens == 150
        assert state.turn_count == 2

    @pytest.mark.asyncio
    async def test_budget_warns_at_threshold(self, mock_store: MockStore) -> None:
        """Setup: 80% warning, 1000 budget. Run: use 850 tokens. Analyze: warn action."""
        from langgraph_kit.core.cost import BudgetConfig, BudgetManager, TokenUsage

        config = BudgetConfig(max_tokens_per_thread=1000, warning_threshold_pct=0.80)
        mgr = BudgetManager(mock_store, config)

        await mgr.record_usage(
            "t1", TokenUsage(input_tokens=500, output_tokens=350, total_tokens=850)
        )
        result = await mgr.check_budget("t1")
        assert result.action == "warn"
        assert result.budget_consumed_pct >= 0.80

    @pytest.mark.asyncio
    async def test_budget_denies_when_exhausted(self, mock_store: MockStore) -> None:
        """Setup: 1000 budget. Run: use 1200 tokens. Analyze: deny action."""
        from langgraph_kit.core.cost import BudgetConfig, BudgetManager, TokenUsage

        config = BudgetConfig(max_tokens_per_thread=1000)
        mgr = BudgetManager(mock_store, config)

        await mgr.record_usage(
            "t1", TokenUsage(input_tokens=700, output_tokens=500, total_tokens=1200)
        )
        result = await mgr.check_budget("t1")
        assert result.action == "deny"
        assert result.remaining_tokens == 0

    @pytest.mark.asyncio
    async def test_budget_downgrade_when_configured(
        self, mock_store: MockStore
    ) -> None:
        """Setup: budget with downgrade model. Run: exceed warning. Analyze: downgrade action."""
        from langgraph_kit.core.cost import BudgetConfig, BudgetManager, TokenUsage

        config = BudgetConfig(
            max_tokens_per_thread=1000,
            warning_threshold_pct=0.80,
            downgrade_model="gpt-4o-mini",
        )
        mgr = BudgetManager(mock_store, config)

        await mgr.record_usage(
            "t1", TokenUsage(input_tokens=500, output_tokens=350, total_tokens=850)
        )
        result = await mgr.check_budget("t1")
        assert result.action == "downgrade"
        assert "gpt-4o-mini" in result.reason

    @pytest.mark.asyncio
    async def test_unlimited_budget_always_allows(self, mock_store: MockStore) -> None:
        """Setup: budget 0 (unlimited). Run: check. Analyze: always allow."""
        from langgraph_kit.core.cost import BudgetConfig, BudgetManager

        config = BudgetConfig(max_tokens_per_thread=0)
        mgr = BudgetManager(mock_store, config)

        result = await mgr.check_budget("any-thread")
        assert result.action == "allow"

    def test_cost_estimation(self) -> None:
        """Test that cost estimation works for known models."""
        from langgraph_kit.core.cost import TokenUsage, estimate_cost

        usage = TokenUsage(input_tokens=1000, output_tokens=500, model="gpt-4o-mini")
        cost = estimate_cost(usage)
        assert cost > 0

    def test_cost_estimation_unknown_model(self) -> None:
        from langgraph_kit.core.cost import TokenUsage, estimate_cost

        usage = TokenUsage(input_tokens=1000, output_tokens=500, model="unknown-model")
        cost = estimate_cost(usage)
        assert cost == 0.0

    def test_token_tracking_callback_extraction(self) -> None:
        """Test that the callback extracts usage from LLM responses."""
        from langgraph_kit.core.cost.callback import _extract_usage

        # Simulate OpenAI-style response
        response = MagicMock()
        response.llm_output = {
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "model_name": "gpt-4o",
        }
        response.generations = []

        usage = _extract_usage(response)
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.model == "gpt-4o"


# ===========================================================================
# Feature 3: Conversation Replay Testing
# ===========================================================================


class TestConversationReplay:
    """Test the record → save → load → replay → assert cycle."""

    def test_recording_model_roundtrip(self) -> None:
        """Setup: create recording. Run: serialize/deserialize. Analyze: data preserved."""
        from langgraph_kit.replay import (
            ConversationRecording,
            LLMInteraction,
            ToolInteraction,
        )

        recording = ConversationRecording(
            id="rec-1",
            agent_id="test-agent",
            thread_id="t1",
            model="gpt-4o",
            interactions=[
                LLMInteraction(
                    sequence_num=1,
                    model="gpt-4o",
                    input_messages=[{"role": "user", "content": "hello"}],
                    output_message={"role": "assistant", "content": "hi there"},
                ),
                ToolInteraction(
                    sequence_num=2,
                    tool_name="search",
                    tool_input={"query": "test"},
                    tool_output="results",
                ),
            ],
        )

        # Roundtrip through JSON
        json_str = recording.model_dump_json()
        loaded = ConversationRecording.model_validate_json(json_str)

        assert loaded.id == "rec-1"
        assert loaded.agent_id == "test-agent"
        assert len(loaded.interactions) == 2
        assert loaded.llm_interactions[0].model == "gpt-4o"
        assert loaded.tool_interactions[0].tool_name == "search"

    def test_recording_properties(self) -> None:
        """Test tool_sequence and interaction filtering properties."""
        from langgraph_kit.replay import (
            ConversationRecording,
            LLMInteraction,
            ToolInteraction,
        )

        recording = ConversationRecording(
            interactions=[
                LLMInteraction(sequence_num=1),
                ToolInteraction(sequence_num=2, tool_name="search"),
                LLMInteraction(sequence_num=3),
                ToolInteraction(sequence_num=4, tool_name="write"),
                ToolInteraction(sequence_num=5, tool_name="search"),
            ],
        )

        assert recording.tool_sequence == ["search", "write", "search"]
        assert len(recording.llm_interactions) == 2
        assert len(recording.tool_interactions) == 3

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Setup: create recording. Run: save to file and reload. Analyze: content matches."""
        from langgraph_kit.replay import ConversationRecording, LLMInteraction
        from langgraph_kit.replay.recorder import ConversationRecorder

        recording = ConversationRecording(
            id="save-test",
            agent_id="my-agent",
            interactions=[
                LLMInteraction(
                    sequence_num=1,
                    input_messages=[{"role": "user", "content": "hi"}],
                    output_message={"role": "assistant", "content": "hello"},
                ),
            ],
        )

        file_path = tmp_path / "test_recording.json"
        file_path.write_text(recording.model_dump_json(indent=2))

        loaded = ConversationRecorder.load(file_path)
        assert loaded.id == "save-test"
        assert loaded.agent_id == "my-agent"

    def test_recorded_chat_model_sequential(self) -> None:
        """Setup: recording with 2 LLM interactions. Run: call model twice. Analyze: correct responses."""
        from langgraph_kit.replay import (
            ConversationRecording,
            LLMInteraction,
            RecordedChatModel,
        )

        recording = ConversationRecording(
            interactions=[
                LLMInteraction(
                    sequence_num=1,
                    output_message={"role": "assistant", "content": "First response"},
                ),
                LLMInteraction(
                    sequence_num=2,
                    output_message={"role": "assistant", "content": "Second response"},
                ),
            ],
        )

        model = RecordedChatModel(recording=recording)

        result1 = model.invoke([HumanMessage(content="q1")])
        assert result1.content == "First response"

        result2 = model.invoke([HumanMessage(content="q2")])
        assert result2.content == "Second response"

    def test_recorded_chat_model_bind_tools_is_noop_returning_self(self) -> None:
        """``bind_tools`` must return a Runnable that still serves scripted responses.

        ``create_agent`` and any LangChain agent flow calls ``bind_tools``
        on its model at construction time. ``BaseChatModel.bind_tools``
        raises ``NotImplementedError`` by default — without an override,
        ``RecordedChatModel`` cannot drive a real compiled graph,
        defeating the whole point of the replay system.

        Tool schemas don't affect a recorded response (the recording
        already has ``tool_calls`` baked into each ``output_message``),
        so the override is a pass-through. This test locks in that
        contract so a future change to ``bind_tools`` doesn't silently
        break the e2e test layer.
        """
        from langgraph_kit.replay import (
            ConversationRecording,
            LLMInteraction,
            RecordedChatModel,
        )

        recording = ConversationRecording(
            interactions=[
                LLMInteraction(
                    sequence_num=1,
                    output_message={"role": "assistant", "content": "bound"},
                ),
            ],
        )
        model = RecordedChatModel(recording=recording)

        def _dummy_tool(x: int) -> int:
            return x

        bound = model.bind_tools([_dummy_tool])
        assert bound is model  # pass-through by design

        result = bound.invoke([HumanMessage(content="ignored")])
        assert result.content == "bound"

    def test_recorded_chat_model_raises_on_exhaustion(self) -> None:
        """Setup: recording with 1 interaction. Run: call twice. Analyze: raises mismatch."""
        from langgraph_kit.replay import (
            ConversationRecording,
            LLMInteraction,
            RecordedChatModel,
            ReplayMismatchError,
        )

        recording = ConversationRecording(
            interactions=[
                LLMInteraction(
                    sequence_num=1,
                    output_message={"role": "assistant", "content": "only one"},
                ),
            ],
        )

        model = RecordedChatModel(recording=recording)
        model.invoke([HumanMessage(content="q1")])

        with pytest.raises(ReplayMismatchError):
            model.invoke([HumanMessage(content="q2")])

    def test_assertions_tool_sequence(self) -> None:
        """Test ReplayAssertions for tool sequence matching."""
        from langgraph_kit.replay import (
            ConversationRecording,
            ReplayAssertions,
            ToolInteraction,
        )

        original = ConversationRecording(
            interactions=[
                ToolInteraction(sequence_num=1, tool_name="search"),
                ToolInteraction(sequence_num=2, tool_name="write"),
            ],
        )
        replayed = ConversationRecording(
            interactions=[
                ToolInteraction(sequence_num=1, tool_name="search"),
                ToolInteraction(sequence_num=2, tool_name="write"),
            ],
        )

        assertions = ReplayAssertions(original, replayed)
        assertions.assert_same_tool_sequence()  # Should not raise

    def test_assertions_tool_sequence_mismatch(self) -> None:
        from langgraph_kit.replay import (
            ConversationRecording,
            ReplayAssertions,
            ToolInteraction,
        )

        original = ConversationRecording(
            interactions=[ToolInteraction(sequence_num=1, tool_name="search")],
        )
        replayed = ConversationRecording(
            interactions=[ToolInteraction(sequence_num=1, tool_name="write")],
        )

        assertions = ReplayAssertions(original, replayed)
        with pytest.raises(AssertionError, match="Tool sequence mismatch"):
            assertions.assert_same_tool_sequence()

    def test_assertions_tool_called(self) -> None:
        from langgraph_kit.replay import (
            ConversationRecording,
            ReplayAssertions,
            ToolInteraction,
        )

        recording = ConversationRecording(
            interactions=[
                ToolInteraction(sequence_num=1, tool_name="search"),
                ToolInteraction(sequence_num=2, tool_name="search"),
                ToolInteraction(sequence_num=3, tool_name="write"),
            ],
        )

        assertions = ReplayAssertions(recording, recording)
        assertions.assert_tool_called("search", times=2)
        assertions.assert_tool_called("write", times=1)
        assertions.assert_tool_not_called("delete")

    def test_assertions_final_output(self) -> None:
        from langgraph_kit.replay import (
            ConversationRecording,
            LLMInteraction,
            ReplayAssertions,
        )

        recording = ConversationRecording(
            interactions=[
                LLMInteraction(
                    sequence_num=1,
                    output_message={"role": "assistant", "content": "The answer is 42"},
                ),
            ],
        )

        assertions = ReplayAssertions(recording, recording)
        assertions.assert_final_output_contains("42")
        assertions.assert_final_output_matches(r"\d+")

    def test_standalone_assert_tool_sequence(self) -> None:
        from langgraph_kit.replay import (
            ConversationRecording,
            ToolInteraction,
            assert_tool_sequence,
        )

        recording = ConversationRecording(
            interactions=[
                ToolInteraction(sequence_num=1, tool_name="a"),
                ToolInteraction(sequence_num=2, tool_name="b"),
            ],
        )
        assert_tool_sequence(recording, ["a", "b"])

        with pytest.raises(AssertionError):
            assert_tool_sequence(recording, ["b", "a"])


# ===========================================================================
# Feature 4: A2A Protocol
# ===========================================================================


class TestA2AProtocol:
    """Test A2A Agent Card generation and task invocation."""

    def test_build_agent_card(self) -> None:
        """Setup: register agent with metadata. Run: build card. Analyze: valid A2A card."""
        from langgraph_kit.contrib.a2a import build_agent_card
        from langgraph_kit.registry import AgentMetadata, _metadata, _registry, register

        _registry.clear()
        _metadata.clear()

        register(
            "test-a2a",
            MagicMock(),
            metadata=AgentMetadata(
                description="Test agent for A2A",
                tags=["test"],
                capabilities=["streaming"],
            ),
        )

        card = build_agent_card("test-a2a", "https://example.com")

        assert card["name"] == "Test A2A"
        assert card["description"] == "Test agent for A2A"
        assert card["url"] == "https://example.com/a2a/test-a2a"
        assert card["version"] == "1.0.0"
        assert card["capabilities"]["streaming"] is True
        assert len(card["skills"]) == 1
        assert card["skills"][0]["tags"] == ["test"]

    def test_build_aggregated_card(self) -> None:
        """Setup: register multiple agents. Run: build aggregated card. Analyze: all agents as skills."""
        from langgraph_kit.contrib.a2a import build_aggregated_card
        from langgraph_kit.registry import AgentMetadata, _metadata, _registry, register

        _registry.clear()
        _metadata.clear()

        register("agent-a", MagicMock(), metadata=AgentMetadata(description="Agent A"))
        register("agent-b", MagicMock(), metadata=AgentMetadata(description="Agent B"))

        card = build_aggregated_card("https://example.com")

        assert card["name"] == "LangGraph Kit Agent Hub"
        assert len(card["skills"]) == 2
        skill_ids = {s["id"] for s in card["skills"]}
        assert skill_ids == {"agent-a", "agent-b"}

    @pytest.mark.asyncio
    async def test_invoke_agent_a2a_success(self) -> None:
        """Setup: register mock agent. Run: invoke via A2A. Analyze: completed task."""
        from langgraph_kit.contrib.a2a import invoke_agent_a2a
        from langgraph_kit.registry import _metadata, _registry, register

        _registry.clear()
        _metadata.clear()

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "messages": [AIMessage(content="I can help!")]
        }
        register("mock-agent", mock_graph)

        result = await invoke_agent_a2a("mock-agent", "Hello")

        assert result["status"]["state"] == "completed"
        assert len(result["artifacts"]) == 1
        assert result["artifacts"][0]["parts"][0]["text"] == "I can help!"

    @pytest.mark.asyncio
    async def test_invoke_agent_a2a_failure(self) -> None:
        """Setup: agent that raises. Run: invoke. Analyze: failed task."""
        from langgraph_kit.contrib.a2a import invoke_agent_a2a
        from langgraph_kit.registry import _metadata, _registry, register

        _registry.clear()
        _metadata.clear()

        mock_graph = AsyncMock()
        mock_graph.ainvoke.side_effect = RuntimeError("LLM error")
        register("failing-agent", mock_graph)

        result = await invoke_agent_a2a("failing-agent", "Hello")
        assert result["status"]["state"] == "failed"


# ===========================================================================
# Feature 5: Supervisor / Router Agent
# ===========================================================================


class TestSupervisorRouter:
    """Test routing strategies and supervisor delegation."""

    @pytest.mark.asyncio
    async def test_keyword_routing_matches_tags(self) -> None:
        """Setup: agents with tags. Run: route coding request. Analyze: picks coding agent."""
        from langgraph_kit.core.orchestration.routing import (
            AgentCapability,
            KeywordRoutingStrategy,
        )

        caps = [
            AgentCapability(
                agent_id="general",
                name="General",
                description="General assistant",
                tags=["general", "chat"],
            ),
            AgentCapability(
                agent_id="coding",
                name="Coding",
                description="Code writing and review",
                tags=["coding", "git", "review"],
            ),
        ]

        router = KeywordRoutingStrategy()
        decision = await router.route("Can you review my code?", caps)

        assert decision.target_agent_id == "coding"
        assert (
            "review" in decision.delegated_prompt.lower()
            or "code" in decision.delegated_prompt.lower()
        )

    @pytest.mark.asyncio
    async def test_keyword_routing_fallback(self) -> None:
        """Setup: agents with obscure tags. Run: unrelated request. Analyze: falls back to first."""
        from langgraph_kit.core.orchestration.routing import (
            AgentCapability,
            KeywordRoutingStrategy,
        )

        caps = [
            AgentCapability(
                agent_id="agent-a",
                name="A",
                description="Handles alphas",
                tags=["alpha"],
            ),
            AgentCapability(
                agent_id="agent-b", name="B", description="Handles betas", tags=["beta"]
            ),
        ]

        router = KeywordRoutingStrategy()
        decision = await router.route("Tell me a joke", caps)

        # Should fall back to first agent (no keyword matches)
        assert decision.target_agent_id == "agent-a"
        assert (
            "default" in decision.reasoning.lower()
            or "match" in decision.reasoning.lower()
        )

    @pytest.mark.asyncio
    async def test_keyword_routing_no_agents(self) -> None:
        from langgraph_kit.core.orchestration.routing import KeywordRoutingStrategy

        router = KeywordRoutingStrategy()
        decision = await router.route("Hello", [])
        assert decision.target_agent_id == "none"

    @pytest.mark.asyncio
    async def test_llm_routing_parses_json(self) -> None:
        """Setup: mock LLM returning JSON. Run: route. Analyze: parsed decision."""
        from langgraph_kit.core.orchestration.routing import (
            AgentCapability,
            LLMRoutingStrategy,
        )

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content='{"target_agent_id": "coding-agent", "reasoning": "code question", "delegated_prompt": "fix the bug"}'
        )

        caps = [
            AgentCapability(
                agent_id="coding-agent",
                name="Coding",
                description="Code",
                tags=["code"],
            ),
        ]

        router = LLMRoutingStrategy(mock_llm)
        decision = await router.route("Fix my code", caps)

        assert decision.target_agent_id == "coding-agent"
        assert decision.reasoning == "code question"
        assert decision.delegated_prompt == "fix the bug"

    @pytest.mark.asyncio
    async def test_llm_routing_fallback_on_error(self) -> None:
        """Setup: LLM that fails. Run: route. Analyze: graceful fallback to first agent."""
        from langgraph_kit.core.orchestration.routing import (
            AgentCapability,
            LLMRoutingStrategy,
        )

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = RuntimeError("API error")

        caps = [
            AgentCapability(agent_id="fallback", name="Fallback", description="Default")
        ]

        router = LLMRoutingStrategy(mock_llm)
        decision = await router.route("Hello", caps)

        assert decision.target_agent_id == "fallback"
        assert "fallback" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_llm_routing_tags_call_as_internal(self) -> None:
        """Router's ainvoke must carry INTERNAL_TAG + AGENT_ROUTING_TAG.

        The router fires an LLM call while an outer graph may be streaming
        to the user; tagging prevents its "here's who should handle this"
        JSON from leaking into the user-visible chat stream.
        """
        from langgraph_kit.core.internal_tags import (
            AGENT_ROUTING_TAG,
            INTERNAL_TAG,
        )
        from langgraph_kit.core.orchestration.routing import (
            AgentCapability,
            LLMRoutingStrategy,
        )

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content='{"target_agent_id": "a", "reasoning": "r", "delegated_prompt": "p"}'
        )

        caps = [AgentCapability(agent_id="a", name="A", description="A", tags=[])]
        router = LLMRoutingStrategy(mock_llm)
        await router.route("hi", caps)

        mock_llm.ainvoke.assert_awaited_once()
        config = mock_llm.ainvoke.call_args.kwargs.get("config")
        assert config is not None
        tags = config.get("tags", [])
        assert INTERNAL_TAG in tags
        assert AGENT_ROUTING_TAG in tags
        assert config.get("run_name") == "agent_routing"

    def test_supervisor_excludes_self_from_capabilities(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: the ID in the self-exclusion check must match the registered ID.

        Before 0.6.0 the agent was registered as ``"supervisor"``; after the
        rename it is ``"supervisor-agent"`` but the filter wasn't updated,
        which made the supervisor include itself in the delegation pool.
        """
        from langgraph_kit.graphs import supervisor_agent
        from langgraph_kit.registry import AgentMetadata

        fake_meta = AgentMetadata(description="test", tags=[], capabilities=[])
        monkeypatch.setattr(supervisor_agent, "get_metadata", lambda _id: fake_meta)

        caps = supervisor_agent._build_capabilities(
            [
                {"id": supervisor_agent.SUPERVISOR_AGENT_ID, "name": "Supervisor"},
                {"id": "echo-agent", "name": "Echo"},
            ]
        )

        ids = [c.agent_id for c in caps]
        assert supervisor_agent.SUPERVISOR_AGENT_ID not in ids
        assert "echo-agent" in ids


# ===========================================================================
# Feature 6: Thread Management API
# ===========================================================================


class TestThreadManagement:
    """Test Store-backed thread metadata CRUD and indexing."""

    @pytest.mark.asyncio
    async def test_ensure_thread_creates_new(self, mock_store: MockStore) -> None:
        """Setup: empty store. Run: ensure_thread. Analyze: metadata created with title from message."""
        from langgraph_kit.core.threads import ThreadManager

        mgr = ThreadManager(mock_store)
        meta = await mgr.ensure_thread(
            "t1", "user-1", "echo-agent", "Hello world, this is my first message"
        )

        assert meta.thread_id == "t1"
        assert meta.user_id == "user-1"
        assert meta.agent_id == "echo-agent"
        assert meta.title == "Hello world, this is my first message"
        assert meta.message_count == 1
        assert meta.created_at != ""

    @pytest.mark.asyncio
    async def test_ensure_thread_updates_existing(self, mock_store: MockStore) -> None:
        """Setup: thread exists. Run: ensure again. Analyze: message count incremented."""
        from langgraph_kit.core.threads import ThreadManager

        mgr = ThreadManager(mock_store)
        await mgr.ensure_thread("t1", "user-1", "echo-agent", "First message")
        meta = await mgr.ensure_thread("t1", "user-1", "echo-agent", "Second message")

        assert meta.message_count == 2
        assert meta.last_message_preview == "Second message"

    @pytest.mark.asyncio
    async def test_get_thread(self, mock_store: MockStore) -> None:
        from langgraph_kit.core.threads import ThreadManager

        mgr = ThreadManager(mock_store)
        await mgr.ensure_thread("t1", "user-1", "echo-agent", "Hi")

        meta = await mgr.get("t1")
        assert meta is not None
        assert meta.thread_id == "t1"

        missing = await mgr.get("nonexistent")
        assert missing is None

    @pytest.mark.asyncio
    async def test_list_for_user(self, mock_store: MockStore) -> None:
        """Setup: create threads for 2 users. Run: list for user-1. Analyze: only user-1 threads."""
        from langgraph_kit.core.threads import ThreadManager

        mgr = ThreadManager(mock_store)
        await mgr.ensure_thread("t1", "user-1", "echo-agent", "User 1 thread")
        await mgr.ensure_thread("t2", "user-2", "echo-agent", "User 2 thread")
        await mgr.ensure_thread(
            "t3", "user-1", "reference-deep-agent", "User 1 another"
        )

        threads, total = await mgr.list_for_user("user-1")
        assert total == 2
        thread_ids = {t.thread_id for t in threads}
        assert thread_ids == {"t1", "t3"}

    @pytest.mark.asyncio
    async def test_list_for_user_filtered_by_agent(self, mock_store: MockStore) -> None:
        from langgraph_kit.core.threads import ThreadManager

        mgr = ThreadManager(mock_store)
        await mgr.ensure_thread("t1", "user-1", "echo-agent", "Echo")
        await mgr.ensure_thread("t2", "user-1", "reference-deep-agent", "Reference")

        threads, total = await mgr.list_for_user("user-1", agent_id="echo-agent")
        assert total == 1
        assert threads[0].agent_id == "echo-agent"

    @pytest.mark.asyncio
    async def test_update_thread(self, mock_store: MockStore) -> None:
        from langgraph_kit.core.threads import ThreadManager

        mgr = ThreadManager(mock_store)
        await mgr.ensure_thread("t1", "user-1", "echo-agent", "Original title")

        updated = await mgr.update("t1", title="New Title", tags=["important"])
        assert updated is not None
        assert updated.title == "New Title"
        assert updated.tags == ["important"]

    @pytest.mark.asyncio
    async def test_delete_thread(self, mock_store: MockStore) -> None:
        from langgraph_kit.core.threads import ThreadManager

        mgr = ThreadManager(mock_store)
        await mgr.ensure_thread("t1", "user-1", "echo-agent", "To delete")

        deleted = await mgr.delete("t1")
        assert deleted is True

        meta = await mgr.get("t1")
        assert meta is None

    @pytest.mark.asyncio
    async def test_search_threads(self, mock_store: MockStore) -> None:
        from langgraph_kit.core.threads import ThreadManager

        mgr = ThreadManager(mock_store)
        await mgr.ensure_thread("t1", "user-1", "echo-agent", "Python debugging help")
        await mgr.ensure_thread("t2", "user-1", "echo-agent", "JavaScript frameworks")
        await mgr.ensure_thread("t3", "user-1", "echo-agent", "Python machine learning")

        results = await mgr.search("user-1", "python")
        titles = {r.title for r in results}
        assert "Python debugging help" in titles
        assert "Python machine learning" in titles
        assert "JavaScript frameworks" not in titles

    @pytest.mark.asyncio
    async def test_title_truncation(self, mock_store: MockStore) -> None:
        """Long messages should be truncated to 60 chars for titles."""
        from langgraph_kit.core.threads import ThreadManager

        mgr = ThreadManager(mock_store)
        long_msg = "A" * 100
        meta = await mgr.ensure_thread("t1", "user-1", "echo-agent", long_msg)

        assert len(meta.title) <= 63  # 60 + "..."
        assert meta.title.endswith("...")


# ===========================================================================
# Feature 7: MCP Server Mode
# ===========================================================================


class TestMCPServerMode:
    """Test MCP server creation and agent tool registration."""

    def test_create_mcp_server_registers_tools(self) -> None:
        """Setup: register agents. Run: create MCP server. Analyze: tools registered."""
        from langgraph_kit.registry import AgentMetadata, _metadata, _registry, register

        _registry.clear()
        _metadata.clear()

        register("echo-agent", MagicMock(), metadata=AgentMetadata(description="Echo"))
        register(
            "basic-deep-agent",
            MagicMock(),
            metadata=AgentMetadata(description="Basic deep"),
        )

        # Mock FastMCP to capture tool registrations
        registered_tools: list[dict[str, Any]] = []

        class FakeMCP:
            def __init__(self, name: str) -> None:
                self.name = name

            def tool(self, name: str = "", description: str = ""):
                def decorator(fn):
                    registered_tools.append(
                        {"name": name, "description": description, "fn": fn}
                    )
                    return fn

                return decorator

        # FastMCP is imported inside create_mcp_server, so patch the source module
        with patch.dict(
            "sys.modules",
            {
                "mcp": MagicMock(),
                "mcp.server": MagicMock(),
                "mcp.server.fastmcp": MagicMock(FastMCP=FakeMCP),
            },
        ):
            # Need to reload the module to pick up the mock
            import importlib

            import langgraph_kit.contrib.mcp_server as mcp_mod

            importlib.reload(mcp_mod)
            mcp_mod.create_mcp_server("test-server")

        assert len(registered_tools) == 2
        tool_names = {t["name"] for t in registered_tools}
        assert "invoke_echo_agent" in tool_names
        assert "invoke_basic_deep_agent" in tool_names


# ===========================================================================
# Feature 8: Graph Execution Trace Export
# ===========================================================================


class TestTraceExport:
    """Test execution trace collection, Mermaid export, and Store persistence."""

    @pytest.mark.asyncio
    async def test_trace_handler_collects_spans(self) -> None:
        """Setup: trace handler. Run: simulate chain→llm→tool lifecycle. Analyze: correct spans."""
        from langgraph_kit.core.tracing import TraceCallbackHandler

        handler = TraceCallbackHandler(agent_id="test", thread_id="t1")

        # Simulate: chain start → llm start → llm end → tool start → tool end → chain end
        await handler.on_chain_start({"name": "agent"}, {}, run_id="chain-1")
        await handler.on_chat_model_start(
            {"kwargs": {"model_name": "gpt-4o"}},
            [[]],
            run_id="llm-1",
            parent_run_id="chain-1",
        )
        await handler.on_llm_end(MagicMock(), run_id="llm-1")
        await handler.on_tool_start(
            {"name": "search"}, "{}", run_id="tool-1", parent_run_id="chain-1"
        )
        await handler.on_tool_end("results", run_id="tool-1")
        await handler.on_chain_end({}, run_id="chain-1")

        trace = handler.get_trace()

        assert trace.agent_id == "test"
        assert trace.thread_id == "t1"
        assert trace.duration_ms > 0
        assert len(trace.spans) == 1  # One root span (chain-1)

        root = trace.spans[0]
        assert root.name == "agent"
        assert root.kind == "chain"
        assert len(root.children) == 2  # llm-1 and tool-1

        llm_span = root.children[0]
        assert llm_span.kind == "llm"
        assert llm_span.duration_ms is not None
        assert llm_span.duration_ms >= 0

        tool_span = root.children[1]
        assert tool_span.kind == "tool"
        assert tool_span.name == "search"

    def test_trace_span_count(self) -> None:
        """Test the span_count property counts nested children."""
        from langgraph_kit.core.tracing import TraceRecord, TraceSpan

        trace = TraceRecord(
            spans=[
                TraceSpan(
                    name="root",
                    children=[
                        TraceSpan(name="child1"),
                        TraceSpan(
                            name="child2",
                            children=[
                                TraceSpan(name="grandchild"),
                            ],
                        ),
                    ],
                ),
            ],
        )
        assert trace.span_count == 4  # root + child1 + child2 + grandchild

    def test_mermaid_sequence_diagram(self) -> None:
        """Setup: trace with spans. Run: generate mermaid. Analyze: valid diagram."""
        from langgraph_kit.core.tracing import TraceRecord, TraceSpan, trace_to_mermaid

        trace = TraceRecord(
            spans=[
                TraceSpan(
                    name="agent",
                    kind="chain",
                    duration_ms=500.0,
                    children=[
                        TraceSpan(name="gpt-4o", kind="llm", duration_ms=200.0),
                        TraceSpan(name="search", kind="tool", duration_ms=100.0),
                    ],
                ),
            ],
        )

        diagram = trace_to_mermaid(trace, style="sequence")
        assert diagram.startswith("sequenceDiagram")
        assert "Agent->>LLM:" in diagram
        assert "Agent->>Tool:" in diagram
        assert "200ms" in diagram

    def test_mermaid_flowchart(self) -> None:
        from langgraph_kit.core.tracing import TraceRecord, TraceSpan, trace_to_mermaid

        trace = TraceRecord(
            spans=[
                TraceSpan(name="route", kind="chain", duration_ms=10.0),
                TraceSpan(name="generate", kind="llm", duration_ms=200.0),
            ],
        )

        diagram = trace_to_mermaid(trace, style="flowchart")
        assert diagram.startswith("flowchart TD")
        assert "route" in diagram
        assert "generate" in diagram

    @pytest.mark.asyncio
    async def test_trace_store_save_and_list(self, mock_store: MockStore) -> None:
        """Setup: trace store. Run: save trace and list. Analyze: trace persisted."""
        from langgraph_kit.core.tracing import TraceRecord, TraceStore

        store = TraceStore(mock_store)
        trace = TraceRecord(
            trace_id="tr-1",
            thread_id="t1",
            duration_ms=150.0,
            started_at="2024-01-01T00:00:00Z",
        )

        await store.save_trace("t1", trace)
        summaries = await store.list_traces("t1")

        assert len(summaries) == 1
        assert summaries[0].trace_id == "tr-1"
        assert summaries[0].duration_ms == 150.0

    @pytest.mark.asyncio
    async def test_trace_store_get(self, mock_store: MockStore) -> None:
        from langgraph_kit.core.tracing import TraceRecord, TraceStore

        store = TraceStore(mock_store)
        trace = TraceRecord(trace_id="tr-1", thread_id="t1", duration_ms=100.0)
        await store.save_trace("t1", trace)

        loaded = await store.get_trace("t1", "tr-1")
        assert loaded is not None
        assert loaded.trace_id == "tr-1"

    @pytest.mark.asyncio
    async def test_trace_handler_error_spans(self) -> None:
        """Test that error spans record the error message."""
        from langgraph_kit.core.tracing import TraceCallbackHandler

        handler = TraceCallbackHandler()
        await handler.on_tool_start({"name": "risky"}, "{}", run_id="t1")
        await handler.on_tool_error(RuntimeError("boom"), run_id="t1")

        trace = handler.get_trace()
        tool_span = trace.spans[0]
        assert tool_span.metadata.get("error") == "boom"


# ===========================================================================
# Feature 9: File Upload
# ===========================================================================


class TestFileUpload:
    """Test file attachment model and multi-part content conversion."""

    def test_file_attachment_model(self) -> None:
        """Setup: create FileAttachment. Analyze: fields correct."""
        from langgraph_kit.models import FileAttachment

        att = FileAttachment(
            name="photo.png",
            type="image/png",
            size=45000,
            data_url="data:image/png;base64,iVBOR...",
        )
        assert att.name == "photo.png"
        assert att.type == "image/png"
        assert att.size == 45000

    def test_chat_message_with_attachments(self) -> None:
        """Setup: ChatMessage with attachments. Analyze: serializes correctly."""
        from langgraph_kit.models import ChatMessage, FileAttachment

        msg = ChatMessage(
            role="user",
            content="What's in this image?",
            attachments=[
                FileAttachment(
                    name="img.png",
                    type="image/png",
                    size=1000,
                    data_url="data:image/png;base64,abc",
                ),
            ],
        )
        data = msg.model_dump()
        assert len(data["attachments"]) == 1
        assert data["attachments"][0]["name"] == "img.png"

    def test_chat_message_without_attachments_backwards_compatible(self) -> None:
        """Setup: ChatMessage without attachments field. Analyze: defaults to empty list."""
        from langgraph_kit.models import ChatMessage

        msg = ChatMessage(role="user", content="Hello")
        assert msg.attachments == []

        # From dict without attachments field
        msg2 = ChatMessage.model_validate({"role": "user", "content": "Hi"})
        assert msg2.attachments == []

    def test_multipart_content_builder_image(self) -> None:
        """Setup: message with image attachment. Run: build multipart. Analyze: image_url part."""
        from langgraph_kit.contrib.fastapi import _build_multipart_content
        from langgraph_kit.models import ChatMessage, FileAttachment

        msg = ChatMessage(
            role="user",
            content="Describe this",
            attachments=[
                FileAttachment(
                    name="photo.png",
                    type="image/png",
                    size=1000,
                    data_url="data:image/png;base64,abc123",
                ),
            ],
        )

        parts = _build_multipart_content(msg)
        assert len(parts) == 2
        assert parts[0] == {"type": "text", "text": "Describe this"}
        assert parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["url"] == "data:image/png;base64,abc123"

    def test_multipart_content_builder_text_file(self) -> None:
        """Setup: message with text file. Run: build multipart. Analyze: decoded text part."""
        import base64

        from langgraph_kit.contrib.fastapi import _build_multipart_content
        from langgraph_kit.models import ChatMessage, FileAttachment

        content = base64.b64encode(b"Hello world").decode()
        msg = ChatMessage(
            role="user",
            content="Read this file",
            attachments=[
                FileAttachment(
                    name="readme.txt",
                    type="text/plain",
                    size=11,
                    data_url=f"data:text/plain;base64,{content}",
                ),
            ],
        )

        parts = _build_multipart_content(msg)
        assert len(parts) == 2
        assert parts[0]["text"] == "Read this file"
        assert "[File: readme.txt]" in parts[1]["text"]
        assert "Hello world" in parts[1]["text"]

    def test_multipart_content_builder_pdf(self) -> None:
        """PDFs should be passed as image_url type (LLM providers support this)."""
        from langgraph_kit.contrib.fastapi import _build_multipart_content
        from langgraph_kit.models import ChatMessage, FileAttachment

        msg = ChatMessage(
            role="user",
            content="",
            attachments=[
                FileAttachment(
                    name="doc.pdf",
                    type="application/pdf",
                    size=5000,
                    data_url="data:application/pdf;base64,abc",
                ),
            ],
        )

        parts = _build_multipart_content(msg)
        assert len(parts) == 1  # No text part (content is empty)
        assert parts[0]["type"] == "image_url"

    def test_extract_text_content_string(self) -> None:
        """Test extracting text from string content."""
        from langgraph_kit.contrib.fastapi import _extract_text_content

        assert _extract_text_content("hello") == "hello"

    def test_extract_text_content_multipart(self) -> None:
        """Test extracting text from multi-part content list."""
        from langgraph_kit.contrib.fastapi import _extract_text_content

        content = [
            {"type": "text", "text": "Hello"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
            {"type": "text", "text": "world"},
        ]
        result = _extract_text_content(content)
        assert "Hello" in result
        assert "world" in result
        assert "[image]" in result
