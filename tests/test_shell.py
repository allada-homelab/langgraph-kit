"""Tests for ``langgraph_kit.shell`` (issue #37 v1)."""

from __future__ import annotations

import sys
from typing import Any

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    AIMessage,
)

from langgraph_kit import registry
from langgraph_kit.shell import _format_assistant_output, run_shell

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _ScriptedGraph:
    """Mock graph that returns a canned final assistant message."""

    def __init__(self, reply: str = "echo") -> None:
        self.reply = reply
        self.calls: list[tuple[Any, Any]] = []

    async def ainvoke(
        self, input_data: Any, config: Any | None = None
    ) -> dict[str, Any]:
        self.calls.append((input_data, config))
        return {"messages": [AIMessage(content=self.reply)]}


class _CrashingGraph:
    """Mock graph whose ``ainvoke`` always raises — exercises REPL error path."""

    async def ainvoke(self, *_: Any, **__: Any) -> Any:
        msg = "boom"
        raise RuntimeError(msg)


class _StubInputs:
    """Replays a fixed list of user inputs through the REPL's read loop.

    Raises ``EOFError`` after the script is exhausted to mimic a closed
    stdin (which the real REPL handles as "exit cleanly").
    """

    def __init__(self, lines: list[str]) -> None:
        self._iter = iter(lines)

    def __call__(self, _prompt: str) -> str:
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise EOFError from exc


# ---------------------------------------------------------------------------
# Reset the in-process registry between tests to keep state isolated.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_registry() -> Any:
    snapshot = dict(registry.get_all())
    # Clear and refill from snapshot so tests can register their own agents
    # without leaking into other tests.
    registry._registry.clear()
    yield
    registry._registry.clear()
    registry._registry.update(snapshot)


# ---------------------------------------------------------------------------
# _format_assistant_output
# ---------------------------------------------------------------------------


class TestFormatAssistantOutput:
    def test_string_content_pass_through(self) -> None:
        result = {"messages": [AIMessage(content="hi")]}
        assert _format_assistant_output(result) == "hi"

    def test_multipart_list_content_flattened(self) -> None:
        result = {
            "messages": [
                AIMessage(
                    content=[
                        {"type": "text", "text": "part1 "},
                        {"type": "text", "text": "part2"},
                    ]
                )
            ]
        }
        assert _format_assistant_output(result) == "part1 part2"

    def test_no_messages_returns_placeholder(self) -> None:
        assert _format_assistant_output({"messages": []}) == "(no message)"

    def test_non_chat_result_falls_back_to_str(self) -> None:
        # Some graphs return a bare string or dict without ``messages``.
        assert _format_assistant_output("just text") == "just text"
        assert _format_assistant_output({"score": 0.9}) == str({"score": 0.9})


# ---------------------------------------------------------------------------
# run_shell
# ---------------------------------------------------------------------------


class TestRunShell:
    @pytest.mark.asyncio
    async def test_invokes_agent_and_renders_reply(self) -> None:
        graph = _ScriptedGraph(reply="hello world")
        registry.register("test-agent", graph)
        outputs: list[str] = []
        rc = await run_shell(
            "test-agent",
            input_fn=_StubInputs(["ping"]),
            output_fn=outputs.append,
        )
        assert rc == 0
        # First two outputs are header + usage hint; the assistant reply follows.
        assert any("hello world" in line for line in outputs), outputs
        assert len(graph.calls) == 1
        input_data, config = graph.calls[0]
        assert input_data["messages"][0].content == "ping"
        assert config["configurable"]["thread_id"].startswith("shell-")
        assert config["configurable"]["user_id"] == "shell-user"

    @pytest.mark.asyncio
    async def test_thread_id_passed_through(self) -> None:
        graph = _ScriptedGraph()
        registry.register("test-agent", graph)
        outputs: list[str] = []
        await run_shell(
            "test-agent",
            thread_id="custom-thread",
            input_fn=_StubInputs(["x"]),
            output_fn=outputs.append,
        )
        _, config = graph.calls[0]
        assert config["configurable"]["thread_id"] == "custom-thread"

    @pytest.mark.asyncio
    async def test_user_id_passed_through(self) -> None:
        graph = _ScriptedGraph()
        registry.register("test-agent", graph)
        outputs: list[str] = []
        await run_shell(
            "test-agent",
            user_id="custom-user",
            input_fn=_StubInputs(["x"]),
            output_fn=outputs.append,
        )
        _, config = graph.calls[0]
        assert config["configurable"]["user_id"] == "custom-user"

    @pytest.mark.asyncio
    async def test_exit_command_stops_loop_cleanly(self) -> None:
        graph = _ScriptedGraph()
        registry.register("test-agent", graph)
        outputs: list[str] = []
        rc = await run_shell(
            "test-agent",
            input_fn=_StubInputs(["/exit"]),
            output_fn=outputs.append,
        )
        assert rc == 0
        # No agent invocation — exit short-circuited.
        assert graph.calls == []

    @pytest.mark.asyncio
    async def test_blank_input_is_skipped(self) -> None:
        graph = _ScriptedGraph()
        registry.register("test-agent", graph)
        outputs: list[str] = []
        await run_shell(
            "test-agent",
            input_fn=_StubInputs(["", "   ", "real"]),
            output_fn=outputs.append,
        )
        # Only the third (non-blank) input went to the agent.
        assert len(graph.calls) == 1
        assert graph.calls[0][0]["messages"][0].content == "real"

    @pytest.mark.asyncio
    async def test_eof_exits_cleanly_with_zero(self) -> None:
        graph = _ScriptedGraph()
        registry.register("test-agent", graph)
        outputs: list[str] = []
        rc = await run_shell(
            "test-agent",
            input_fn=_StubInputs([]),  # no inputs — first read raises EOFError
            output_fn=outputs.append,
        )
        assert rc == 0
        assert graph.calls == []

    @pytest.mark.asyncio
    async def test_unknown_agent_returns_exit_code_2(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Don't register anything; rely on register_all (which the helper
        # invokes when the requested id isn't there).
        rc = await run_shell(
            "no-such-agent",
            input_fn=_StubInputs([]),
            output_fn=lambda _: None,
        )
        assert rc == 2
        captured = capsys.readouterr()
        assert "not registered" in captured.err

    @pytest.mark.asyncio
    async def test_user_module_import_failure_returns_2(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = await run_shell(
            "any",
            user_module="this.module.definitely.does.not.exist",
            input_fn=_StubInputs([]),
            output_fn=lambda _: None,
        )
        assert rc == 2
        captured = capsys.readouterr()
        assert "Couldn't import" in captured.err

    @pytest.mark.asyncio
    async def test_agent_crash_keeps_repl_alive(self) -> None:
        registry.register("crashing", _CrashingGraph())
        outputs: list[str] = []
        rc = await run_shell(
            "crashing",
            input_fn=_StubInputs(["first try", "/exit"]),
            output_fn=outputs.append,
        )
        assert rc == 0
        # Error rendered via the assistant prefix; loop continued past the crash.
        assert any("(error" in line for line in outputs), outputs

    @pytest.mark.asyncio
    async def test_user_module_imported_when_provided(self) -> None:
        """``user_module`` is imported before the registry lookup."""
        # Use ``sys`` itself as the "user module" — it's always importable
        # and importing it is a no-op. Then register the agent inline so the
        # lookup still works.
        registry.register("u-agent", _ScriptedGraph(reply="ok"))
        sys.modules.pop("langgraph_kit.shell", None)  # nudge importlib cache
        outputs: list[str] = []
        rc = await run_shell(
            "u-agent",
            user_module="sys",
            input_fn=_StubInputs(["hi"]),
            output_fn=outputs.append,
        )
        assert rc == 0
        assert any("ok" in line for line in outputs)

    @pytest.mark.asyncio
    async def test_info_slash_command_prints_session_metadata(self) -> None:
        """``/info`` reports agent / thread / user / module without invoking the agent."""
        graph = _ScriptedGraph(reply="ignored")
        registry.register("test-agent", graph)
        outputs: list[str] = []
        await run_shell(
            "test-agent",
            thread_id="my-thread",
            user_id="me",
            input_fn=_StubInputs(["/info"]),
            output_fn=outputs.append,
        )

        info_lines = [line for line in outputs if "agent_id" in line]
        assert info_lines, outputs
        info = info_lines[0]
        assert "test-agent" in info
        assert "my-thread" in info
        assert "me" in info
        # /info must not invoke the graph — that would burn budget on a
        # cosmetic introspection command.
        assert graph.calls == []
