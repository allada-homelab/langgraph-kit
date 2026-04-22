"""Assertion helpers for replay testing."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langgraph_kit.replay.models import ConversationRecording

if TYPE_CHECKING:
    from pathlib import Path


class ReplayAssertions:
    """Assertion helpers comparing an original recording against a replayed one.

    Usage::

        assertions = ReplayAssertions(original, replayed)
        assertions.assert_same_tool_sequence()
        assertions.assert_tool_called("web_search", times=2)
        assertions.assert_final_output_contains("summary")
    """

    def __init__(
        self,
        original: ConversationRecording,
        replayed: ConversationRecording,
    ) -> None:
        self.original = original
        self.replayed = replayed

    def assert_same_tool_sequence(self) -> None:
        """Assert both recordings called the same tools in the same order."""
        orig_seq = self.original.tool_sequence
        replay_seq = self.replayed.tool_sequence
        if orig_seq != replay_seq:
            msg = f"Tool sequence mismatch:\n  original: {orig_seq}\n  replayed: {replay_seq}"
            raise AssertionError(msg)

    def assert_same_tool_calls(self) -> None:
        """Assert both recordings called the same tools with the same arguments."""
        orig_tools = self.original.tool_interactions
        replay_tools = self.replayed.tool_interactions

        if len(orig_tools) != len(replay_tools):
            msg = (
                f"Tool call count mismatch: original={len(orig_tools)}, "
                f"replayed={len(replay_tools)}"
            )
            raise AssertionError(msg)

        for i, (orig, replay) in enumerate(zip(orig_tools, replay_tools, strict=True)):
            if orig.tool_name != replay.tool_name:
                msg = (
                    f"Tool #{i + 1} name mismatch: "
                    f"original={orig.tool_name!r}, replayed={replay.tool_name!r}"
                )
                raise AssertionError(msg)
            if orig.tool_input != replay.tool_input:
                msg = (
                    f"Tool #{i + 1} ({orig.tool_name}) args mismatch:\n"
                    f"  original: {orig.tool_input}\n"
                    f"  replayed: {replay.tool_input}"
                )
                raise AssertionError(msg)

    def assert_tool_called(self, name: str, *, times: int | None = None) -> None:
        """Assert a specific tool was called, optionally a specific number of times."""
        count = sum(1 for t in self.replayed.tool_interactions if t.tool_name == name)
        if count == 0:
            msg = f"Tool {name!r} was never called"
            raise AssertionError(msg)
        if times is not None and count != times:
            msg = f"Tool {name!r} called {count} times, expected {times}"
            raise AssertionError(msg)

    def assert_tool_not_called(self, name: str) -> None:
        """Assert a specific tool was never called."""
        count = sum(1 for t in self.replayed.tool_interactions if t.tool_name == name)
        if count > 0:
            msg = f"Tool {name!r} was called {count} times, expected 0"
            raise AssertionError(msg)

    def assert_final_output_contains(self, text: str) -> None:
        """Assert the final LLM output contains the given text."""
        final = self._get_final_output()
        if text not in final:
            msg = f"Final output does not contain {text!r}. Got: {final[:200]!r}"
            raise AssertionError(msg)

    def assert_final_output_matches(self, pattern: str) -> None:
        """Assert the final LLM output matches the given regex pattern."""
        final = self._get_final_output()
        if not re.search(pattern, final):
            msg = (
                f"Final output does not match pattern {pattern!r}. Got: {final[:200]!r}"
            )
            raise AssertionError(msg)

    def assert_no_errors(self) -> None:
        """Assert no tool interactions have error status."""
        errors = [t for t in self.replayed.tool_interactions if t.status == "error"]
        if errors:
            details = "; ".join(f"{t.tool_name}: {t.tool_output[:100]}" for t in errors)
            msg = f"{len(errors)} tool error(s): {details}"
            raise AssertionError(msg)

    def assert_output_similarity(self, *, min_ratio: float = 0.5) -> None:
        """Assert the replayed output is sufficiently similar to the original.

        Uses SequenceMatcher to compare the final LLM output text from both
        recordings. Useful when exact matching is too strict but you want to
        ensure the agent didn't produce wildly different output.
        """
        from difflib import SequenceMatcher

        orig_output = self._get_final_output_from(self.original)
        replay_output = self._get_final_output()
        if not orig_output and not replay_output:
            return  # Both empty is a match
        ratio = SequenceMatcher(None, orig_output, replay_output).ratio()
        if ratio < min_ratio:
            msg = (
                f"Output similarity {ratio:.2f} below threshold {min_ratio:.2f}.\n"
                f"  original: {orig_output[:150]!r}\n"
                f"  replayed: {replay_output[:150]!r}"
            )
            raise AssertionError(msg)

    def _get_final_output_from(self, recording: ConversationRecording) -> str:
        """Extract the final LLM output content from a recording."""
        llm_interactions = recording.llm_interactions
        if not llm_interactions:
            return ""
        return llm_interactions[-1].output_message.get("content", "")

    def _get_final_output(self) -> str:
        """Extract the final LLM output content from the replayed recording."""
        llm_interactions = self.replayed.llm_interactions
        if not llm_interactions:
            msg = "No LLM interactions in replayed recording"
            raise AssertionError(msg)
        content = llm_interactions[-1].output_message.get("content", "")
        if isinstance(content, list):
            # Multi-part content — join text parts
            return " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        return content


# ---------------------------------------------------------------------------
# Standalone assertion functions (for pytest-style usage)
# ---------------------------------------------------------------------------


def assert_replay_matches(
    original_path: Path,
    replayed: ConversationRecording,
    *,
    check_tool_args: bool = True,
) -> ReplayAssertions:
    """Load an original recording and assert it matches a replayed one.

    Returns the ``ReplayAssertions`` instance for further checks.
    """
    original = ConversationRecording.model_validate_json(original_path.read_text())
    assertions = ReplayAssertions(original, replayed)
    if check_tool_args:
        assertions.assert_same_tool_calls()
    else:
        assertions.assert_same_tool_sequence()
    return assertions


def assert_tool_sequence(
    recording: ConversationRecording,
    expected: list[str],
) -> None:
    """Assert the recording's tool call sequence matches the expected list."""
    actual = recording.tool_sequence
    if actual != expected:
        msg = f"Tool sequence mismatch:\n  expected: {expected}\n  actual:   {actual}"
        raise AssertionError(msg)
