"""Shared helpers for the examples directory.

Hermetic-by-default: examples run with no env vars, no API keys, using
:class:`~langgraph_kit.replay.RecordedChatModel`-backed scripted LLMs.
Real-LLM mode is gated behind ``LANGGRAPH_KIT_EXAMPLES_LLM=real`` and
requires ``AGENT_LLM_API_KEY``. All persisted state goes through
:func:`tmp_workspace` so demos never write to the user's home or repo.

This module intentionally mirrors patterns from
``tests/e2e/conftest.py`` + ``tests/e2e/helpers.py`` rather than
introducing new infrastructure — if a test pattern works for the
package's own e2e suite, it works for a demo.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from langgraph_kit.replay import (
    ConversationRecording,
    LLMInteraction,
    RecordedChatModel,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


# Environment variables that gate execution mode.
_LLM_MODE_ENV = "LANGGRAPH_KIT_EXAMPLES_LLM"
_NETWORK_ENV = "RUN_NETWORK"
_REAL_LLM_KEY_ENV = "AGENT_LLM_API_KEY"

# Cost guardrail for real-LLM mode.
DEFAULT_REAL_LLM_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_OUTPUT_TOKENS = 512


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------


def hermetic() -> bool:
    """Return ``True`` when the example should use a scripted LLM.

    Default. Flip with ``LANGGRAPH_KIT_EXAMPLES_LLM=real``.
    """
    return os.environ.get(_LLM_MODE_ENV, "scripted").lower() != "real"


def network_enabled() -> bool:
    """Whether the runner has explicitly opted in to network-touching demos."""
    return os.environ.get(_NETWORK_ENV) == "1"


def assert_real_llm_or_skip() -> None:
    """For examples that strictly require a real LLM (e.g. observability).

    Exits 0 (a graceful skip — counts as success in the smoke suite) when
    hermetic mode is active or the API key is missing.
    """
    if hermetic():
        sys.stdout.write(
            f"skipping: set {_LLM_MODE_ENV}=real and {_REAL_LLM_KEY_ENV}=... to run\n"
        )
        sys.exit(0)
    if not os.environ.get(_REAL_LLM_KEY_ENV):
        sys.stdout.write(f"skipping: {_REAL_LLM_KEY_ENV} not set\n")
        sys.exit(0)


# ---------------------------------------------------------------------------
# Workspace lifecycle
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def tmp_workspace() -> Iterator[Path]:
    """Yield a path to a fresh tempdir; auto-cleaned on exit.

    All persisted demo state (sqlite checkpoints, store snapshots, audit
    logs, DR exports) goes through here so a curious user doesn't end up
    with leftover files in their home directory.
    """
    with tempfile.TemporaryDirectory(prefix="lgk-example-") as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# Persistence factories — InMemory by default; switch to sqlite if a demo
# really needs durability across processes.
# ---------------------------------------------------------------------------


def make_in_memory_persistence() -> tuple[Any, Any]:
    """Return ``(checkpointer, store)`` backed by in-memory implementations."""
    from langgraph.checkpoint.memory import (  # pyright: ignore[reportMissingImports]
        InMemorySaver,
    )
    from langgraph.store.memory import (  # pyright: ignore[reportMissingImports]
        InMemoryStore,
    )

    return InMemorySaver(), InMemoryStore()


# ---------------------------------------------------------------------------
# Scripted-LLM helpers — ported from tests/e2e/helpers.py.
# ---------------------------------------------------------------------------


def tool_call_turn(
    name: str,
    args: dict[str, Any] | None = None,
    call_id: str | None = None,
) -> dict[str, Any]:
    """Build a recorded ``output_message`` for a turn that calls one tool."""
    return {
        "content": "",
        "tool_calls": [
            {
                "id": call_id or f"call_{name}",
                "name": name,
                "args": args or {},
            }
        ],
    }


def answer(content: str) -> dict[str, Any]:
    """Build a recorded ``output_message`` for a final text response."""
    return {"content": content, "tool_calls": []}


class _LoopingScriptedChatModel(RecordedChatModel):
    """``RecordedChatModel`` that re-serves the last canned response.

    Plain ``RecordedChatModel`` raises :class:`ReplayMismatchError` once
    the recorded sequence is exhausted. That's right for tests but wrong
    for a demo, where middleware (extraction, consolidation, pressure
    monitor) may make extra LLM calls beyond the user-scripted turns and
    we'd rather degrade gracefully than crash the example.

    On overflow this subclass re-serves the last interaction verbatim.
    Extraction middleware that gets a non-JSON answer logs a warning and
    moves on; the demo's primary output (the user-visible reply) still
    shows up cleanly.
    """

    def _generate(  # type: ignore[override]
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        interactions = self.recording.llm_interactions
        if self._call_index < len(interactions):
            return super()._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )

        # Out of script — re-serve the last canned response so background
        # middleware calls don't blow up the demo.
        from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
            AIMessage,
        )
        from langchain_core.outputs import (  # pyright: ignore[reportMissingModuleSource]
            ChatGeneration,
            ChatResult,
        )

        if not interactions:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=""))]
            )

        last = interactions[-1].output_message
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=last.get("content", ""),
                        tool_calls=last.get("tool_calls") or [],
                    )
                )
            ]
        )


def scripted_llm(turns: list[dict[str, Any]]) -> RecordedChatModel:
    """Wrap *turns* as a looping scripted :class:`RecordedChatModel`.

    Examples run more LLM calls than they explicitly script (extraction
    middleware, consolidation, pressure monitor). Use the looping
    subclass so an example can declare its primary turn(s) without
    having to know how many internal calls the kit's middleware makes.
    """
    return _LoopingScriptedChatModel(
        recording=ConversationRecording(
            interactions=[
                LLMInteraction(sequence_num=i + 1, output_message=msg)
                for i, msg in enumerate(turns)
            ],
        )
    )


# Every place ``build_llm`` is bound *as a local name* — patching the
# canonical ``langgraph_kit.llm.build_llm`` symbol alone misses already-
# imported aliases. Add a target here when a new graph module starts
# calling ``build_llm`` directly.
_BUILD_LLM_PATCH_TARGETS: tuple[str, ...] = (
    "langgraph_kit.llm.build_llm",
    "langgraph_kit.graphs._builder.build_llm",
    "langgraph_kit.graphs.echo_agent.build_llm",
    "langgraph_kit.graphs.basic_deep_agent.build_llm",
    "langgraph_kit.graphs.supervisor_agent.build_llm",
)


@contextlib.contextmanager
def patch_build_llm(model: Any) -> Iterator[None]:
    """Patch every known ``build_llm`` site to return *model*.

    The patch must be active when :func:`build_llm` would normally fire
    — usually that's at graph-build time for the deep-agent flows and at
    invoke time for the simple echo flow. Wrap both build and invoke in
    the same ``with`` block to be safe.
    """
    with contextlib.ExitStack() as stack:
        for target in _BUILD_LLM_PATCH_TARGETS:
            try:
                stack.enter_context(patch(target, return_value=model))
            except (ModuleNotFoundError, AttributeError):
                # Not every install has every optional graph module; the
                # patch list is best-effort.
                continue
        yield


# ---------------------------------------------------------------------------
# Real-LLM mode — used by the network-tier examples.
# ---------------------------------------------------------------------------


def configure_real_llm(workspace: Path) -> None:
    """Install an :class:`AgentConfig` suitable for real-LLM example mode.

    Pins a cheap model (``claude-haiku-4-5``) and points the database
    URL at the workspace tempdir. Caller is responsible for setting
    ``AGENT_LLM_API_KEY`` before calling.
    """
    from langgraph_kit import AgentConfig, configure

    api_key = os.environ.get(_REAL_LLM_KEY_ENV, "")
    if not api_key:
        msg = (
            f"{_REAL_LLM_KEY_ENV} is not set; call assert_real_llm_or_skip() before "
            "configure_real_llm() in examples that hard-require the real model."
        )
        raise RuntimeError(msg)
    db_path = workspace / "checkpoints.db"
    configure(
        AgentConfig(
            llm_model=DEFAULT_REAL_LLM_MODEL,
            llm_api_key=api_key,
            database_url=f"sqlite:///{db_path}",
        )
    )


# ---------------------------------------------------------------------------
# Output formatting helpers — keep the demos visually consistent.
# ---------------------------------------------------------------------------


def banner(title: str) -> None:
    """Print a section banner. Examples use this for demarcation."""
    bar = "=" * len(title)
    sys.stdout.write(f"\n{title}\n{bar}\n")


def line(text: str = "") -> None:
    """Print a normal line. Centralised so a future renderer can intercept."""
    sys.stdout.write(text + "\n")
