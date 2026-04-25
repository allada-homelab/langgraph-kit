"""Pytest fixtures for the prompt-bench harness.

Lives at the package root (under ``tests/``) so unit tests can import
fixtures with no ``--rootdir`` gymnastics. Provides:

- ``bench_llm`` — chat model used for *agent execution*. Hermetic by
  default (a deterministic stub that echoes the input). When
  ``PROMPT_BENCH_LLM=real`` and ``AGENT_LLM_API_KEY`` is set, returns
  a real Claude model pinned to ``DEFAULT_EXECUTION_MODEL``.
- ``judge_llms`` — two-model panel for pairwise judging. Same hermetic
  default; real mode pulls in a Claude judge + an external judge if
  configured (otherwise two Claude judges with different temperature).
- ``bench_pairwise_panel`` — a :class:`PairwisePanel` wired up with
  ``judge_llms``.
- ``mock_section_registry_factory`` — returns a fresh
  ``SectionRegistry`` populated with ``reference_deep_agent``'s
  ``_CORE_SECTIONS``.
"""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING, Any

import pytest

from tests.prompt_bench.pairwise import PairwiseJudge, PairwisePanel

if TYPE_CHECKING:
    from collections.abc import Iterator


# ---------------------------------------------------------------------------
# Mode detection — separate env vars from the examples harness so the two
# harnesses don't accidentally cross-talk (examples use scripted by default
# even with PROMPT_BENCH_LLM=real, and vice versa).
# ---------------------------------------------------------------------------

_LLM_MODE_ENV = "PROMPT_BENCH_LLM"
_API_KEY_ENV = "AGENT_LLM_API_KEY"

# The model under evaluation (we want quality signal — Sonnet matters).
DEFAULT_EXECUTION_MODEL = "claude-sonnet-4-6"
# Judges should be capable but cheaper than the executor — Opus is the
# default high-end judge; the second judge is OpenAI / Gemini if their
# env vars are set, falling back to a second Claude with different temp.
DEFAULT_JUDGE_MODEL_PRIMARY = "claude-opus-4-7"
DEFAULT_JUDGE_MODEL_SECONDARY = "claude-haiku-4-5"


def _real_llm_enabled() -> bool:
    return os.environ.get(_LLM_MODE_ENV, "stub").lower() == "real"


# ---------------------------------------------------------------------------
# Stub chat model for hermetic tests
# ---------------------------------------------------------------------------


class _DeterministicStub:
    """Tiny chat-model substitute used in unit tests.

    Returns a fixed JSON for the pairwise judge protocol so we can drive
    panel logic without invoking a real LLM. Configure via ``responses``
    (a list of dicts that get returned in order; loops on overflow).
    """

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        super().__init__()
        self._responses = responses or [{"winner": "tie", "confidence": 0.5, "reason": "stub"}]
        self._call_index = 0

    async def ainvoke(self, _messages: list[Any]) -> Any:
        import json

        idx = self._call_index % len(self._responses)
        self._call_index += 1
        payload = json.dumps(self._responses[idx])

        class _Response:
            content = payload

        return _Response()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_llm() -> _DeterministicStub:
    """Default: a stub that always returns ``{"winner": "tie"}``."""
    return _DeterministicStub()


@pytest.fixture
def stub_llm_factory():
    """Factory: build a stub with a custom response list."""
    def _make(responses: list[dict[str, Any]]) -> _DeterministicStub:
        return _DeterministicStub(responses=responses)

    return _make


@pytest.fixture
def bench_llm() -> Any:
    """Chat model used for *agent execution* in scenarios.

    Hermetic stub by default; switches to real Claude when
    ``PROMPT_BENCH_LLM=real`` + ``AGENT_LLM_API_KEY`` are set.
    """
    if _real_llm_enabled() and os.environ.get(_API_KEY_ENV):
        return _build_real_chat_model(DEFAULT_EXECUTION_MODEL)
    return _DeterministicStub()


@pytest.fixture
def judge_llms() -> list[Any]:
    """Two-model panel for pairwise judging (real or stub)."""
    if _real_llm_enabled() and os.environ.get(_API_KEY_ENV):
        return [
            _build_real_chat_model(DEFAULT_JUDGE_MODEL_PRIMARY),
            _build_real_chat_model(DEFAULT_JUDGE_MODEL_SECONDARY),
        ]
    return [_DeterministicStub(), _DeterministicStub()]


@pytest.fixture
def bench_pairwise_panel(judge_llms: list[Any]) -> PairwisePanel:
    """Two-judge panel wired to :func:`judge_llms`."""
    judges = [
        PairwiseJudge(name=f"judge_{i + 1}", llm=llm)
        for i, llm in enumerate(judge_llms)
    ]
    return PairwisePanel(judges=judges, rng=random.Random(0))


@pytest.fixture
def reference_section_registry_factory():
    """Return a callable that builds a fresh registry for the reference agent."""
    from langgraph_kit.core.prompt_assembly.sections import SectionRegistry
    from langgraph_kit.graphs.reference_deep_agent import _CORE_SECTIONS

    def _make() -> SectionRegistry:
        registry = SectionRegistry()
        registry.register_many(_CORE_SECTIONS)
        return registry

    return _make


# ---------------------------------------------------------------------------
# Real chat model construction
# ---------------------------------------------------------------------------


def _build_real_chat_model(model_id: str) -> Any:
    """Build a real Anthropic chat model with the given model id.

    Matches what ``langgraph_kit.llm.build_llm`` would have produced,
    but lets us pin different models for executor vs judges without
    going through ``configure(AgentConfig(...))`` (which is global).
    """
    from langchain_anthropic import (  # pyright: ignore[reportMissingImports]
        ChatAnthropic,
    )

    # Pass via **kwargs so basedpyright's stricter stub doesn't flag the
    # well-known runtime kwargs (``model``, ``max_tokens``, ``timeout``)
    # as unrecognised — langchain_anthropic accepts them at runtime.
    kwargs: dict[str, Any] = {
        "model": model_id,
        "api_key": os.environ[_API_KEY_ENV],
        "max_tokens": 1024,
        "timeout": 60,
    }
    return ChatAnthropic(**kwargs)  # pyright: ignore[reportCallIssue]


# ---------------------------------------------------------------------------
# Disable the warnings-as-errors filter for the bench harness — we
# explicitly invoke real or stubbed LLM clients here and don't want a
# stray DeprecationWarning from langchain to nuke a 30-minute bench run.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _silence_bench_warnings() -> Iterator[None]:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        yield
