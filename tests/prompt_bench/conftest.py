"""Pytest fixtures for the prompt-bench harness.

Lives at the package root (under ``tests/``) so unit tests can import
fixtures with no ``--rootdir`` gymnastics. Provides:

- ``bench_llm`` — chat model used for *agent execution*. Hermetic by
  default (a deterministic stub). When ``LLM_BASE_URL`` /
  ``LLM_API_KEY`` / ``LLM_MODEL`` are all set, returns a
  ``ChatOpenAI`` pointed at the proxy, optionally pinned via
  ``BENCH_EXECUTOR_MODEL``.
- ``judge_llms`` — two-model panel for pairwise judging. Stub by
  default; real mode reads ``BENCH_JUDGE_MODEL_A`` and
  ``BENCH_JUDGE_MODEL_B`` (each falling back to ``LLM_MODEL`` if
  unset).
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
# Mode detection — uses the same OpenAI-compatible env vars as ``run.py``
# (``LLM_BASE_URL`` / ``LLM_API_KEY`` / ``LLM_MODEL``). Real mode is active
# only when all three are present; otherwise fixtures fall back to the
# deterministic stub so unit tests don't accidentally hit a network.
# ---------------------------------------------------------------------------

_BASE_URL_ENV = "LLM_BASE_URL"
_API_KEY_ENV = "LLM_API_KEY"
_MODEL_ENV = "LLM_MODEL"
_EXECUTOR_MODEL_ENV = "BENCH_EXECUTOR_MODEL"
_JUDGE_A_MODEL_ENV = "BENCH_JUDGE_MODEL_A"
_JUDGE_B_MODEL_ENV = "BENCH_JUDGE_MODEL_B"


def _real_llm_enabled() -> bool:
    return all(
        os.environ.get(name) for name in (_BASE_URL_ENV, _API_KEY_ENV, _MODEL_ENV)
    )


def _role_model(role_env: str) -> str:
    return os.environ.get(role_env) or os.environ[_MODEL_ENV]


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
        self._responses = responses or [
            {"winner": "tie", "confidence": 0.5, "reason": "stub"}
        ]
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

    Hermetic stub by default; switches to a real ``ChatOpenAI`` pointed
    at ``LLM_BASE_URL`` when the three required env vars are all set.
    """
    if _real_llm_enabled():
        return _build_real_chat_model(_role_model(_EXECUTOR_MODEL_ENV))
    return _DeterministicStub()


@pytest.fixture
def judge_llms() -> list[Any]:
    """Two-model panel for pairwise judging (real or stub)."""
    if _real_llm_enabled():
        return [
            _build_real_chat_model(_role_model(_JUDGE_A_MODEL_ENV)),
            _build_real_chat_model(_role_model(_JUDGE_B_MODEL_ENV)),
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


def _build_real_chat_model(model_name: str) -> Any:
    """Return a ``ChatOpenAI`` pointed at ``LLM_BASE_URL``.

    The bench targets OpenAI-compatible endpoints (any proxy that
    speaks ``/v1/chat/completions``). Per-role model overrides go
    through ``BENCH_EXECUTOR_MODEL`` / ``BENCH_JUDGE_MODEL_A`` /
    ``BENCH_JUDGE_MODEL_B``; this function takes a resolved name.
    """
    from langchain_openai import (  # pyright: ignore[reportMissingImports]
        ChatOpenAI,
    )

    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": os.environ[_API_KEY_ENV],
        "base_url": os.environ[_BASE_URL_ENV],
        "max_tokens": 1024,
        "timeout": 60,
    }
    return ChatOpenAI(**kwargs)  # pyright: ignore[reportCallIssue]


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
        # Langfuse / langsmith / langgraph clients sometimes leak
        # sockets and event loops during pytest's GC sweep, which
        # pytest's unraisable-exception plugin then re-raises as a
        # session-level error. The leaks aren't from our code (they're
        # from third-party HTTP client singletons) and don't affect
        # correctness — silence them here.
        warnings.simplefilter("ignore", ResourceWarning)
        yield


# Note: pytest's ``unraisableexception`` plugin is disabled at the
# project level via ``addopts`` in ``pyproject.toml``. Third-party HTTP
# clients (langfuse / langsmith / langgraph defaults) leak sockets +
# event loops during pytest's GC sweep at session exit, which the
# plugin would otherwise re-raise as a session-level error. Disabling
# it lets exit codes track actual test outcomes.
