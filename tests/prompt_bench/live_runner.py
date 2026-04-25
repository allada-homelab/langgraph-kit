"""Live execution wiring — connect ``BenchRunner`` to a real LangGraph agent.

Phase 0 shipped the abstract runner; this module gives it a concrete
``run_one`` callable that:

1. Applies *overlay*'s middleware patches for the lifetime of the call
   (sections are baked into the agent at build time; middleware
   constants are read each turn, so the patch must span build + invoke).
2. Builds the agent graph via :func:`profiles.build_profile_graph` with
   the active overlay and a fresh ``(checkpointer, store)`` pair.
3. Drives the scenario's turns through ``graph.ainvoke`` on a
   per-sample thread, capturing the final assistant text and every
   tool call observed in the message stream.
4. Returns a populated :class:`BenchSample`. Exceptions surface as
   ``error=`` on the sample so a single bad scenario can't abort
   the bench.

The driver and graph builder are injected so unit tests can substitute
mocks without paying LangGraph's import cost.
"""

from __future__ import annotations

import contextlib
import logging
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from tests.prompt_bench.profiles import build_profile_graph, install_default_builders
from tests.prompt_bench.runner import BenchSample
from tests.prompt_bench.variants import patch_middleware_prompts

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterator

    from tests.prompt_bench.scenarios import Scenario
    from tests.prompt_bench.variants import PromptOverlay

logger = logging.getLogger(__name__)


# Every site where the kit binds ``build_llm`` as a local name. Patching
# the canonical ``langgraph_kit.llm.build_llm`` symbol alone misses
# already-imported aliases — match this list to ``examples/_lib.py``.
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

    The patch must be active when the graph is built (the kit resolves
    ``build_llm`` eagerly during compilation). Wrapping both build and
    invoke in the same ``with`` is safe and idempotent — once the graph
    is compiled, it holds the model directly.
    """
    with contextlib.ExitStack() as stack:
        for target in _BUILD_LLM_PATCH_TARGETS:
            try:
                stack.enter_context(patch(target, return_value=model))
            except (ModuleNotFoundError, AttributeError):
                continue
        yield


# ---------------------------------------------------------------------------
# Default deps factory — fresh in-memory checkpointer + store per sample.
# ---------------------------------------------------------------------------


def in_memory_deps_factory() -> Callable[[], Any]:
    """Return ``() -> (checkpointer, store)`` using LangGraph's in-memory impls.

    Each sample gets a fresh pair so memory / checkpointer state can't
    leak from one scenario into another.
    """

    def _make() -> tuple[Any, Any]:
        from langgraph.checkpoint.memory import (  # pyright: ignore[reportMissingImports]
            InMemorySaver,
        )
        from langgraph.store.memory import (  # pyright: ignore[reportMissingImports]
            InMemoryStore,
        )

        return InMemorySaver(), InMemoryStore()

    return _make


# ---------------------------------------------------------------------------
# Default scenario driver — drives messages through ``graph.ainvoke``.
# ---------------------------------------------------------------------------


DriveScenario = Callable[[Any, "Scenario"], "Awaitable[dict[str, Any]]"]


async def default_drive_scenario(graph: Any, scenario: Scenario) -> dict[str, Any]:
    """Drive *scenario*'s turns through *graph*; capture trace fields.

    Uses the LangGraph ``messages`` channel (same shape as
    ``examples/_lib.py``). Each turn appends a HumanMessage; the final
    assistant message after all turns is the output we evaluate.
    """
    from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
        AIMessage,
        HumanMessage,
    )

    thread_id = f"prompt-bench-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}
    messages: list[Any] = []
    tool_calls: list[dict[str, Any]] = []
    final_output = ""

    for turn in scenario.turns:
        messages.append(HumanMessage(content=turn.user))
        result = await graph.ainvoke({"messages": messages}, config=config)
        if isinstance(result, dict) and "messages" in result:
            messages = list(result["messages"])

        for m in messages:
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                for call in m.tool_calls:
                    if isinstance(call, dict):
                        tool_calls.append(
                            {
                                "name": call.get("name"),
                                "args": call.get("args"),
                            }
                        )

        last_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage)),
            None,
        )
        if last_ai is not None:
            content = last_ai.content
            final_output = content if isinstance(content, str) else str(content)

    return {"final_output": final_output, "tool_calls": tool_calls}


# ---------------------------------------------------------------------------
# Run-one factory
# ---------------------------------------------------------------------------


BuildGraph = Callable[["PromptOverlay", Any], Any]


def make_run_one(
    profile_name: str,
    executor_llm: Any,
    *,
    deps_factory: Callable[[], Any] | None = None,
    build_graph: BuildGraph | None = None,
    drive_scenario: DriveScenario | None = None,
) -> Callable[[Scenario, PromptOverlay, int], Awaitable[BenchSample]]:
    """Build a profile-specific ``run_one`` callable for :class:`BenchRunner`.

    Parameters
    ----------
    profile_name:
        Key into :data:`profiles.PROFILES`. Determines the agent build path.
    executor_llm:
        Chat model the agent will call. Real Claude in production; a
        deterministic stub in unit tests.
    deps_factory:
        ``() -> (checkpointer, store)`` for each sample. Defaults to
        in-memory implementations.
    build_graph / drive_scenario:
        Override hooks for unit tests. Defaults wire through to
        :func:`profiles.build_profile_graph` and the LangGraph driver.
    """
    install_default_builders()
    deps_factory = deps_factory or in_memory_deps_factory()
    build_graph = build_graph or (
        lambda overlay, deps: build_profile_graph(profile_name, overlay, deps)
    )
    drive = drive_scenario or default_drive_scenario

    async def run_one(
        scenario: Scenario, overlay: PromptOverlay, sample_index: int
    ) -> BenchSample:
        deps = deps_factory()
        start = time.monotonic()
        try:
            with patch_middleware_prompts(overlay), patch_build_llm(executor_llm):
                graph = build_graph(overlay, deps)
                result = await drive(graph, scenario)
        except Exception as exc:
            logger.exception(
                "Scenario %s sample %d failed under overlay %s",
                scenario.id,
                sample_index,
                overlay.name,
            )
            return BenchSample(
                scenario_id=scenario.id,
                sample_index=sample_index,
                overlay_name=overlay.name,
                duration_ms=(time.monotonic() - start) * 1000,
                final_output="",
                error=type(exc).__name__,
            )

        duration_ms = (time.monotonic() - start) * 1000
        final_output = (
            result.get("final_output", "") if isinstance(result, dict) else str(result)
        )
        tool_calls = result.get("tool_calls", []) if isinstance(result, dict) else []
        return BenchSample(
            scenario_id=scenario.id,
            sample_index=sample_index,
            overlay_name=overlay.name,
            duration_ms=duration_ms,
            final_output=final_output,
            tool_calls=tool_calls,
        )

    return run_one
