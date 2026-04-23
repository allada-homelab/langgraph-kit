"""Cluster J — CLI scaffolder round-trip.

``python -m langgraph_kit.cli new <agent_id>`` writes a Python file
derived from ``_AGENT_TEMPLATE`` and instructs the user to register it
via ``build_<fn_name>(checkpointer, store)``. These tests verify that:

- The scaffolder produces an importable Python module with no syntax
  errors or broken imports.
- The generated ``build_<fn_name>`` function produces a working graph
  that can be driven end-to-end by a scripted LLM.

This catches template drift — if the kit renames a helper function
(e.g. ``build_middleware_stack`` moves modules) but the template isn't
updated, every newly scaffolded agent breaks silently until someone
tries to import their new agent.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
    HumanMessage,
)

from langgraph_kit.cli import _generate_agent
from tests.e2e.helpers import answer, last_ai_message, scripted_llm

pytestmark = pytest.mark.e2e


def _import_module_from_path(name: str, path: Path) -> Any:
    """Import a freestanding Python file as a module."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_main_new_invokes_generator(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``python -m langgraph_kit.cli new <name>`` runs end-to-end.

    Exercises the argparse entry point (``main()``) that users hit
    from the shell. Guards against the subparser being renamed or the
    ``--output-dir`` flag regressing.
    """
    from langgraph_kit import cli as cli_mod

    argv = ["langgraph-kit", "new", "main-smoke", "--output-dir", str(tmp_path)]
    with patch.object(sys, "argv", argv):
        cli_mod.main()

    output = capsys.readouterr().out
    assert "Generated agent:" in output, (
        f"CLI ``new`` should print 'Generated agent:'; got {output!r}"
    )
    assert (tmp_path / "main_smoke.py").exists(), (
        "CLI ``new`` should write <fn_name>.py to the output dir"
    )


def test_cli_main_list_enumerates_templates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``python -m langgraph_kit.cli list`` runs and prints the template catalog."""
    from langgraph_kit import cli as cli_mod

    with patch.object(sys, "argv", ["langgraph-kit", "list"]):
        cli_mod.main()

    output = capsys.readouterr().out
    assert "Available templates" in output
    assert "default" in output


def test_cli_main_no_args_prints_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invoked with no subcommand, ``main()`` falls through to ``print_help``."""
    from langgraph_kit import cli as cli_mod

    with patch.object(sys, "argv", ["langgraph-kit"]):
        cli_mod.main()

    output = capsys.readouterr().out
    assert "usage:" in output.lower()


def test_cli_scaffolder_produces_an_importable_file(tmp_path: Path) -> None:
    """The scaffolder's output must be importable with no exceptions.

    Regression guard for template drift (kit-level rename of helper
    functions breaks every newly-scaffolded agent if the template isn't
    updated in lockstep).
    """
    out = _generate_agent("e2e-scaffolded", output_dir=tmp_path)
    assert out.exists()
    assert out.name == "e2e_scaffolded.py"

    module = _import_module_from_path("_e2e_scaffolded", out)
    assert hasattr(module, "build_e2e_scaffolded"), (
        "Scaffolded module should export ``build_<fn_name>`` per the template."
        f" Has: {[n for n in dir(module) if not n.startswith('_')]}"
    )


@pytest.mark.asyncio
async def test_scaffolded_build_function_returns_a_working_graph(
    tmp_path: Path,
    checkpointer: Any,
    e2e_store: Any,
) -> None:
    """End-to-end: scaffolded agent invokes cleanly through a scripted LLM.

    This is the contract the scaffolder promises: the user runs
    ``new <name>``, registers the returned ``(graph, dispatcher)``, and
    the agent just works. If any of the kit's internal APIs drift
    (tool registration, middleware stack, prompt composition), the
    scaffolded agent won't even reach the LLM and the smoke test dies.
    """
    out = _generate_agent("smoke-scaffolded", output_dir=tmp_path)
    module = _import_module_from_path("_smoke_scaffolded", out)

    scripted = scripted_llm([answer("scaffolded agent responded")])
    # The template imports ``build_llm`` directly from ``langgraph_kit.llm``
    # into its own module namespace, so the patch must target the
    # scaffolded module's local name, not the builder module the rest
    # of the e2e suite patches.
    with patch.object(module, "build_llm", return_value=scripted):
        graph, _dispatcher = module.build_smoke_scaffolded(
            checkpointer=checkpointer,
            store=e2e_store,
        )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="hello scaffold")]},
        config={"configurable": {"thread_id": "cli-smoke"}},  # pyright: ignore[reportArgumentType]
    )
    assert "scaffolded agent responded" in str(last_ai_message(result).content)
