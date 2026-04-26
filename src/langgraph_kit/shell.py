"""Interactive REPL for a registered agent.

Run with::

    python -m langgraph_kit.cli shell <agent-id>

Loads ``agent-id`` from the in-memory registry, builds it against an
in-memory checkpointer + store, and enters a read-eval-print loop:
type a prompt, the agent runs, the final assistant message prints.
``/exit`` (or Ctrl-D / Ctrl-C) ends the session.

By default the shell calls
:func:`langgraph_kit.graphs.register_all` to expose the kit's built-in
agents (``echo-agent``, ``basic-deep-agent``, ``reference-deep-agent``,
``coding-agent``). Pass ``--module my_app.agents`` to import a
user-supplied module first — that module is responsible for calling
:func:`langgraph_kit.registry.register` so the agent shows up.

Scope (issue #37 v1):

- Single-agent session, in-process invocation (no FastAPI hop).
- Plain stdin input + stdout output. No ``prompt_toolkit``, no token
  streaming, no syntax-highlighted output.
- ``--thread-id`` accepted; persistence is in-memory only so threads
  are session-scoped, not cross-session.
- ``/exit`` is the only built-in slash command; agent-defined slash
  commands route through the agent's normal dispatcher (they go in
  as user input and the agent's command middleware handles them).

Deferred to follow-ups:

- Token streaming via ``graph.astream_events`` rendered as the model
  emits.
- ``prompt_toolkit`` line editing + tab completion.
- HITL interrupt prompts.
- Transcript-write mode (``--transcript out.md``).
- Multi-agent ``/switch`` and richer slash-command tooling.
"""

from __future__ import annotations

import importlib
import logging
import sys
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


_EXIT_COMMANDS = {"/exit", "/quit", "/q"}
"""Slash commands that end the session — same set GitHub Copilot CLI uses."""

_PROMPT = "you > "
_ASSISTANT_PREFIX = "agent > "


def _build_in_memory_deps() -> tuple[Any, Any]:
    """Return ``(checkpointer, store)`` using LangGraph's in-memory impls.

    Lazy-imported so the kit imports cleanly without LangGraph
    available at the top level (matches ``contrib/`` modules' pattern).
    """
    from langgraph.checkpoint.memory import (  # pyright: ignore[reportMissingImports]
        InMemorySaver,
    )
    from langgraph.store.memory import (  # pyright: ignore[reportMissingImports]
        InMemoryStore,
    )

    return InMemorySaver(), InMemoryStore()


def _ensure_agent_registered(
    agent_id: str,
    user_module: str | None,
) -> None:
    """Resolve *agent_id* in the registry, registering kit defaults if needed.

    If *user_module* is set, import it first (the module is expected to
    call :func:`langgraph_kit.registry.register`). Otherwise call
    :func:`langgraph_kit.graphs.register_all` to expose the built-ins.
    """
    from langgraph_kit import registry

    if user_module:
        # Importing the module triggers whatever ``register(...)`` calls
        # it makes at module level. ``importlib.import_module`` raises
        # ImportError on failure with a useful message.
        importlib.import_module(user_module)

    if agent_id not in registry.get_all():
        # Built-ins haven't been registered yet — do it now. Idempotent
        # (register() upserts) so this is safe even if a user module
        # already registered some agents.
        from langgraph_kit.graphs import register_all

        checkpointer, store = _build_in_memory_deps()
        register_all(checkpointer, store)


def _format_assistant_output(result: Any) -> str:
    """Pull the final assistant text out of a graph result.

    Matches the shape ``graph.ainvoke`` returns for chat-style agents
    in this kit (``{"messages": [...]}``). Falls back to ``str(result)``
    for non-chat graphs so the shell stays useful for arbitrary
    callables.
    """
    if isinstance(result, dict) and "messages" in result:
        msgs = result["messages"] or []
        if not msgs:
            return "(no message)"
        last = msgs[-1]
        content = getattr(last, "content", None)
        if content is None:
            return str(last)
        if isinstance(content, str):
            return content
        # Multi-part content (list of dicts with ``text`` keys) — flatten.
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    parts.append(str(part["text"]))
                else:
                    parts.append(str(part))
            return "".join(parts)
        return str(content)
    return str(result)


async def _invoke_once(graph: Any, user_input: str, config: dict[str, Any]) -> str:
    """Send one user turn through *graph* and return the rendered reply."""
    from langchain_core.messages import (  # pyright: ignore[reportMissingModuleSource]
        HumanMessage,
    )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )
    return _format_assistant_output(result)


async def run_shell(
    agent_id: str,
    *,
    user_module: str | None = None,
    thread_id: str | None = None,
    user_id: str = "shell-user",
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
) -> int:
    """Entry point for the shell. Returns a process exit code.

    ``input_fn`` and ``output_fn`` are injected so tests can drive
    the loop without touching real stdin/stdout. They default to
    ``input`` / ``print`` for the CLI path. The output function
    receives one fully-formed line at a time (newline included is
    the caller's choice — defaults match ``print``).
    """
    from langgraph_kit import registry

    read_input = input_fn or input
    write_output: Callable[[str], None] = output_fn or print  # type: ignore[assignment]

    try:
        _ensure_agent_registered(agent_id, user_module)
    except ImportError as exc:
        sys.stderr.write(f"Couldn't import {user_module!r}: {exc}\n")
        return 2

    try:
        graph = registry.get(agent_id)
    except KeyError:
        sys.stderr.write(
            f"Agent {agent_id!r} not registered. Available: "
            f"{sorted(registry.get_all())}\n"
        )
        return 2

    thread_id = thread_id or f"shell-{uuid.uuid4().hex[:8]}"
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id, "user_id": user_id}
    }

    write_output(f"langgraph-kit shell — agent={agent_id!r} thread={thread_id!r}")
    write_output("Type your message; '/exit' (or Ctrl-D / Ctrl-C) to quit.")
    write_output("Built-in slash commands: /exit, /info.")

    while True:
        try:
            user_input = read_input(_PROMPT)
        except (EOFError, KeyboardInterrupt):
            write_output("")  # newline so the prompt doesn't bleed into shell PS1
            return 0
        if not user_input.strip():
            continue
        if user_input.strip().lower() in _EXIT_COMMANDS:
            return 0
        if user_input.strip().lower() == "/info":
            write_output(
                f"  agent_id  = {agent_id}\n"
                f"  thread_id = {thread_id}\n"
                f"  user_id   = {user_id}\n"
                f"  module    = {user_module or '(built-in)'}"
            )
            continue
        try:
            reply = await _invoke_once(graph, user_input, config)
        except Exception:
            logger.exception("Agent invocation failed")
            write_output(f"{_ASSISTANT_PREFIX}(error — see logs)")
            continue
        write_output(f"{_ASSISTANT_PREFIX}{reply}")


__all__ = ["run_shell"]
