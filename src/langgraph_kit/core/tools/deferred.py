"""Deferred tool loading — discover and activate tools on demand to reduce prompt bloat.

A deferred tool is registered with the agent *but not bound to its LLM*,
so it doesn't eat into the model's tool-call schema budget. The agent
discovers candidates via :func:`build_tool_search`, then invokes any of
them via :func:`build_call_deferred_tool` — a single dispatcher tool that
looks the capability up by id and runs it with the provided arguments.

This is a workaround for the fact that LangChain's tool-calling surface
is frozen at agent construction time: once ``create_agent(tools=[...])``
binds its list, the model can't see new tools mid-run. The dispatcher
keeps the active binding small (two tools: ``tool_search`` +
``call_deferred_tool``) while still exposing an arbitrarily large
catalog at runtime.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any

from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk

logger = logging.getLogger(__name__)

_CALL_TOOL_NAME = "call_deferred_tool"


class DeferredToolRegistry:
    """Registry of tools available for on-demand discovery but not bound to the LLM.

    Deferred tools are NOT included in the agent's initial tool surface.
    The agent discovers them via ``tool_search`` and invokes them via
    ``call_deferred_tool`` — both registered on the *active* ToolRegistry
    by :func:`register_search_tool`.
    """

    def __init__(self, *, allow_destructive: bool = False) -> None:
        super().__init__()
        self._tools: dict[str, ToolCapability] = {}
        self._allow_destructive = allow_destructive

    @property
    def allow_destructive(self) -> bool:
        """When False, ``call_deferred_tool`` refuses to invoke DESTRUCTIVE
        capabilities. Defaults to False so catalog authors can safely
        register risky tools without immediately exposing them through
        the auto-dispatch path."""
        return self._allow_destructive

    def __len__(self) -> int:
        return len(self._tools)

    def __bool__(self) -> bool:
        return bool(self._tools)

    def register(self, capability: ToolCapability) -> None:
        self._tools[capability.id] = capability

    def register_many(self, capabilities: list[ToolCapability]) -> None:
        for cap in capabilities:
            self.register(cap)

    def search(self, query: str, limit: int = 5) -> list[ToolCapability]:
        """Search deferred tools by keyword matching against name, description, and tags.

        Simple keyword matching — no vector search needed for a tool catalog.
        """
        query_lower = query.lower()
        scored: list[tuple[int, ToolCapability]] = []

        for cap in self._tools.values():
            score = 0
            if query_lower in cap.name.lower():
                score += 3
            if query_lower in cap.description.lower():
                score += 2
            for tag in cap.tags:
                if query_lower in tag.lower():
                    score += 1
            if score > 0:
                scored.append((score, cap))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [cap for _, cap in scored[:limit]]

    def get(self, tool_id: str) -> ToolCapability | None:
        return self._tools.get(tool_id)

    def list_all(self) -> list[ToolCapability]:
        return list(self._tools.values())

    def activate(self, tool_id: str) -> ToolCapability | None:
        """Pop a tool from the deferred registry and return it.

        Historically intended to move a tool to the active binding, but
        LangChain's tool list is frozen at graph construction — dynamic
        rebinding isn't supported. Invocation goes through
        ``call_deferred_tool`` instead, which leaves the tool in the
        registry so it can be called repeatedly. Kept for callers that
        want to remove a tool permanently.
        """
        return self._tools.pop(tool_id, None)


def _describe_signature(fn: Any) -> str:
    """Return a one-line parameter summary for an LLM to consume.

    Best-effort: some wrapped callables (Pydantic tools, MCP adapters)
    hide their real signature behind ``*args, **kwargs``; we fall back
    to a generic hint there.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return "(arguments unknown — inspect signature failed)"

    parts: list[str] = []
    for name, param in sig.parameters.items():
        if name.startswith("_") or name in {"self", "cls"}:
            continue
        # Render "name: Type" or "name: Type = default"
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            ann = "Any"
        else:
            ann = getattr(annotation, "__name__", None) or str(annotation)
        token = f"{name}: {ann}"
        if param.default is not inspect.Parameter.empty:
            token += f" = {param.default!r}"
        parts.append(token)
    return "(" + ", ".join(parts) + ")" if parts else "()"


def build_tool_search(deferred: DeferredToolRegistry) -> Any:
    """Create a ``tool_search`` tool that agents use to discover deferred capabilities.

    The returned callable is an async function suitable for passing to
    ``create_deep_agent(tools=[...])``. Its response tells the agent to
    use :func:`build_call_deferred_tool`'s output (``call_deferred_tool``)
    to actually invoke any discovered tool — the LLM cannot call deferred
    tools directly because they are not bound to its tool surface.
    """

    async def tool_search(query: str) -> str:
        """Search for additional tools and capabilities not loaded by default.

        Use this when you need a capability that isn't in your current tool set.
        The search checks tool names, descriptions, and tags.

        After finding a tool, invoke it via ``call_deferred_tool(tool_id, arguments)``
        where ``tool_id`` is the ``id`` field shown in the results and ``arguments``
        is a dict of keyword arguments matching the tool's signature.

        Args:
            query: Keywords describing the capability you need
        """
        results = deferred.search(query, limit=5)
        if not results:
            return f"No deferred tools found matching '{query}'."

        lines = [f"Found {len(results)} available tool(s):\n"]
        for cap in results:
            tags = f" [{', '.join(cap.tags)}]" if cap.tags else ""
            lines.append(f"- **{cap.name}** (id: `{cap.id}`){tags}")
            lines.append(f"  {cap.description}")
            lines.append(f"  Signature: `{cap.name}{_describe_signature(cap.fn)}`")
            if cap.prompt_guidance:
                lines.append(f"  Guidance: {cap.prompt_guidance}")
            lines.append("")
        lines.append(
            f'To invoke one of these, call `{_CALL_TOOL_NAME}(tool_id="<id>", arguments={{...}})` with a dict of keyword arguments matching the signature above.'
        )
        return "\n".join(lines)

    return tool_search


def build_call_deferred_tool(deferred: DeferredToolRegistry) -> Any:
    """Create the dispatcher tool that lets the agent invoke deferred tools.

    Deferred tools aren't part of the LLM's tool-call surface — binding
    is frozen at graph construction time. This dispatcher is bound to
    the LLM so the model can reach the deferred registry at runtime:
    ``call_deferred_tool(tool_id="...", arguments={...})``.

    Returns a callable whose name matches ``_CALL_TOOL_NAME`` so the
    guidance embedded in :func:`build_tool_search`'s output resolves.
    """

    async def call_deferred_tool(
        tool_id: str, arguments: dict[str, Any] | str | None = None
    ) -> str:
        """Invoke a deferred tool discovered via ``tool_search``.

        Args:
            tool_id: The exact ``id`` value returned by ``tool_search``
                (not the display name — ids are stable, names may collide).
            arguments: A dict of keyword arguments matching the tool's
                signature. Pass ``{}`` for tools with no parameters. A
                JSON-string payload is also accepted and parsed — some
                models (Qwen variants) stringify the arguments dict when
                emitting tool calls; rejecting that shape at the schema
                layer would trap the agent in a retry loop with the same
                malformed payload.
        """
        args: Any = arguments or {}
        cap = deferred.get(tool_id)
        if cap is None:
            available = ", ".join(c.id for c in deferred.list_all()[:10]) or "(none)"
            return (
                f"Error: deferred tool '{tool_id}' not found. "
                f"Use tool_search to discover tools. "
                f"First 10 available ids: {available}"
            )

        # Risk gate: block destructive deferred tools unless the registry
        # was explicitly built with allow_destructive=True. Discovery via
        # tool_search still surfaces the tool; dispatch refuses.
        if cap.risk == ToolRisk.DESTRUCTIVE and not deferred.allow_destructive:
            return (
                f"Error: '{tool_id}' is marked destructive and cannot be "
                f"invoked through call_deferred_tool. The operator must "
                f"opt in via DeferredToolRegistry(allow_destructive=True)."
            )

        if not isinstance(args, dict):
            # LLMs sometimes pass a JSON string instead of a dict; the
            # function signature admits both so LangChain's Pydantic
            # schema lets the value through and this branch can run.
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError as exc:
                    return (
                        f"Error: arguments for '{tool_id}' must be a dict or JSON "
                        f"object, got unparsable string: {exc}"
                    )
            else:
                return (
                    f"Error: arguments for '{tool_id}' must be a dict, "
                    f"got {type(args).__name__}"
                )
            if not isinstance(args, dict):
                return (
                    f"Error: arguments for '{tool_id}' must decode to a dict, "
                    f"got {type(args).__name__}"
                )

        try:
            result = cap.fn(**args)
            if inspect.isawaitable(result):
                result = await result
        except TypeError as exc:
            # Most common failure mode: wrong argument shape. Return the
            # error to the model so it can correct and retry.
            return f"Error calling '{tool_id}': {exc}"
        except Exception as exc:
            logger.exception("Deferred tool %r raised", tool_id)
            return f"Error calling '{tool_id}': {type(exc).__name__}: {exc}"

        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, default=str)
        except (TypeError, ValueError):
            return str(result)

    # Name the function so LangChain picks up a stable tool name.
    call_deferred_tool.__name__ = _CALL_TOOL_NAME
    return call_deferred_tool
