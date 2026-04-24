"""Built-in command implementations.

Factory functions that produce command handlers for common harness operations:
/help, /memory, /context, /compact, /tools, /skills, /status.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph_kit.core.commands.dispatch import (
    CommandHandler,
    CommandResult,
)
from langgraph_kit.core.context_management.pressure_middleware import microcompact

# Compaction thresholds — delegates to the shared ``microcompact`` helper
# so these stay aligned with PressureMiddleware's in-loop microcompaction.
_COMPACT_RECENT_WINDOW = 10  # messages to skip (keep recent context intact)
_COMPACT_CONTENT_THRESHOLD = 2000  # chars — truncate tool outputs larger than this
_COMPACT_PREVIEW_CHARS = 200  # chars to keep from truncated content

if TYPE_CHECKING:
    from langgraph_kit.core.commands.dispatch import CommandDispatcher
    from langgraph_kit.core.context_management.pressure import (
        PressureMonitor,
    )
    from langgraph_kit.core.memory.persistent import (
        PersistentMemoryManager,
    )
    from langgraph_kit.core.skills.registry import SkillRegistry
    from langgraph_kit.core.tools.registry import ToolRegistry


def build_help_command(dispatcher: CommandDispatcher) -> CommandHandler:
    """Create a /help handler that lists all registered commands."""

    async def handle_help(_args: str, _ctx: dict[str, Any]) -> CommandResult:
        commands = dispatcher.list_commands()
        if not commands:
            return CommandResult(output="No commands registered.")

        lines = ["**Available Commands:**\n"]
        for cmd in commands:
            usage = f" {cmd.usage}" if cmd.usage else ""
            desc = f" — {cmd.description}" if cmd.description else ""
            lines.append(f"- `/{cmd.name}{usage}`{desc}")
        return CommandResult(output="\n".join(lines))

    return handle_help


def build_memory_command(memory_mgr: PersistentMemoryManager) -> CommandHandler:
    """Create a /memory handler that inspects current memory state."""

    from langgraph_kit.core.memory.models import MemoryScope

    async def handle_memory(args: str, _ctx: dict[str, Any]) -> CommandResult:
        scope_str = args.strip() if args.strip() else "user"
        try:
            scope = MemoryScope(scope_str)
        except ValueError:
            valid = [s.value for s in MemoryScope]
            return CommandResult(output=f"Invalid scope: '{scope_str}'. Valid: {valid}")

        records = await memory_mgr.list_by_scope(scope, limit=20)
        if not records:
            return CommandResult(
                output=f"No memories in scope '{scope.value}'.",
                metadata={"scope": scope.value, "count": 0},
            )

        lines = [f"**Memories ({scope.value}) — {len(records)} record(s):**\n"]
        for r in records:
            lines.append(f"- [{r.type.value}] **{r.title}**: {r.summary}")
        return CommandResult(
            output="\n".join(lines),
            metadata={"scope": scope.value, "count": len(records)},
        )

    return handle_memory


def build_compact_command(pressure_monitor: PressureMonitor) -> CommandHandler:
    """Create a /compact handler that truncates large tool outputs to free context space."""

    async def handle_compact(_args: str, ctx: dict[str, Any]) -> CommandResult:
        messages: list[Any] = ctx.get("messages", [])
        if not messages:
            return CommandResult(output="Nothing to compact — conversation is empty.")

        signals = pressure_monitor.assess(messages)
        before_tokens = signals.estimated_tokens

        # Apply microcompaction: truncate large tool outputs outside the recent window
        compacted = _microcompact(messages)
        if compacted is None:
            return CommandResult(
                output=(
                    f"**No compaction needed.** Context is at"
                    f" {signals.pressure_pct:.0%} ({before_tokens:,} tokens)."
                ),
                metadata={
                    "before_tokens": before_tokens,
                    "after_tokens": before_tokens,
                },
            )

        after_signals = pressure_monitor.assess(compacted)
        saved = before_tokens - after_signals.estimated_tokens

        lines = [
            "**Compaction complete.**\n",
            f"- Before: {before_tokens:,} tokens ({signals.pressure_pct:.0%})",
            f"- After: {after_signals.estimated_tokens:,} tokens"
            f" ({after_signals.pressure_pct:.0%})",
            f"- Freed: ~{saved:,} tokens",
        ]
        return CommandResult(
            output="\n".join(lines),
            metadata={
                "compacted_messages": compacted,
                "before_tokens": before_tokens,
                "after_tokens": after_signals.estimated_tokens,
            },
        )

    return handle_compact


def _microcompact(messages: list[Any]) -> list[Any] | None:
    """Thin alias to the shared ``microcompact`` helper.

    Kept for internal callers that imported the private name; the
    canonical implementation now lives in
    ``core.context_management.pressure_middleware``.
    """
    return microcompact(
        messages,
        recent_window=_COMPACT_RECENT_WINDOW,
        content_threshold=_COMPACT_CONTENT_THRESHOLD,
        preview_chars=_COMPACT_PREVIEW_CHARS,
    )


def build_context_command(pressure_monitor: PressureMonitor) -> CommandHandler:
    """Create a /context handler that shows context window status."""

    async def handle_context(_args: str, ctx: dict[str, Any]) -> CommandResult:
        messages: list[Any] = ctx.get("messages", [])
        signals = pressure_monitor.assess(messages)
        pressure_pct = f"{signals.pressure_pct:.0%}"
        lines = [
            "**Context Window Status:**\n",
            f"- Estimated tokens: {signals.estimated_tokens:,}",
            f"- Window limit: {signals.window_limit:,}",
            f"- Pressure: {pressure_pct}",
            f"- Large tool outputs: {signals.large_tool_outputs}",
            f"- Compaction failures: {signals.compaction_failures}",
        ]
        strategy = pressure_monitor.choose_mitigation(signals)
        if strategy.value != "none":
            lines.append(f"- Recommended action: {strategy.value}")
        return CommandResult(
            output="\n".join(lines),
            metadata={
                "estimated_tokens": signals.estimated_tokens,
                "window_limit": signals.window_limit,
                "pressure_pct": signals.pressure_pct,
                "mitigation": strategy.value,
            },
        )

    return handle_context


def build_tools_command(tool_registry: ToolRegistry) -> CommandHandler:
    """Create a /tools handler that lists registered tools with risk levels."""

    async def handle_tools(args: str, _ctx: dict[str, Any]) -> CommandResult:
        caps = tool_registry.list_all()
        if not caps:
            return CommandResult(output="No tools registered.")

        # Optional tag filter
        tag_filter = args.strip().lower() if args.strip() else None
        if tag_filter:
            caps = [c for c in caps if tag_filter in [t.lower() for t in c.tags]]

        lines = [f"**Registered Tools ({len(caps)}):**\n"]
        for cap in sorted(caps, key=lambda c: c.name):
            risk_badge = {
                "read_only": "RO",
                "mutating": "MUT",
                "destructive": "DEST",
            }.get(cap.risk.value, cap.risk.value)
            tags = ", ".join(cap.tags) if cap.tags else ""
            lines.append(f"- `{cap.name}` [{risk_badge}] {tags}")
        return CommandResult(
            output="\n".join(lines),
            metadata={"count": len(caps)},
        )

    return handle_tools


def build_skills_command(skill_registry: SkillRegistry) -> CommandHandler:
    """Create a /skills handler that lists available skills."""

    async def handle_skills(_args: str, _ctx: dict[str, Any]) -> CommandResult:
        skills = skill_registry.list_all()
        if not skills:
            return CommandResult(output="No skills loaded.")

        lines = [f"**Available Skills ({len(skills)}):**\n"]
        for skill in skills:
            tags = f" ({', '.join(skill.tags)})" if skill.tags else ""
            lines.append(f"- **{skill.name}**{tags}: {skill.description}")
        return CommandResult(
            output="\n".join(lines),
            metadata={"count": len(skills)},
        )

    return handle_skills


def build_status_command(
    pressure_monitor: PressureMonitor,
    memory_mgr: PersistentMemoryManager,
) -> CommandHandler:
    """Create a /status handler that shows a combined dashboard."""

    from langgraph_kit.core.memory.models import MemoryScope

    async def handle_status(_args: str, ctx: dict[str, Any]) -> CommandResult:
        messages: list[Any] = ctx.get("messages", [])
        signals = pressure_monitor.assess(messages)
        strategy = pressure_monitor.choose_mitigation(signals)

        # Per-scope memory counts — earlier versions only counted the
        # ``user`` scope, which undercounted assistants and project teams
        # that stored most of their memory elsewhere.
        scope_counts: list[tuple[str, int]] = []
        total_memories = 0
        for scope in MemoryScope:
            records = await memory_mgr.list_by_scope(scope, limit=100)
            scope_counts.append((scope.value, len(records)))
            total_memories += len(records)

        memory_line = ", ".join(
            f"{name}={count}" for name, count in scope_counts if count
        ) or "none"

        lines = [
            "**Agent Status Dashboard:**\n",
            f"**Context:** {signals.estimated_tokens:,} tokens"
            f" ({signals.pressure_pct:.0%} of {signals.window_limit:,})",
            f"**Messages:** {len(messages)}",
            f"**Memories:** {total_memories} total ({memory_line})",
            f"**Large outputs:** {signals.large_tool_outputs}",
        ]
        if strategy.value != "none":
            lines.append(f"**Action needed:** {strategy.value}")
        return CommandResult(output="\n".join(lines))

    return handle_status
