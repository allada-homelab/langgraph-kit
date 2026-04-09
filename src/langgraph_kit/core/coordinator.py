"""Coordinator mode — supervisor profile emphasizing orchestration over direct execution."""

from __future__ import annotations

from typing import Any

from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionStability,
)
from langgraph_kit.core.tools.capability import ToolRisk
from langgraph_kit.core.tools.registry import ToolRegistry

# Coordinator-specific prompt sections
COORDINATOR_SECTIONS = [
    PromptSection(
        id="coordinator_role",
        content=(
            "# Coordinator Mode\n"
            "You are operating as a COORDINATOR. Your job is to:\n"
            "1. Understand what the user needs\n"
            "2. Plan the work breakdown\n"
            "3. Delegate concrete tasks to specialized workers\n"
            "4. Synthesize worker results into coherent answers\n"
            "5. Decide what to do next based on findings\n\n"
            "You should NOT do direct implementation work yourself. "
            "Delegate implementation, research, and verification to workers."
        ),
        stability=SectionStability.STABLE,
        priority=95,
        condition="coordinator",
    ),
    PromptSection(
        id="coordinator_delegation",
        content=(
            "# Delegation Rules\n"
            "- Brief workers like a colleague who hasn't seen the conversation\n"
            "- Include: the goal, relevant context, what success looks like\n"
            "- Never delegate understanding — read worker results yourself first\n"
            "- Use parallel workers for independent tasks\n"
            "- After receiving results, synthesize before delegating follow-up"
        ),
        stability=SectionStability.STABLE,
        priority=90,
        condition="coordinator",
    ),
    PromptSection(
        id="coordinator_synthesis",
        content=(
            "# Synthesis Discipline\n"
            "When workers return results:\n"
            "1. Read and understand the findings yourself\n"
            "2. Identify gaps, contradictions, or next steps\n"
            "3. Synthesize into a clear answer for the user\n"
            "4. Only then decide whether more work is needed\n\n"
            "Do NOT relay worker output verbatim. Add value through integration."
        ),
        stability=SectionStability.STABLE,
        priority=85,
        condition="coordinator",
    ),
]


class CoordinatorMode:
    """Configures an agent for coordinator/supervisor operation.

    Provides:
    - Narrowed tool surface (read-only + delegation, no mutating tools)
    - Coordinator-specific prompt sections
    - Worker definitions filtered for coordinator use
    """

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._registry = tool_registry

    def get_coordinator_tools(self) -> list[Any]:
        """Get tools filtered for coordinator mode — read-only only."""
        return self._registry.compile_tools_filtered(max_risk=ToolRisk.READ_ONLY)

    def get_coordinator_sections(self) -> list[PromptSection]:
        """Get prompt sections for coordinator mode."""
        return list(COORDINATOR_SECTIONS)

    @staticmethod
    def get_conditions() -> set[str]:
        """Get the condition set needed to activate coordinator prompt sections."""
        return {"coordinator", "orchestration"}
