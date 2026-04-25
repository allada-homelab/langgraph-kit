"""Prompt assembly: layered sections + context providers, cache-aware.

What this shows
---------------
- Building a :class:`SectionRegistry` with stable, volatile, and
  conditional sections at different priorities
- :class:`PromptComposer` orders stable sections first (cache-friendly)
  and includes conditional sections only when their key is in the
  active conditions set
- ``compose_sections_only`` is fully synchronous — useful for unit
  testing prompt structure without spinning up a graph

No LLM. The same composer powers the deep agents' system-prompt
construction.

How to run
----------
    uv run python -m examples.prompt_assembly_sections

Expected output
---------------
    Active section ids (no conditions): ['identity', 'core_rules']
    Active section ids (conditions={'memory'}): ['identity', 'core_rules', 'memory_briefing']
    Composed prompt (memory active):
    -------- BEGIN --------
    You are a careful assistant.
    ...
    -------- END --------
"""

from __future__ import annotations

import asyncio

from examples._lib import banner, line


async def main() -> None:
    banner("prompt_assembly_sections")

    from langgraph_kit.core.prompt_assembly.composer import PromptComposer
    from langgraph_kit.core.prompt_assembly.sections import (
        PromptSection,
        SectionRegistry,
        SectionStability,
    )

    registry = SectionRegistry()
    registry.register_many(
        [
            PromptSection(
                id="identity",
                content="You are a careful assistant.",
                stability=SectionStability.STABLE,
                priority=100,
            ),
            PromptSection(
                id="core_rules",
                content="Always reason step by step before acting.",
                stability=SectionStability.STABLE,
                priority=90,
            ),
            PromptSection(
                id="memory_briefing",
                content="The user prefers terse responses.",
                stability=SectionStability.CONDITIONAL,
                condition="memory",
                priority=50,
            ),
            PromptSection(
                id="skills_briefing",
                content="When asked to ship code, drive the change end-to-end.",
                stability=SectionStability.CONDITIONAL,
                condition="skills",
                priority=40,
            ),
        ]
    )

    composer = PromptComposer(registry)

    line(f"Active section ids (no conditions): {composer.get_active_section_ids()}")
    line(
        "Active section ids (conditions={'memory'}): "
        f"{composer.get_active_section_ids(conditions={'memory'})}"
    )

    composed = composer.compose_sections_only(conditions={"memory"})
    line("Composed prompt (memory active):")
    line("-------- BEGIN --------")
    line(composed)
    line("-------- END --------")


if __name__ == "__main__":
    asyncio.run(main())
