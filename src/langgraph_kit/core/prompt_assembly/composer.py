"""Prompt composer that assembles layered prompts from sections and context providers.

Integrates PromptCache for stability-aware caching and orders sections
STABLE-first to maximize Anthropic prompt cache hits.
"""

from __future__ import annotations

from typing import Any

from langgraph_kit.core.prompt_assembly.cache import PromptCache
from langgraph_kit.core.prompt_assembly.context_providers import (
    ContextProvider,
)
from langgraph_kit.core.prompt_assembly.sections import (
    PromptSection,
    SectionRegistry,
    SectionStability,
)

_STABILITY_ORDER = {
    SectionStability.STABLE: 0,
    SectionStability.CONDITIONAL: 1,
    SectionStability.VOLATILE: 2,
}


def _stable_first_sort(sections: list[PromptSection]) -> list[PromptSection]:
    """Sort sections: STABLE first, then CONDITIONAL, then VOLATILE.

    Within each stability tier, sections are sorted by priority descending.
    This ordering places cacheable content at the front of the prompt,
    maximizing Anthropic prompt cache hits on stable prefixes.
    """
    return sorted(sections, key=lambda s: (_STABILITY_ORDER[s.stability], -s.priority))


class PromptComposer:
    """Assembles a final prompt from registered sections and dynamic context providers.

    Uses ``PromptCache`` to avoid recomputing stable sections and orders
    sections so that stable content appears first in the prompt.
    """

    def __init__(
        self,
        registry: SectionRegistry,
        providers: list[ContextProvider] | None = None,
        *,
        cache: PromptCache | None = None,
    ) -> None:
        super().__init__()
        self._registry = registry
        self._providers: list[ContextProvider] = providers or []
        self._cache = cache or PromptCache()

    async def compose(
        self,
        conditions: set[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Compose a full prompt from active sections and provider outputs."""
        sections = _stable_first_sort(self._registry.get_active(conditions))
        prompt, _changes = self._cache.compose_with_cache(sections)

        # Provider outputs are always volatile — appended after cached sections
        ctx = context or {}
        provider_parts: list[str] = []
        for provider in self._providers:
            output = await provider.provide(ctx)
            if output:
                provider_parts.append(output)

        if provider_parts:
            prompt += "\n\n" + "\n\n".join(provider_parts)
        return prompt

    def compose_sections_only(self, conditions: set[str] | None = None) -> str:
        """Compose a prompt from active sections only, without providers."""
        sections = _stable_first_sort(self._registry.get_active(conditions))
        prompt, _changes = self._cache.compose_with_cache(sections)
        return prompt

    def get_active_section_ids(self, conditions: set[str] | None = None) -> list[str]:
        return [s.id for s in self._registry.get_active(conditions)]
