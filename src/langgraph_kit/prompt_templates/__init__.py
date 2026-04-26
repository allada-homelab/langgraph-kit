"""Curated, reusable :class:`PromptSection` library for langgraph-kit consumers.

Each shipped section is a versioned, frozen ``PromptSection`` with a
clear id and stability label — drop them into a
:class:`SectionRegistry` to get a sensible baseline prompt without
hand-rolling the same boilerplate per agent.

Usage::

    from langgraph_kit.core.prompt_assembly.sections import SectionRegistry
    from langgraph_kit.prompt_templates import (
        core_identity,
        operate_carefully,
        be_concise,
    )

    registry = SectionRegistry()
    registry.register_many([core_identity, operate_carefully, be_concise])

To customize a shipped section without forking the kit:

    from langgraph_kit.prompt_templates import core_identity
    custom = core_identity.model_copy(update={
        "content": "You are an internal-tools agent for ...",
        "version": "custom-1",
    })
    registry.register(custom)  # replaces by id

Override-by-id replaces the entire section. Per the original issue
spec, partial-merge overrides (changing only ``content`` while
keeping the kit's ``priority``/``stability``) aren't supported — use
``model_copy`` to derive your own and replace whole.

Use :func:`diff_section` to introspect what your override actually
changed vs the shipped baseline; useful as a startup-log entry on
deploys that customize prompts so the actual prompt in use is
visible without reading source.

Scope (issue #43 v1):

- Eight shipped sections covering the common prompt families
  (identity, operating discipline, conciseness, memory awareness,
  tool guidance, error handling, output format, completion
  signaling). Each has ``version="1"``.
- ``diff_section`` helper for visualizing customizations.
- Sections are *importable*. Auto-registration on a fresh
  ``SectionRegistry`` and ``AgentConfig.prompt_overrides`` lifecycle
  integration are intentionally deferred to follow-ups so this PR
  doesn't change graph-build semantics for existing callers.
- Existing scattered prompts (``BASIC_SYSTEM_PROMPT``,
  ``_EXTRACTION_PROMPT`` etc.) are left in place; consumers who
  want the library can adopt it incrementally. Migration of those
  is its own follow-up.
"""

from __future__ import annotations

from langgraph_kit.prompt_templates.core import (
    be_concise,
    completion_signal,
    core_identity,
    operate_carefully,
)
from langgraph_kit.prompt_templates.diff import diff_section
from langgraph_kit.prompt_templates.memory import (
    memory_awareness,
)
from langgraph_kit.prompt_templates.safety import (
    error_handling,
    output_format_natural,
    output_format_structured,
)
from langgraph_kit.prompt_templates.tools import tool_use_discipline

__all__ = [
    "be_concise",
    "completion_signal",
    "core_identity",
    "diff_section",
    "error_handling",
    "memory_awareness",
    "operate_carefully",
    "output_format_natural",
    "output_format_structured",
    "tool_use_discipline",
]
