"""Variant overlay mechanism — swap a section's text or a middleware prompt.

A ``PromptOverlay`` is a small data object describing what to change for
a single bench run. Two override flavors:

- **section_overrides** — map of ``PromptSection.id`` → replacement text.
  Applied via :meth:`SectionRegistry.remove` + :meth:`SectionRegistry.register`.
  The composer's content-hash cache keys correctly invalidate when the
  text changes (``sections.py:42-51``).

- **middleware_overrides** — map of fully-qualified module attribute
  (e.g. ``"langgraph_kit.core.memory.extraction._EXTRACTION_PROMPT"``)
  → replacement text. Applied via :func:`unittest.mock.patch.object`
  inside a context manager so the original is restored on exit.

Variant files on disk live under
``tests/prompt_bench/variants/<target>/<variant_name>.md``. The first
non-frontmatter paragraph is the prompt text; everything else is
human-facing notes ignored by the loader.
"""

from __future__ import annotations

import contextlib
import importlib
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from langgraph_kit.core.prompt_assembly.sections import SectionRegistry


class PromptOverlay(BaseModel):
    """Describes a set of prompt overrides for a single bench run."""

    model_config = ConfigDict(extra="forbid")

    name: str
    section_overrides: dict[str, str] = Field(default_factory=dict)
    middleware_overrides: dict[str, str] = Field(default_factory=dict)


def load_variant(path: Path, name: str | None = None) -> str:
    """Load a variant prompt from a markdown file.

    Strips a leading YAML frontmatter block (delimited by ``---`` lines)
    if present. The remainder is returned verbatim — that's the prompt
    text the overlay will inject.
    """
    text = path.read_text(encoding="utf-8")
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            text = text[end + 5 :]
    return text.strip()


def apply_section_overlay(
    registry: SectionRegistry, overlay: PromptOverlay
) -> None:
    """Mutate *registry* in-place to swap any section IDs named in *overlay*.

    The replacement section preserves the original's stability,
    priority, and condition. Caller is responsible for snapshotting the
    registry beforehand if they need to restore it.
    """
    from langgraph_kit.core.prompt_assembly.sections import PromptSection

    for section_id, new_content in overlay.section_overrides.items():
        original = registry.get(section_id)
        if original is None:
            msg = f"Cannot overlay unknown section id: {section_id!r}"
            raise KeyError(msg)
        registry.remove(section_id)
        registry.register(
            PromptSection(
                id=original.id,
                content=new_content,
                stability=original.stability,
                priority=original.priority,
                condition=original.condition,
            )
        )


@contextlib.contextmanager
def patch_middleware_prompts(overlay: PromptOverlay) -> Iterator[None]:
    """Context manager that monkey-patches module-level prompt constants.

    Each key in ``overlay.middleware_overrides`` is a fully-qualified
    attribute path of the form ``module.path:ATTRIBUTE_NAME`` *or*
    ``module.path.ATTRIBUTE_NAME``. The attribute is patched to the
    overlay value for the duration of the ``with`` block and restored
    on exit.
    """
    with contextlib.ExitStack() as stack:
        for path_str, new_value in overlay.middleware_overrides.items():
            module_path, attr_name = _split_attr_path(path_str)
            module = importlib.import_module(module_path)
            if not hasattr(module, attr_name):
                msg = (
                    f"Module {module_path!r} has no attribute {attr_name!r} "
                    f"(can't patch for overlay {overlay.name!r})"
                )
                raise AttributeError(msg)
            stack.enter_context(patch.object(module, attr_name, new_value))
        yield


def _split_attr_path(path_str: str) -> tuple[str, str]:
    """Split ``module.path:ATTR`` or ``module.path.ATTR`` into (module, attr)."""
    if ":" in path_str:
        module_path, attr_name = path_str.split(":", 1)
        return module_path, attr_name
    if "." not in path_str:
        msg = f"Invalid attribute path {path_str!r} — need module.path.ATTR"
        raise ValueError(msg)
    module_path, attr_name = path_str.rsplit(".", 1)
    return module_path, attr_name


def discover_variants(root: Path, target: str) -> dict[str, Path]:
    """Return ``{variant_name: variant_path}`` for the named target."""
    base = root / "variants" / target.replace(".", "_")
    if not base.is_dir():
        base = root / "variants" / target.split(".", 1)[0]
    if not base.is_dir():
        return {}
    return {p.stem: p for p in sorted(base.glob("*.md"))}


def overlay_from_variant_file(
    name: str,
    section_id: str | None = None,
    middleware_attr: str | None = None,
    *,
    text: str,
) -> PromptOverlay:
    """Build a ``PromptOverlay`` for a single section or middleware constant.

    Exactly one of ``section_id`` or ``middleware_attr`` must be given.
    """
    if (section_id is None) == (middleware_attr is None):
        msg = "Provide exactly one of section_id or middleware_attr"
        raise ValueError(msg)
    if section_id is not None:
        return PromptOverlay(
            name=name,
            section_overrides={section_id: text},
        )
    return PromptOverlay(
        name=name,
        middleware_overrides={middleware_attr or "": text},
    )


def snapshot_section(registry: SectionRegistry, section_id: str) -> dict[str, Any]:
    """Capture a section's current content + metadata for later restoration."""
    section = registry.get(section_id)
    if section is None:
        msg = f"No section registered under id {section_id!r}"
        raise KeyError(msg)
    return {
        "id": section.id,
        "content": section.content,
        "stability": section.stability,
        "priority": section.priority,
        "condition": section.condition,
    }


def restore_section(registry: SectionRegistry, snapshot: dict[str, Any]) -> None:
    """Restore a section from a :func:`snapshot_section` snapshot."""
    from langgraph_kit.core.prompt_assembly.sections import PromptSection

    registry.remove(snapshot["id"])
    registry.register(
        PromptSection(
            id=snapshot["id"],
            content=snapshot["content"],
            stability=snapshot["stability"],
            priority=snapshot["priority"],
            condition=snapshot["condition"],
        )
    )
