"""Tools: registering callables with rich capability metadata.

What this shows
---------------
- Defining tool callables and wrapping them in :class:`ToolCapability`
- Setting risk levels (``READ_ONLY`` / ``MUTATING`` / ``DESTRUCTIVE``)
- Filtering the registry by ``max_risk`` so a worker only sees
  capabilities it's allowed to call
- Inspecting the resulting tool list

No LLM. The :class:`ToolCapability` model is what the deep agents bind
to their LLM at construction time and what bundled middleware reads to
gate HITL approval / output persistence.

How to run
----------
    uv run python -m examples.tools_register_capability

Expected output
---------------
    Registered 3 tools: ['list_files', 'rename_file', 'delete_branch']
    Filtered to max_risk=MUTATING: ['list_files', 'rename_file']
    Compiled 2 LangChain tool object(s):
      - list_files
      - rename_file
"""

from __future__ import annotations

from examples._lib import banner, line


def list_files(path: str = ".") -> str:
    """List entries at *path*. Read-only example tool."""
    return f"[stub] would list {path}"


def rename_file(src: str, dst: str) -> str:
    """Rename src -> dst. Mutating example tool."""
    return f"[stub] would rename {src} -> {dst}"


def delete_branch(name: str) -> str:
    """Delete a branch. Destructive example tool — gated by HITL."""
    return f"[stub] would delete branch {name}"


def main() -> None:
    banner("tools_register_capability")

    from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk
    from langgraph_kit.core.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(
        ToolCapability(
            id="list_files",
            name="list_files",
            description="List directory entries.",
            fn=list_files,
            risk=ToolRisk.READ_ONLY,
            tags=["filesystem"],
            prompt_guidance="Use this for read-only directory inspection.",
        )
    )
    registry.register(
        ToolCapability(
            id="rename_file",
            name="rename_file",
            description="Rename a file in place.",
            fn=rename_file,
            risk=ToolRisk.MUTATING,
            tags=["filesystem"],
            prompt_guidance="Confirm the destination path before calling.",
        )
    )
    registry.register(
        ToolCapability(
            id="delete_branch",
            name="delete_branch",
            description="Delete a git branch — irreversible.",
            fn=delete_branch,
            risk=ToolRisk.DESTRUCTIVE,
            tags=["git"],
            interrupt_before=True,  # Gates execution behind HITL approval
            prompt_guidance="Use only after confirming the branch is merged.",
        )
    )

    all_names = [cap.name for cap in registry.list_all()]
    line(f"Registered {len(all_names)} tools: {all_names}")

    safe_caps = registry.filter(max_risk=ToolRisk.MUTATING)
    line(f"Filtered to max_risk=MUTATING: {[c.name for c in safe_caps]}")

    callables = registry.compile_tools(max_risk=ToolRisk.MUTATING)
    line(f"Compiled {len(callables)} callable(s) for binding to an LLM:")
    for cap in safe_caps:
        line(f"  - {cap.name}  ({cap.risk.value})")


if __name__ == "__main__":
    main()
