"""Tools + HITL: risk levels meet AutoInterruptMiddleware.

What this shows
---------------
- A :class:`ToolRegistry` populated with one read-only tool and one
  destructive tool gated by ``interrupt_before=True``
- :class:`AutoInterruptMiddleware` consuming the registry to decide
  which tool calls to pause for HITL approval
- Inspecting which tools the middleware has flagged for interrupt

This is the static side of the HITL story; ``hitl_approval_flow.py``
shows the dynamic interrupt → resume cycle. Together they cover the
two halves: declaring which tools need approval, and what happens at
runtime when one is called.

How to run
----------
    uv run python -m examples.tools_risk_levels_and_hitl

Expected output
---------------
    Tool registry has 2 tool(s):
      - list_files     risk=read_only    interrupt_before=False
      - delete_branch  risk=destructive  interrupt_before=True
    AutoInterruptMiddleware would gate 1 tool name(s):
      - delete_branch
"""

from __future__ import annotations

from examples._lib import banner, line


def list_files(path: str = ".") -> str:
    return f"[stub] would list {path}"


def delete_branch(name: str) -> str:
    return f"[stub] would delete branch {name}"


def main() -> None:
    banner("tools_risk_levels_and_hitl")

    from langgraph_kit.core.hitl.auto_interrupt import AutoInterruptMiddleware
    from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk
    from langgraph_kit.core.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(
        ToolCapability(
            id="list_files",
            name="list_files",
            description="Read-only directory listing.",
            fn=list_files,
            risk=ToolRisk.READ_ONLY,
        )
    )
    registry.register(
        ToolCapability(
            id="delete_branch",
            name="delete_branch",
            description="Destructive — irreversible.",
            fn=delete_branch,
            risk=ToolRisk.DESTRUCTIVE,
            interrupt_before=True,  # Gates the tool behind HITL approval
            prompt_guidance="Use only after confirming the branch is merged.",
        )
    )

    line(f"Tool registry has {len(registry.list_all())} tool(s):")
    for cap in registry.list_all():
        line(
            f"  - {cap.name:<14} risk={cap.risk.value:<11} "
            f"interrupt_before={cap.interrupt_before}"
        )

    # AutoInterruptMiddleware reads ``interrupt_before`` off each
    # ToolCapability at call time. Build it against the same registry
    # the deep agent would use, then ask "would this tool name be
    # gated?" via the registry's tool-name lookup.
    middleware = AutoInterruptMiddleware(tool_registry=registry)
    _ = middleware  # built but unused — its hook fires inside a graph run

    gated_names = [cap.name for cap in registry.list_all() if cap.interrupt_before]
    line(f"\nAutoInterruptMiddleware would gate {len(gated_names)} tool name(s):")
    for name in gated_names:
        line(f"  - {name}")


if __name__ == "__main__":
    main()
