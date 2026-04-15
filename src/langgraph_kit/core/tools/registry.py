"""Tool registry that manages capabilities, filtering, compilation, and prompt fragment collection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

from .capability import ToolCapability, ToolRisk

_RISK_LEVELS: dict[ToolRisk, int] = {
    ToolRisk.READ_ONLY: 0,
    ToolRisk.MUTATING: 1,
    ToolRisk.DESTRUCTIVE: 2,
}


def _risk_level(risk: ToolRisk) -> int:
    return _RISK_LEVELS[risk]


class ToolRegistry:
    def __init__(self) -> None:
        super().__init__()
        self._tools: dict[str, ToolCapability] = {}

    def register(self, capability: ToolCapability) -> None:
        self._tools[capability.id] = capability

    def register_many(self, capabilities: Sequence[ToolCapability]) -> None:
        for cap in capabilities:
            self.register(cap)

    def get(self, tool_id: str) -> ToolCapability | None:
        return self._tools.get(tool_id)

    def remove(self, tool_id: str) -> None:
        self._tools.pop(tool_id, None)

    def list_all(self) -> list[ToolCapability]:
        return list(self._tools.values())

    def filter(
        self,
        *,
        profile: str | None = None,
        worker_type: str | None = None,
        tags: set[str] | None = None,
        max_risk: ToolRisk | None = None,
    ) -> list[ToolCapability]:
        result: list[ToolCapability] = []
        for cap in self._tools.values():
            if cap.profiles is not None and (
                profile is None or profile not in cap.profiles
            ):
                continue
            if cap.worker_types is not None and (
                worker_type is None or worker_type not in cap.worker_types
            ):
                continue
            if tags is not None and not tags.intersection(cap.tags):
                continue
            if max_risk is not None and _risk_level(cap.risk) > _risk_level(max_risk):
                continue
            result.append(cap)
        return result

    def compile_tools(
        self,
        *,
        profile: str | None = None,
        worker_type: str | None = None,
        max_risk: ToolRisk | None = None,
    ) -> list[Any]:
        """Filter then compile tools to callable list."""
        return [
            cap.fn
            for cap in self.filter(
                profile=profile, worker_type=worker_type, max_risk=max_risk
            )
        ]

    def collect_prompt_fragments(
        self,
        *,
        profile: str | None = None,
        worker_type: str | None = None,
    ) -> str:
        caps = self.filter(profile=profile, worker_type=worker_type)
        fragments: list[str] = []
        for cap in caps:
            if cap.prompt_guidance is not None:
                fragments.append(f"### {cap.name}\n{cap.prompt_guidance}")
        if not fragments:
            return ""
        return "## Tool Guidance\n\n" + "\n\n".join(fragments)

