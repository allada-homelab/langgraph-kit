"""System-environment context provider for the reference deep agent.

Mirrors :class:`GitContextProvider`'s shape but targets the no-domain
case: a minimal exemplar of the ``extra_providers=`` extension point so
new domain agents have an in-tree pattern that doesn't assume git.

Output is deterministic given a fixed clock (callers can inject
``clock=`` for tests). Calling ``provide`` twice with the same clock
returns the same string, so prompt-cache reuse is preserved when the
provider is invoked multiple times within a single turn.
"""

from __future__ import annotations

import os
import platform
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class SystemContextProvider:
    """Injects basic system / runtime info into the prompt.

    Provides: current UTC datetime (ISO 8601), `platform.system()`,
    `os.name`, and the kit version. Output is rendered under a
    ``# System Context`` heading.

    ``clock`` accepts a no-arg callable returning a ``datetime``;
    defaults to ``datetime.now(UTC)``. Replace it in tests to make the
    output deterministic.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__()
        self._clock = clock or (lambda: datetime.now(UTC))

    async def provide(self, context: dict[str, Any]) -> str:  # noqa: ARG002
        try:
            from langgraph_kit import __version__ as kit_version
        except ImportError:
            kit_version = "unknown"

        now = self._clock().isoformat(timespec="seconds")
        return (
            "# System Context\n"
            f"- Current time (UTC): {now}\n"
            f"- Platform: {platform.system()}\n"
            f"- OS name: {os.name}\n"
            f"- Kit version: {kit_version}"
        )
