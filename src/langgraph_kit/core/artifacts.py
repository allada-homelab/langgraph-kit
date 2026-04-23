"""Artifact system — structured UI events emitted alongside text tokens.

Agents call ``create_artifact`` to produce rich content (code blocks, markdown,
tables, diagrams) that the frontend renders in a side panel. The tool returns
a :data:`ARTIFACT_SENTINEL`-prefixed JSON payload; the streaming layer
(:mod:`langgraph_kit.streaming`) detects the prefix on ``on_tool_end`` events
and emits a dedicated ``artifact`` SSE event — no queue required.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ArtifactType(StrEnum):
    CODE = "code"
    MARKDOWN = "markdown"
    TABLE = "table"
    DIAGRAM = "diagram"
    JSON = "json"
    DIFF = "diff"
    HTML = "html"


class ArtifactEvent(BaseModel):
    """A structured artifact to render in the frontend side panel."""

    id: str
    type: ArtifactType
    title: str
    content: str
    language: str | None = None  # for code blocks
    metadata: dict[str, Any] = Field(default_factory=dict)


# Prefix returned by the ``create_artifact`` tool so the streaming layer can
# detect artifact outputs among normal tool results.
ARTIFACT_SENTINEL = "__artifact__:"


# ---------------------------------------------------------------------------
# Agent tool
# ---------------------------------------------------------------------------


def build_artifact_tool() -> Any:
    """Create the ``create_artifact`` tool for agents.

    Returns an async callable suitable for registration in ToolRegistry.
    """

    async def create_artifact(
        artifact_type: str,
        title: str,
        content: str,
        language: str = "",
    ) -> str:
        """Create a rich artifact to display alongside your response.

        Use this to present structured content that benefits from dedicated
        rendering — code with syntax highlighting, markdown documents, data
        tables, or mermaid diagrams.

        Args:
            artifact_type: One of "code", "markdown", "table", "diagram", "json"
            title: Short title for the artifact panel header
            content: The artifact body (code, markdown text, JSON string, mermaid markup)
            language: Programming language for code artifacts (e.g. "python", "typescript")
        """
        try:
            art_type = ArtifactType(artifact_type.lower())
        except ValueError:
            valid = ", ".join(t.value for t in ArtifactType)
            return f"Invalid artifact type '{artifact_type}'. Use one of: {valid}"

        artifact = ArtifactEvent(
            id=str(uuid4()),
            type=art_type,
            title=title,
            content=content,
            language=language or None,
        )

        # Return a sentinel-prefixed JSON string that the streaming layer
        # detects and emits as a dedicated SSE artifact event.
        return ARTIFACT_SENTINEL + json.dumps(artifact.model_dump(mode="json"))

    return create_artifact
