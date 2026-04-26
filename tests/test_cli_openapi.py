"""Tests for the ``langgraph-kit openapi`` subcommand (issue #39 v1)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langgraph_kit.cli import _cmd_openapi

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class TestOpenapiCommand:
    def test_stdout_emits_valid_openapi_3x_document(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = _cmd_openapi()
        assert rc == 0
        out = capsys.readouterr().out
        spec = json.loads(out)
        assert spec["openapi"].startswith("3."), spec.get("openapi")
        assert spec["info"]["title"] == "langgraph-kit"
        # Paths exist and at least one agent route is present.
        assert "paths" in spec
        assert any("agents" in path for path in spec["paths"]), (
            f"no /agents path in {sorted(spec['paths'])}"
        )

    def test_output_writes_file_when_path_given(self, tmp_path: Path) -> None:
        out_file = tmp_path / "spec.json"
        rc = _cmd_openapi(output=out_file)
        assert rc == 0
        assert out_file.exists()
        spec = json.loads(out_file.read_text())
        assert "paths" in spec
        # File ends with a newline (POSIX-friendly).
        assert out_file.read_text().endswith("\n")

    def test_indent_zero_emits_compact_json(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = _cmd_openapi(indent=0)
        assert rc == 0
        out = capsys.readouterr().out
        # Compact JSON has no inline whitespace after separators.
        assert ": " not in out  # default ``json.dumps`` uses ", " and ": "
        # Still parseable.
        json.loads(out)

    def test_indent_default_is_human_friendly(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = _cmd_openapi(indent=2)
        assert rc == 0
        out = capsys.readouterr().out
        # Indented output puts opening brace on its own line and uses 2-space indent.
        assert out.startswith("{\n")
        assert "  " in out

    def test_components_schemas_include_kit_request_response_models(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The kit's Pydantic request/response models should appear in components.

        Concrete check: ``InvokeRequest`` and ``InvokeResponse`` are referenced
        by routes the agent router exposes, so FastAPI must include them in
        ``components.schemas``. Confirms the spec is rich enough to drive
        ``openapi-python-client`` (issue #40).
        """
        _cmd_openapi()
        spec = json.loads(capsys.readouterr().out)
        schemas = spec.get("components", {}).get("schemas", {})
        assert "InvokeRequest" in schemas, sorted(schemas)
        assert "InvokeResponse" in schemas, sorted(schemas)
