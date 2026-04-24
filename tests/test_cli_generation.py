"""Regression tests for Phase L CLI fixes."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from langgraph_kit.cli import _generate_agent


def test_generate_agent_refuses_to_overwrite_by_default(tmp_path: Path) -> None:
    _generate_agent("my-agent", output_dir=tmp_path)
    with pytest.raises(FileExistsError, match="--force"):
        _generate_agent("my-agent", output_dir=tmp_path)


def test_generate_agent_overwrites_with_force(tmp_path: Path) -> None:
    path1 = _generate_agent("my-agent", output_dir=tmp_path)
    original_text = path1.read_text(encoding="utf-8")

    # Tamper with the existing file; force should overwrite it back.
    path1.write_text("tampered", encoding="utf-8")
    path2 = _generate_agent("my-agent", output_dir=tmp_path, force=True)

    assert path1 == path2
    assert path2.read_text(encoding="utf-8") == original_text


def test_cli_new_exits_nonzero_on_conflict_without_force(tmp_path: Path) -> None:
    # First invocation succeeds.
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "langgraph_kit.cli",
            "new",
            "conf-agent",
            "--output-dir",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    # Second invocation without --force must fail.
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "langgraph_kit.cli",
            "new",
            "conf-agent",
            "--output-dir",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--force" in result.stderr


def test_public_graph_builder_surface_is_importable() -> None:
    """The CLI template imports builder helpers from
    ``langgraph_kit.graphs`` — not the private ``_builder`` path. Ensure
    the re-export is stable."""
    from langgraph_kit.graphs import (
        build_backend_factory,  # noqa: F401
        build_command_dispatcher,  # noqa: F401
        build_deep_agent,  # noqa: F401
        build_middleware_stack,  # noqa: F401
        register_standard_tools,  # noqa: F401
    )


def test_cli_docstring_does_not_advertise_missing_features_flag() -> None:
    """``--features`` was never implemented. Previously it was advertised
    in the module docstring, which misled users."""
    import langgraph_kit.cli as cli_mod

    doc = cli_mod.__doc__ or ""
    assert "--features" not in doc
