"""Tests for streaming module — sentinel parsing."""

from __future__ import annotations

from langgraph_kit.streaming import _parse_sentinel


class TestParseSentinel:
    def test_artifact_sentinel(self) -> None:
        from langgraph_kit.core.artifacts import ARTIFACT_SENTINEL

        result = _parse_sentinel(f'{ARTIFACT_SENTINEL}{{"type": "code", "title": "test"}}')
        assert result is not None
        assert "artifact" in result
        assert result["artifact"]["type"] == "code"

    def test_progress_sentinel(self) -> None:
        from langgraph_kit.core.ui_events import PROGRESS_SENTINEL

        result = _parse_sentinel(f'{PROGRESS_SENTINEL}{{"step": 1, "total": 3}}')
        assert result is not None
        assert "progress" in result
        assert result["progress"]["step"] == 1

    def test_unknown_prefix_returns_none(self) -> None:
        result = _parse_sentinel("Just normal tool output")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        result = _parse_sentinel("")
        assert result is None

    def test_malformed_json_returns_none(self) -> None:
        from langgraph_kit.core.artifacts import ARTIFACT_SENTINEL

        result = _parse_sentinel(f"{ARTIFACT_SENTINEL}{{not valid json}}")
        assert result is None
