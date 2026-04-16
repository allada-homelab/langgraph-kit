"""Tests for core/memory/_parsing — shared JSON array parser."""

from __future__ import annotations

from langgraph_kit.core.memory._parsing import parse_json_array


class TestParseJsonArray:
    def test_valid_json_array(self) -> None:
        result = parse_json_array('[{"action": "create", "title": "Test"}]')
        assert len(result) == 1
        assert result[0]["action"] == "create"

    def test_multiple_items(self) -> None:
        raw = '[{"a": 1}, {"a": 2}, {"a": 3}]'
        result = parse_json_array(raw)
        assert len(result) == 3

    def test_json_embedded_in_text(self) -> None:
        raw = 'Here are the results:\n[{"action": "update"}]\nDone.'
        result = parse_json_array(raw)
        assert len(result) == 1
        assert result[0]["action"] == "update"

    def test_json_in_markdown_fence(self) -> None:
        raw = '```json\n[{"title": "Note"}]\n```'
        result = parse_json_array(raw)
        assert len(result) == 1

    def test_empty_array(self) -> None:
        result = parse_json_array("[]")
        assert result == []

    def test_malformed_json(self) -> None:
        result = parse_json_array("{not valid json}")
        assert result == []

    def test_empty_string(self) -> None:
        result = parse_json_array("")
        assert result == []

    def test_whitespace_only(self) -> None:
        result = parse_json_array("   \n  ")
        assert result == []

    def test_json_object_not_array(self) -> None:
        result = parse_json_array('{"key": "value"}')
        assert result == []

    def test_custom_context_in_warning(self) -> None:
        # Just verify it doesn't crash with custom context
        result = parse_json_array("bad", context="test context")
        assert result == []

    def test_nested_arrays(self) -> None:
        raw = '[{"tags": ["a", "b"]}, {"tags": []}]'
        result = parse_json_array(raw)
        assert len(result) == 2
        assert result[0]["tags"] == ["a", "b"]
