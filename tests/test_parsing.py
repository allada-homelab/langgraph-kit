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

    def test_two_arrays_in_prose_picks_first(self) -> None:
        """Regression: earlier regex ``\\[.*\\]`` was greedy across the
        whole text, so two valid arrays in the same response concatenated
        into an un-parseable string and the parser silently returned []."""
        raw = (
            'First candidate set: [{"action": "create", "title": "A"}] '
            'Second candidate set: [{"action": "delete", "id": "x"}]'
        )
        result = parse_json_array(raw)
        assert len(result) == 1
        assert result[0]["title"] == "A"

    def test_array_with_strings_containing_brackets(self) -> None:
        """Bracket-balancer must skip brackets inside JSON strings."""
        raw = '[{"text": "literal [x] inside"}]'
        result = parse_json_array(raw)
        assert len(result) == 1
        assert result[0]["text"] == "literal [x] inside"

    def test_bare_fence_without_json_tag(self) -> None:
        raw = '```\n[{"a": 1}]\n```'
        result = parse_json_array(raw)
        assert result == [{"a": 1}]

    def test_drops_non_dict_items(self) -> None:
        """Callers dereference ``.get(...)`` — filter out bare ints/strings."""
        raw = '[{"a": 1}, "skip me", 42, {"b": 2}]'
        result = parse_json_array(raw)
        assert result == [{"a": 1}, {"b": 2}]
