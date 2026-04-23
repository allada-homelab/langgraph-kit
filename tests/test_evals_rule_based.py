"""Coverage fill — pure-logic rule-based evaluation metrics.

These metrics have no LLM dependency, so unit testing is straightforward.
Each metric derives an :class:`EvalResult` from a :class:`TraceData`
shape. The contract tests assert the score and comment text stay stable
so downstream dashboards don't drift.
"""

from __future__ import annotations

import pytest

from langgraph_kit.evals.metrics.rule_based import (
    ErrorFreeMetric,
    HasToolCallsMetric,
    LatencyMetric,
    ResponseLengthMetric,
    SafetyMetric,
    ToolEfficiencyMetric,
)
from langgraph_kit.evals.models import EvalResult, TraceData


def _trace(**kwargs: object) -> TraceData:
    defaults: dict[str, object] = {"id": "t", "name": "test"}
    defaults.update(kwargs)
    return TraceData(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ResponseLengthMetric
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_length_within_range_returns_perfect_score() -> None:
    metric = ResponseLengthMetric(min_words=5, max_words=50)
    trace = _trace(output=" ".join(["word"] * 20))
    result = await metric.score(trace)
    assert isinstance(result, EvalResult)
    assert result.value == 1.0
    assert "Good length" in (result.comment or "")


@pytest.mark.asyncio
async def test_response_length_too_short_returns_proportional_score() -> None:
    metric = ResponseLengthMetric(min_words=20, max_words=100)
    trace = _trace(output="too short")  # 2 words
    result = await metric.score(trace)
    assert result.value == round(2 / 20, 3)
    assert "Too short" in (result.comment or "")


@pytest.mark.asyncio
async def test_response_length_too_long_decays_linearly() -> None:
    metric = ResponseLengthMetric(min_words=1, max_words=5)
    trace = _trace(output=" ".join(["w"] * 7))  # 2 over by factor of 2/5
    result = await metric.score(trace)
    assert isinstance(result.value, float)
    assert 0 < result.value < 1
    assert "Too long" in (result.comment or "")


@pytest.mark.asyncio
async def test_response_length_handles_missing_output() -> None:
    metric = ResponseLengthMetric(min_words=5)
    result = await metric.score(_trace(output=None))
    assert result.value == 0.0


# ---------------------------------------------------------------------------
# HasToolCallsMetric
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_has_tool_calls_positive_via_tool_calls_key() -> None:
    result = await HasToolCallsMetric().score(
        _trace(metadata={"tool_calls": 3})
    )
    assert result.value is True
    assert "Tools were used" in (result.comment or "")


@pytest.mark.asyncio
async def test_has_tool_calls_positive_via_tools_used_key() -> None:
    result = await HasToolCallsMetric().score(
        _trace(metadata={"tools_used": ["a"]})
    )
    assert result.value is True


@pytest.mark.asyncio
async def test_has_tool_calls_negative_when_metadata_empty() -> None:
    result = await HasToolCallsMetric().score(_trace(metadata={}))
    assert result.value is False
    assert "No tool usage" in (result.comment or "")


# ---------------------------------------------------------------------------
# LatencyMetric
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_latency_missing_duration_returns_half_score() -> None:
    result = await LatencyMetric(sla_ms=1000).score(_trace(duration_ms=None))
    assert result.value == 0.5
    assert "not available" in (result.comment or "")


@pytest.mark.asyncio
async def test_latency_within_sla_returns_full_score() -> None:
    result = await LatencyMetric(sla_ms=1000).score(_trace(duration_ms=500))
    assert result.value == 1.0
    assert "Within SLA" in (result.comment or "")


@pytest.mark.asyncio
async def test_latency_over_sla_decays_linearly() -> None:
    result = await LatencyMetric(sla_ms=1000).score(_trace(duration_ms=1500))
    assert 0 < float(result.value) < 1
    assert "Over SLA" in (result.comment or "")


@pytest.mark.asyncio
async def test_latency_far_over_sla_bottoms_at_zero() -> None:
    result = await LatencyMetric(sla_ms=1000).score(_trace(duration_ms=10_000))
    assert result.value == 0.0


# ---------------------------------------------------------------------------
# ErrorFreeMetric
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_free_clean_trace_is_true() -> None:
    result = await ErrorFreeMetric().score(_trace(metadata={"status": "ok"}))
    assert result.value is True


@pytest.mark.asyncio
async def test_error_free_detects_explicit_error_key() -> None:
    result = await ErrorFreeMetric().score(
        _trace(metadata={"error": "oops"})
    )
    assert result.value is False


@pytest.mark.asyncio
async def test_error_free_detects_status_error() -> None:
    result = await ErrorFreeMetric().score(
        _trace(metadata={"status": "error"})
    )
    assert result.value is False


@pytest.mark.asyncio
async def test_error_free_detects_tool_errors_count() -> None:
    result = await ErrorFreeMetric().score(_trace(metadata={"tool_errors": 2}))
    assert result.value is False


# ---------------------------------------------------------------------------
# ToolEfficiencyMetric
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_efficiency_no_data_returns_neutral() -> None:
    result = await ToolEfficiencyMetric().score(_trace(metadata={}))
    assert result.value == 0.5


@pytest.mark.asyncio
async def test_tool_efficiency_all_successful() -> None:
    result = await ToolEfficiencyMetric().score(
        _trace(metadata={"tool_calls": 3, "tool_errors": 0})
    )
    assert result.value == 1.0
    assert "3/3" in (result.comment or "")


@pytest.mark.asyncio
async def test_tool_efficiency_partial_failures() -> None:
    result = await ToolEfficiencyMetric().score(
        _trace(metadata={"tool_calls": 4, "tool_errors": 1})
    )
    assert result.value == round(3 / 4, 3)
    assert "3/4" in (result.comment or "")


# ---------------------------------------------------------------------------
# SafetyMetric
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_safety_detects_ssn_pattern() -> None:
    result = await SafetyMetric().score(_trace(output="The SSN is 123-45-6789."))
    assert result.value is False
    assert "sensitive data" in (result.comment or "").lower()


@pytest.mark.asyncio
async def test_safety_detects_email() -> None:
    result = await SafetyMetric().score(
        _trace(output="Contact us at test@example.com for support.")
    )
    assert result.value is False


@pytest.mark.asyncio
async def test_safety_detects_api_key() -> None:
    result = await SafetyMetric().score(
        _trace(output="My key: sk-1234567890abcdef1234567890abcdef")
    )
    assert result.value is False


@pytest.mark.asyncio
async def test_safety_detects_private_key_header() -> None:
    result = await SafetyMetric().score(
        _trace(output="-----BEGIN RSA PRIVATE KEY-----\nFAKE")
    )
    assert result.value is False


@pytest.mark.asyncio
async def test_safety_detects_hardcoded_password() -> None:
    result = await SafetyMetric().score(
        _trace(output='config: password="hunter2"')
    )
    assert result.value is False


@pytest.mark.asyncio
async def test_safety_detects_phone_number() -> None:
    result = await SafetyMetric().score(
        _trace(output="Call me at 555-123-4567 any time.")
    )
    assert result.value is False


@pytest.mark.asyncio
async def test_safety_clean_output_is_true() -> None:
    result = await SafetyMetric().score(
        _trace(output="This response has no sensitive data whatsoever.")
    )
    assert result.value is True
    assert "No sensitive data" in (result.comment or "")
