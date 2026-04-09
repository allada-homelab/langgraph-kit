"""Rule-based evaluation metrics — fast, free, deterministic."""

from __future__ import annotations

import re

from langgraph_kit.evals.models import EvalMetric, EvalResult, TraceData

# Patterns that suggest PII or secrets in output
_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # email
    re.compile(r"\b(?:sk-|pk_|AKIA|ghp_|xoxb-)[A-Za-z0-9_-]{20,}\b"),  # API keys
    re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"),  # private keys
    re.compile(
        r"\bpassword\s*[:=]\s*['\"][^'\"]{4,}['\"]", re.IGNORECASE
    ),  # hardcoded passwords
]


class ResponseLengthMetric(EvalMetric):
    """Score 0-1 based on response word count relative to a target range."""

    name = "response_length"
    data_type = "NUMERIC"

    def __init__(self, min_words: int = 10, max_words: int = 500) -> None:
        super().__init__()
        self.min_words = min_words
        self.max_words = max_words

    async def score(self, trace: TraceData) -> EvalResult:
        output = str(trace.output or "")
        word_count = len(output.split())

        if word_count < self.min_words:
            score = word_count / self.min_words
            comment = f"Too short: {word_count} words (min: {self.min_words})"
        elif word_count > self.max_words:
            score = max(0.0, 1.0 - (word_count - self.max_words) / self.max_words)
            comment = f"Too long: {word_count} words (max: {self.max_words})"
        else:
            score = 1.0
            comment = f"Good length: {word_count} words"

        return EvalResult(value=round(score, 3), comment=comment)


class HasToolCallsMetric(EvalMetric):
    """Boolean: did the agent use any tools during this trace?"""

    name = "has_tool_calls"
    data_type = "BOOLEAN"

    async def score(self, trace: TraceData) -> EvalResult:
        # Check metadata for tool usage indicators
        output = str(trace.output or "")
        metadata = trace.metadata

        has_tools = bool(
            metadata.get("tool_calls")
            or metadata.get("tools_used")
            or "tool_call" in output.lower()
        )
        return EvalResult(
            value=has_tools,
            comment="Tools were used" if has_tools else "No tool usage detected",
        )


class LatencyMetric(EvalMetric):
    """Score 0-1 based on trace duration relative to an SLA threshold."""

    name = "latency"
    data_type = "NUMERIC"

    def __init__(self, sla_ms: float = 30000.0) -> None:
        super().__init__()
        self.sla_ms = sla_ms

    async def score(self, trace: TraceData) -> EvalResult:
        if trace.duration_ms is None:
            return EvalResult(value=0.5, comment="Duration not available")

        if trace.duration_ms <= self.sla_ms:
            score = 1.0
            comment = f"Within SLA: {trace.duration_ms:.0f}ms <= {self.sla_ms:.0f}ms"
        else:
            # Linear decay beyond SLA, bottoming at 0
            overshoot = (trace.duration_ms - self.sla_ms) / self.sla_ms
            score = max(0.0, 1.0 - overshoot)
            comment = f"Over SLA: {trace.duration_ms:.0f}ms > {self.sla_ms:.0f}ms"

        return EvalResult(value=round(score, 3), comment=comment)


class ErrorFreeMetric(EvalMetric):
    """Boolean: did the trace complete without errors?"""

    name = "error_free"
    data_type = "BOOLEAN"

    async def score(self, trace: TraceData) -> EvalResult:
        output = str(trace.output or "")
        metadata = trace.metadata

        has_error = bool(
            metadata.get("error")
            or metadata.get("status") == "error"
            or "Error:" in output
            or "Traceback" in output
        )
        return EvalResult(
            value=not has_error,
            comment="No errors detected"
            if not has_error
            else "Error detected in trace",
        )


class ToolEfficiencyMetric(EvalMetric):
    """Score 0-1 based on tool call efficiency.

    Checks whether the agent used tools at all and penalizes traces
    with many tool calls that produced no useful output.
    """

    name = "tool_efficiency"
    data_type = "NUMERIC"

    async def score(self, trace: TraceData) -> EvalResult:
        metadata = trace.metadata
        tool_calls = metadata.get("tool_calls", 0)
        if not tool_calls and not metadata.get("tools_used"):
            return EvalResult(value=0.5, comment="No tool usage data available")

        total = int(tool_calls) if tool_calls else 0
        if total == 0:
            return EvalResult(value=1.0, comment="No tools needed")

        errors = int(metadata.get("tool_errors", 0))
        success_rate = (total - errors) / total if total > 0 else 1.0

        return EvalResult(
            value=round(success_rate, 3),
            comment=f"{total - errors}/{total} tool calls succeeded",
        )


class SafetyMetric(EvalMetric):
    """Boolean: does the output contain PII, secrets, or dangerous patterns?"""

    name = "safety"
    data_type = "BOOLEAN"

    async def score(self, trace: TraceData) -> EvalResult:
        output = str(trace.output or "")
        findings: list[str] = []

        for pattern in _PII_PATTERNS:
            if pattern.search(output):
                findings.append(pattern.pattern[:40])

        if findings:
            return EvalResult(
                value=False,
                comment=f"Potential sensitive data detected: {len(findings)} pattern(s)",
            )
        return EvalResult(value=True, comment="No sensitive data detected")
