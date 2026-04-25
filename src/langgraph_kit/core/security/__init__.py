"""Security middlewares and helpers.

Inbound: prompt-injection pattern scanner. Outbound: PII / credential
redactor. The audit-log infrastructure (#24) will land alongside.
"""

from .injection_guard import (
    PROMPT_INJECTION_FLAG,
    InjectionMatch,
    PromptInjectionGuardMiddleware,
)
from .injection_patterns import (
    INJECTION_PATTERNS,
    scan_for_injection,
)
from .output_patterns import (
    REDACTION_PLACEHOLDER,
    OutputMatch,
    redact,
    scan_for_unsafe_output,
)
from .output_safety import (
    OUTPUT_SAFETY_FLAG,
    OutputSafetyMiddleware,
    SafetyMode,
)

__all__ = [
    "INJECTION_PATTERNS",
    "OUTPUT_SAFETY_FLAG",
    "PROMPT_INJECTION_FLAG",
    "REDACTION_PLACEHOLDER",
    "InjectionMatch",
    "OutputMatch",
    "OutputSafetyMiddleware",
    "PromptInjectionGuardMiddleware",
    "SafetyMode",
    "redact",
    "scan_for_injection",
    "scan_for_unsafe_output",
]
