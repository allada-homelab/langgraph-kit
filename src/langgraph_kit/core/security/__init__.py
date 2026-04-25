"""Security middlewares and helpers.

Currently houses the inbound-prompt-injection scanner. The outbound
PII/secret scanner and the audit-log infrastructure
(issues #23 / #24) will join this module as they land.
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

__all__ = [
    "INJECTION_PATTERNS",
    "PROMPT_INJECTION_FLAG",
    "InjectionMatch",
    "PromptInjectionGuardMiddleware",
    "scan_for_injection",
]
