"""Dedicated compaction prompt pack with structured output for conversation summarization."""

from __future__ import annotations

import json
import re
from enum import StrEnum

from pydantic import BaseModel

_RE_SUMMARY = re.compile(r"<summary>\s*(.*?)\s*</summary>", re.DOTALL)
_RE_ANALYSIS = re.compile(r"<analysis>\s*(.*?)\s*</analysis>", re.DOTALL)

_NO_TOOLS_PREAMBLE = """CRITICAL INSTRUCTIONS:
- Respond with TEXT ONLY.
- Do NOT call any tools. Tool use during compaction is a failure mode.
- Do NOT attempt to solve or continue the user's task.
- Your only job is to produce a faithful summary of what has happened so far."""

_FULL_COMPACTION_PROMPT = """{preamble}

You are performing FULL conversation compaction. Summarize the ENTIRE conversation history.

First, produce a private <analysis> block to organize your thoughts about the important context.
Then, produce a <summary> block with the following JSON structure:

<summary>
{{
  "user_intent": "what the user originally asked for",
  "key_decisions": ["list of important decisions made"],
  "important_files": ["list of files that were read, modified, or are relevant"],
  "errors_and_fixes": ["list of errors encountered and how they were resolved"],
  "current_state": "what state the work is in right now",
  "pending_work": ["list of things still to be done"],
  "next_step": "the immediate next action that should be taken"
}}
</summary>

Preserve continuity-critical details: exact file paths, exact error messages, exact decisions.
Do not generalize or lose specifics that would be needed to resume the work."""

_PARTIAL_COMPACTION_PROMPT = """{preamble}

You are performing PARTIAL conversation compaction. Summarize only the RECENT portion of the conversation (the messages provided below). Earlier context is already retained.

Focus on:
- Recent user requests or clarifications
- Recent changes made
- Recent errors encountered
- Current state of work
- Immediate next steps

First, produce a private <analysis> block to organize your thoughts.
Then, produce a <summary> block with the following JSON structure:

<summary>
{{
  "user_intent": "what the user most recently asked for",
  "key_decisions": ["decisions made in the recent portion"],
  "important_files": ["files read, modified, or referenced recently"],
  "errors_and_fixes": ["errors encountered and how they were resolved"],
  "current_state": "what state the work is in right now",
  "pending_work": ["things still to be done"],
  "next_step": "the immediate next action that should be taken"
}}
</summary>

Preserve exact file paths, error messages, and decisions — do not generalize."""


class CompactionMode(StrEnum):
    """Whether to summarize the full conversation or only the recent tail."""

    FULL = "full"
    PARTIAL = "partial"


class CompactionResult(BaseModel):
    """Structured output from a compaction pass."""

    user_intent: str
    key_decisions: list[str]
    important_files: list[str]
    errors_and_fixes: list[str]
    current_state: str
    pending_work: list[str]
    next_step: str
    mode: CompactionMode


class CompactionPromptPack:
    """Builds compaction prompts and parses their structured output."""

    def build_prompt(
        self,
        mode: CompactionMode,
        session_notebook: str | None = None,
    ) -> str:
        """Build the compaction prompt for the given mode.

        If a session notebook is provided, it's included as additional
        continuity context the compactor can reference.
        """
        if mode == CompactionMode.FULL:
            prompt = _FULL_COMPACTION_PROMPT.format(preamble=_NO_TOOLS_PREAMBLE)
        else:
            prompt = _PARTIAL_COMPACTION_PROMPT.format(preamble=_NO_TOOLS_PREAMBLE)

        if session_notebook:
            prompt += (
                f"\n\n## Current Session Notebook (for reference)\n{session_notebook}"
            )

        return prompt

    def parse_output(
        self, raw: str, mode: CompactionMode | None = None
    ) -> CompactionResult | None:
        """Parse the compaction output, extracting the <summary> JSON block.

        Returns None if parsing fails (the <summary> block is missing or malformed).
        The *mode* parameter is stamped onto the result; defaults to FULL if not given.
        """
        match = _RE_SUMMARY.search(raw)
        if not match:
            return None

        try:
            data = json.loads(match.group(1))
            return CompactionResult(**data, mode=mode or CompactionMode.FULL)
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            return None

    def parse_analysis(self, raw: str) -> str:
        """Extract the <analysis> block content (scratchpad, not retained long-term)."""
        match = _RE_ANALYSIS.search(raw)
        return match.group(1).strip() if match else ""
