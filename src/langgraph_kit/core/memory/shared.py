"""Team/shared memory sync — publish, validate, and sync memories across scopes."""

from __future__ import annotations

import logging
import re

from langgraph_kit.core.memory.models import (
    MemoryRecord,
    MemoryScope,
    MemoryType,
)
from langgraph_kit.core.memory.persistent import PersistentMemoryManager

logger = logging.getLogger(__name__)

# Patterns that suggest secrets — reject before publishing to shared scope
_SECRET_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"(?:api[_-]?key|apikey)\s*[:=]\s*\S+",
        r"(?:secret|token|password|passwd|pwd)\s*[:=]\s*\S+",
        r"Bearer\s+[A-Za-z0-9\-._~+/]+=*",
        r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
        r"gh[prost]_[A-Za-z0-9]{36,}",  # GitHub PATs and tokens (ghp_, gho_, ghr_, ghs_, ght_)
        r"github_pat_[A-Za-z0-9_]{20,}",  # GitHub fine-grained PATs
        r"sk-(?:proj-|ant-)?[A-Za-z0-9_\-]{32,}",  # OpenAI + Anthropic API keys
        r"AKIA[A-Z0-9]{16}",  # AWS access key
        r"xox[bpras]-[A-Za-z0-9\-]{10,}",  # Slack tokens
        r"sk_(?:live|test)_[A-Za-z0-9]{20,}",  # Stripe keys
        r'"type"\s*:\s*"service_account"',  # GCP service account JSON
    ]
]

# Memory types that are generally safe to share
_SHAREABLE_TYPES = {MemoryType.PROJECT, MemoryType.REFERENCE}


class SecretDetectedError(Exception):
    """Raised when a memory contains suspected secret material."""


class SharedMemoryManager:
    """Manages publishing and syncing memories to/from team scope.

    Validates content before publishing to shared scope:
    - Scans for secret patterns (API keys, tokens, passwords)
    - Only allows shareable memory types by default
    """

    def __init__(self, memory_manager: PersistentMemoryManager) -> None:
        super().__init__()
        self._memory = memory_manager

    def scan_for_secrets(self, text: str) -> list[str]:
        """Check text for patterns that look like secrets.

        Returns list of matched pattern descriptions. Empty if clean.
        """
        matches: list[str] = []
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                matches.append(pattern.pattern)
        return matches

    async def publish_to_team(
        self,
        record: MemoryRecord,
        *,
        allow_all_types: bool = False,
    ) -> MemoryRecord:
        """Publish a memory record to team scope after validation.

        Raises SecretDetectedError if the record contains suspected secrets.
        Raises ValueError if the memory type is not shareable (unless allow_all_types=True).
        """
        # Type check
        if not allow_all_types and record.type not in _SHAREABLE_TYPES:
            msg = (
                f"Memory type '{record.type.value}' is not shareable by default. "
                f"Shareable types: {[t.value for t in _SHAREABLE_TYPES]}. "
                "Use allow_all_types=True to override."
            )
            raise ValueError(msg)

        # Secret scan
        full_text = f"{record.title} {record.summary} {record.body}"
        secrets = self.scan_for_secrets(full_text)
        if secrets:
            msg = f"Memory contains suspected secrets ({len(secrets)} pattern(s) matched). Refusing to publish to shared scope."
            raise SecretDetectedError(msg)

        # Create in team scope
        team_record = MemoryRecord(
            title=record.title,
            type=record.type,
            scope=MemoryScope.TEAM,
            summary=record.summary,
            body=record.body,
            source=f"published_from:{record.scope.value}:{record.id}",
        )
        return await self._memory.create(team_record)

    async def sync_from_team(
        self,
        target_scope: MemoryScope = MemoryScope.PROJECT,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        """Pull team memories into a target scope (e.g., project).

        Only copies records that don't already exist in the target scope
        (matched by title + type).
        """
        team_records = await self._memory.list_by_scope(MemoryScope.TEAM, limit=limit)
        existing = await self._memory.list_by_scope(target_scope, limit=200)

        # Build set of (title, type) for dedup
        existing_keys = {(r.title, r.type) for r in existing}

        synced: list[MemoryRecord] = []
        for record in team_records:
            if (record.title, record.type) in existing_keys:
                continue

            local = MemoryRecord(
                title=record.title,
                type=record.type,
                scope=target_scope,
                summary=record.summary,
                body=record.body,
                source=f"synced_from:team:{record.id}",
            )
            created = await self._memory.create(local)
            synced.append(created)

        logger.info(
            "Synced %d records from team to %s", len(synced), target_scope.value
        )
        return synced

    async def list_team_memories(
        self,
        memory_type: MemoryType | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        """List memories in the team scope."""
        return await self._memory.list_by_scope(
            MemoryScope.TEAM, memory_type=memory_type, limit=limit
        )
