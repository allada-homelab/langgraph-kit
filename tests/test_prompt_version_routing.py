"""Tests for ``PromptVersionRouter`` and rollout strategies (#18)."""

from __future__ import annotations

import pytest

from langgraph_kit.core.prompt_assembly import (
    PromptSection,
    PromptVersionRouter,
    RunContext,
    SectionRegistry,
    SectionStability,
    percentage_rollout,
    stable_bucket,
)


def _section(*, version: str, content: str = "x") -> PromptSection:
    return PromptSection(
        id="core_role",
        content=content,
        stability=SectionStability.STABLE,
        priority=100,
        version=version,
    )


def _registry_with_two_versions() -> SectionRegistry:
    reg = SectionRegistry()
    reg.register(_section(version="1", content="v1"))
    # Stage v2 without promoting it: live traffic stays on v1 unless
    # the strategy says otherwise.
    reg.register(_section(version="2", content="v2"), set_current=False)
    return reg


class TestStableBucket:
    """The stable_bucket helper underlies the rollout strategy."""

    def test_same_input_yields_same_bucket(self) -> None:
        a = stable_bucket("user-42", 100)
        b = stable_bucket("user-42", 100)
        assert a == b

    def test_different_inputs_yield_different_buckets(self) -> None:
        # Not strictly guaranteed for any pair, but BLAKE2b on these
        # short distinct strings should disagree at 100 buckets.
        assert stable_bucket("user-1", 100) != stable_bucket("user-2", 100)

    def test_bucket_in_range(self) -> None:
        for i in range(50):
            bucket = stable_bucket(f"user-{i}", 7)
            assert 0 <= bucket < 7

    def test_zero_buckets_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            stable_bucket("user-1", 0)


class TestPercentageRollout:
    """``percentage_rollout`` produces stable, percentage-bucketed strategies."""

    def test_zero_percent_keeps_base_for_everyone(self) -> None:
        strategy = percentage_rollout(
            "core_role",
            new_version="2",
            base_version="1",
            percent_new=0.0,
        )
        for i in range(100):
            ctx = RunContext(user_id=f"user-{i}")
            assert strategy(ctx) == {"core_role": "1"}

    def test_one_hundred_percent_flips_everyone(self) -> None:
        strategy = percentage_rollout(
            "core_role",
            new_version="2",
            base_version="1",
            percent_new=1.0,
        )
        for i in range(100):
            ctx = RunContext(user_id=f"user-{i}")
            assert strategy(ctx) == {"core_role": "2"}

    def test_fifty_percent_is_approximately_split(self) -> None:
        """50% rollout sends ~half to v2; tolerance is generous."""
        strategy = percentage_rollout(
            "core_role",
            new_version="2",
            base_version="1",
            percent_new=0.5,
        )
        new_count = sum(
            1
            for i in range(2000)
            if strategy(RunContext(user_id=f"user-{i}"))["core_role"] == "2"
        )
        # Expected ~1000; allow ±10% slack for hash distribution noise.
        assert 800 < new_count < 1200, new_count

    def test_anonymous_user_falls_back_to_base(self) -> None:
        strategy = percentage_rollout(
            "core_role",
            new_version="2",
            base_version="1",
            percent_new=1.0,  # everyone-on-canary, but anon should still be base
        )
        assert strategy(RunContext(user_id=None)) == {"core_role": "1"}

    def test_assignment_is_stable_for_same_user(self) -> None:
        strategy = percentage_rollout(
            "core_role",
            new_version="2",
            base_version="1",
            percent_new=0.5,
        )
        ctx = RunContext(user_id="user-42")
        first = strategy(ctx)
        for _ in range(10):
            assert strategy(ctx) == first

    def test_invalid_percent_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            percentage_rollout(
                "core_role", new_version="2", base_version="1", percent_new=1.5
            )

    def test_custom_bucket_key(self) -> None:
        """Bucket on something other than user_id (e.g. tenant)."""
        strategy = percentage_rollout(
            "core_role",
            new_version="2",
            base_version="1",
            percent_new=1.0,
            bucket_key=lambda ctx: ctx.extra.get("tenant"),
        )
        # No tenant set → falls back to base.
        anon = RunContext(user_id="user-1")
        assert strategy(anon) == {"core_role": "1"}
        # Tenant set → flips because percent_new=1.0.
        tenanted = RunContext(user_id="user-1", extra={"tenant": "acme"})
        assert strategy(tenanted) == {"core_role": "2"}


class TestPromptVersionRouter:
    """``snapshot`` resolves the per-run versions via the strategy."""

    def test_snapshot_returns_current_when_strategy_is_silent(self) -> None:
        reg = _registry_with_two_versions()
        # Strategy that returns no overrides — every section uses
        # whatever the registry's current pointer says.
        router = PromptVersionRouter(reg, lambda _ctx: {})
        snap = router.snapshot(RunContext(user_id="u1"))
        assert snap == {"core_role": "1"}  # v1 is current; v2 was staged

    def test_snapshot_applies_strategy_override(self) -> None:
        reg = _registry_with_two_versions()
        router = PromptVersionRouter(
            reg,
            percentage_rollout(
                "core_role",
                new_version="2",
                base_version="1",
                percent_new=1.0,
            ),
        )
        snap = router.snapshot(RunContext(user_id="u1"))
        assert snap == {"core_role": "2"}

    def test_snapshot_raises_on_unknown_section(self) -> None:
        reg = SectionRegistry()
        reg.register(_section(version="1"))
        router = PromptVersionRouter(reg, lambda _ctx: {"missing": "1"})
        with pytest.raises(KeyError, match="unknown section id"):
            router.snapshot(RunContext(user_id="u1"))

    def test_snapshot_raises_on_unknown_version(self) -> None:
        reg = SectionRegistry()
        reg.register(_section(version="1"))
        router = PromptVersionRouter(reg, lambda _ctx: {"core_role": "999"})
        with pytest.raises(KeyError, match="unknown version"):
            router.snapshot(RunContext(user_id="u1"))

    def test_snapshot_does_not_mutate_registry(self) -> None:
        reg = _registry_with_two_versions()
        before = reg.current_versions()
        router = PromptVersionRouter(
            reg,
            percentage_rollout(
                "core_role",
                new_version="2",
                base_version="1",
                percent_new=1.0,
            ),
        )
        router.snapshot(RunContext(user_id="u1"))
        after = reg.current_versions()
        assert before == after

    def test_snapshot_includes_other_section_currents(self) -> None:
        """Sections the strategy doesn't touch keep their current version."""
        reg = SectionRegistry()
        reg.register(_section(version="1"))
        reg.register(
            PromptSection(
                id="memory_instructions",
                content="m1",
                stability=SectionStability.STABLE,
                priority=50,
                version="1",
            )
        )
        # Strategy only routes core_role; memory_instructions inherits.
        router = PromptVersionRouter(reg, lambda _ctx: {"core_role": "1"})
        snap = router.snapshot(RunContext(user_id="u1"))
        assert snap == {"core_role": "1", "memory_instructions": "1"}
