"""Tests for tool capability and registry module."""

from __future__ import annotations

from langgraph_kit.core.tools.capability import ToolCapability, ToolRisk
from langgraph_kit.core.tools.registry import ToolRegistry


def _dummy_fn() -> str:
    return "ok"


def _make_capability(
    tool_id: str = "tool1",
    name: str = "Tool One",
    description: str = "A test tool",
    risk: ToolRisk = ToolRisk.READ_ONLY,
    tags: list[str] | None = None,
    prompt_guidance: str | None = None,
    profiles: list[str] | None = None,
    worker_types: list[str] | None = None,
) -> ToolCapability:
    return ToolCapability(
        id=tool_id,
        name=name,
        description=description,
        fn=_dummy_fn,
        risk=risk,
        tags=tags or [],
        prompt_guidance=prompt_guidance,
        profiles=profiles,
        worker_types=worker_types,
    )


# ---------------------------------------------------------------------------
# ToolCapability
# ---------------------------------------------------------------------------


class TestToolCapability:
    def test_capability_defaults(self) -> None:
        cap = ToolCapability(id="t", name="T", description="desc", fn=_dummy_fn)
        assert cap.risk == ToolRisk.READ_ONLY
        assert cap.tags == []
        assert cap.prompt_guidance is None
        assert cap.profiles is None
        assert cap.worker_types is None
        assert cap.max_output_tokens is None
        assert cap.offload_large_results is False
        assert cap.interrupt_before is False

    def test_capability_with_all_fields(self) -> None:
        cap = ToolCapability(
            id="t",
            name="T",
            description="desc",
            fn=_dummy_fn,
            risk=ToolRisk.DESTRUCTIVE,
            tags=["io", "net"],
            prompt_guidance="Be careful",
            profiles=["admin"],
            worker_types=["planner"],
            max_output_tokens=500,
            offload_large_results=True,
            interrupt_before=True,
        )
        assert cap.risk == ToolRisk.DESTRUCTIVE
        assert cap.tags == ["io", "net"]
        assert cap.prompt_guidance == "Be careful"
        assert cap.profiles == ["admin"]
        assert cap.worker_types == ["planner"]
        assert cap.max_output_tokens == 500
        assert cap.offload_large_results is True
        assert cap.interrupt_before is True


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_register_and_get(self) -> None:
        registry = ToolRegistry()
        cap = _make_capability()
        registry.register(cap)
        assert registry.get("tool1") is cap
        assert registry.get("nonexistent") is None

    def test_register_many(self) -> None:
        registry = ToolRegistry()
        caps = [_make_capability(tool_id="a"), _make_capability(tool_id="b")]
        registry.register_many(caps)
        assert registry.get("a") is not None
        assert registry.get("b") is not None

    def test_filter_by_profile(self) -> None:
        registry = ToolRegistry()
        registry.register_many(
            [
                _make_capability(tool_id="admin_tool", profiles=["admin"]),
                _make_capability(
                    tool_id="public_tool"
                ),  # profiles=None -> no restriction
            ]
        )
        admin_tools = registry.filter(profile="admin")
        admin_ids = {t.id for t in admin_tools}
        assert "admin_tool" in admin_ids
        assert "public_tool" in admin_ids

        user_tools = registry.filter(profile="user")
        user_ids = {t.id for t in user_tools}
        assert "admin_tool" not in user_ids
        assert "public_tool" in user_ids

    def test_filter_by_worker_type(self) -> None:
        registry = ToolRegistry()
        registry.register_many(
            [
                _make_capability(tool_id="planner_tool", worker_types=["planner"]),
                _make_capability(
                    tool_id="any_tool"
                ),  # worker_types=None -> no restriction
            ]
        )
        planner_tools = registry.filter(worker_type="planner")
        planner_ids = {t.id for t in planner_tools}
        assert "planner_tool" in planner_ids
        assert "any_tool" in planner_ids

        coder_tools = registry.filter(worker_type="coder")
        coder_ids = {t.id for t in coder_tools}
        assert "planner_tool" not in coder_ids

    def test_filter_by_max_risk(self) -> None:
        registry = ToolRegistry()
        registry.register_many(
            [
                _make_capability(tool_id="reader", risk=ToolRisk.READ_ONLY),
                _make_capability(tool_id="writer", risk=ToolRisk.MUTATING),
                _make_capability(tool_id="destroyer", risk=ToolRisk.DESTRUCTIVE),
            ]
        )
        read_only = registry.filter(max_risk=ToolRisk.READ_ONLY)
        assert {t.id for t in read_only} == {"reader"}

        mutating = registry.filter(max_risk=ToolRisk.MUTATING)
        assert {t.id for t in mutating} == {"reader", "writer"}

        destructive = registry.filter(max_risk=ToolRisk.DESTRUCTIVE)
        assert {t.id for t in destructive} == {"reader", "writer", "destroyer"}

    def test_filter_by_tags(self) -> None:
        registry = ToolRegistry()
        registry.register_many(
            [
                _make_capability(tool_id="t1", tags=["io", "net"]),
                _make_capability(tool_id="t2", tags=["db"]),
                _make_capability(tool_id="t3", tags=["io"]),
            ]
        )
        io_tools = registry.filter(tags={"io"})
        assert {t.id for t in io_tools} == {"t1", "t3"}

    def test_compile_tools(self) -> None:
        registry = ToolRegistry()
        registry.register_many(
            [_make_capability(tool_id="a"), _make_capability(tool_id="b")]
        )
        tools = registry.compile_tools()
        assert len(tools) == 2
        assert all(fn is _dummy_fn for fn in tools)

    def test_collect_prompt_fragments(self) -> None:
        registry = ToolRegistry()
        registry.register_many(
            [
                _make_capability(
                    tool_id="t1", name="Search", prompt_guidance="Use for lookups"
                ),
                _make_capability(
                    tool_id="t2", name="Write", prompt_guidance="Use for mutations"
                ),
            ]
        )
        output = registry.collect_prompt_fragments()
        assert "## Tool Guidance" in output
        assert "### Search" in output
        assert "Use for lookups" in output
        assert "### Write" in output
        assert "Use for mutations" in output

    def test_collect_prompt_fragments_empty(self) -> None:
        registry = ToolRegistry()
        registry.register(_make_capability(tool_id="no_guidance"))
        output = registry.collect_prompt_fragments()
        assert output == ""

    def test_remove(self) -> None:
        registry = ToolRegistry()
        registry.register(_make_capability(tool_id="x"))
        registry.remove("x")
        assert registry.get("x") is None

    def test_empty_registry_compile_tools_returns_empty_list(self) -> None:
        """An empty registry must not raise when ``compile_tools`` is called.

        The kit's default-build path relies on ``compile_tools()`` being
        safe to call even if the caller never registered anything — a
        surface assertion because deepagents' ``create_agent`` accepts
        an empty tool list but would reject ``None``.
        """
        registry = ToolRegistry()
        tools = registry.compile_tools()
        assert tools == []

    def test_empty_registry_filter_returns_empty_list(self) -> None:
        """``filter`` with any combination on an empty registry returns an empty list."""
        registry = ToolRegistry()
        assert registry.filter() == []
        assert registry.filter(tags={"io"}) == []
        assert registry.filter(max_risk=ToolRisk.READ_ONLY) == []

    def test_register_upserts_on_id_collision(self) -> None:
        """Registering a second capability with the same id replaces the first.

        The builder relies on this for the "caller's ``configure_tools``
        wins over plugin defaults on id collisions" contract. Asserting
        it directly at the registry level guards against a future
        refactor changing semantics to "first wins" or "raise on
        collision".
        """
        registry = ToolRegistry()
        registry.register(_make_capability(tool_id="x", name="first"))
        registry.register(_make_capability(tool_id="x", name="second"))

        caps = registry.list_all()
        assert len(caps) == 1, (
            f"id collision should upsert, not append; got {[c.name for c in caps]}"
        )
        assert caps[0].name == "second", (
            "Second registration should win (upsert semantics)"
        )
