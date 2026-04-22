"""Tests for ``langgraph_kit.contrib.fastapi``.

Regression coverage for the ``create_agent_router`` OpenAPI schema bug
where ``from __future__ import annotations`` plus a local ``CurrentUser``
alias caused ``typing.get_type_hints`` to fail on routes that referenced
it, because the name only existed inside the factory's closure — not in
module globals. See CHANGELOG entry for 0.8.0.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from fastapi import Depends, FastAPI

from langgraph_kit.contrib.fastapi import create_agent_router


def _get_user() -> dict[str, str]:
    return {"id": "u1", "email": "x@y"}


def test_openapi_schema_generation_with_annotated_user_alias() -> None:
    """``app.openapi()`` must not raise after including the agent router.

    This is the precise failure mode reported downstream: building the
    OpenAPI schema walks route annotations eagerly, and any unresolved
    ``ForwardRef('CurrentUser')`` raises ``PydanticUserError``.
    """
    current_user: Any = Annotated[dict[str, str], Depends(_get_user)]

    app = FastAPI()
    app.include_router(create_agent_router(get_current_user=current_user))

    schema = app.openapi()
    assert schema["openapi"].startswith("3."), schema
    assert "paths" in schema


def test_openapi_schema_generation_is_deterministic() -> None:
    """Calling ``app.openapi()`` twice (e.g. via a pre-commit hook) stays green."""
    current_user: Any = Annotated[dict[str, str], Depends(_get_user)]

    app = FastAPI()
    app.include_router(create_agent_router(get_current_user=current_user))

    first = app.openapi()
    second = app.openapi()
    assert first is second  # FastAPI memoises, confirm stable identity


def test_router_routes_are_registered() -> None:
    """Sanity check that the factory still wires all expected routes."""
    current_user: Any = Annotated[dict[str, str], Depends(_get_user)]
    router = create_agent_router(get_current_user=current_user)

    paths = {route.path for route in router.routes}  # type: ignore[attr-defined]
    assert "/agents/" in paths
    assert "/agents/threads" in paths
    assert "/agents/{agent_id}/invoke" in paths
    assert "/agents/{agent_id}/stream" in paths


@pytest.mark.parametrize(
    "user_payload_type",
    [dict[str, str], Any],
)
def test_openapi_schema_generation_with_varied_user_types(
    user_payload_type: Any,
) -> None:
    """The inner payload type of the ``Annotated`` alias should not matter."""
    current_user: Any = Annotated[user_payload_type, Depends(_get_user)]

    app = FastAPI()
    app.include_router(create_agent_router(get_current_user=current_user))

    app.openapi()
