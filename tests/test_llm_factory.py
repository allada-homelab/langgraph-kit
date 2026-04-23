"""Coverage fill — ``build_llm`` provider-routing.

``build_llm`` dispatches to provider-specific factory helpers based on
the model-name prefix. Tests assert each dispatch branch calls the
right chat-model class with the config-supplied kwargs. We don't
actually talk to the providers — the classes are stubbed out.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from langgraph_kit._config import AgentConfig
from langgraph_kit.llm import _build_anthropic, _build_google, _build_openai, build_llm


def _config(**overrides: Any) -> AgentConfig:
    defaults: dict[str, Any] = {
        "llm_model": "gpt-4o-mini",
        "llm_base_url": "",
        "llm_api_key": "",
    }
    defaults.update(overrides)
    return AgentConfig(**defaults)


def test_build_openai_passes_model_base_url_and_api_key() -> None:
    with patch("langchain_openai.ChatOpenAI") as ChatOpenAI:
        _build_openai(
            "gpt-4o", _config(llm_base_url="https://x", llm_api_key="sk-x")
        )
    ChatOpenAI.assert_called_once_with(
        model="gpt-4o", streaming=True, base_url="https://x", api_key="sk-x"
    )


def test_build_openai_skips_empty_base_url_and_key() -> None:
    with patch("langchain_openai.ChatOpenAI") as ChatOpenAI:
        _build_openai("gpt-4o", _config())
    # With both empty, only model + streaming are passed.
    call_kwargs = ChatOpenAI.call_args.kwargs
    assert call_kwargs == {"model": "gpt-4o", "streaming": True}


def test_build_anthropic_respects_config() -> None:
    fake_anthropic = MagicMock()
    import sys

    fake_module = MagicMock(ChatAnthropic=fake_anthropic)
    original = sys.modules.get("langchain_anthropic")
    sys.modules["langchain_anthropic"] = fake_module
    try:
        _build_anthropic(
            "claude-sonnet-4-5",
            _config(llm_api_key="sk-ant", llm_base_url="https://x"),
        )
    finally:
        if original is None:
            sys.modules.pop("langchain_anthropic", None)
        else:
            sys.modules["langchain_anthropic"] = original

    fake_anthropic.assert_called_once_with(
        model="claude-sonnet-4-5",
        streaming=True,
        api_key="sk-ant",
        base_url="https://x",
    )


def test_build_google_uses_google_api_key_kwarg() -> None:
    fake_google = MagicMock()
    import sys

    fake_module = MagicMock(ChatGoogleGenerativeAI=fake_google)
    original = sys.modules.get("langchain_google_genai")
    sys.modules["langchain_google_genai"] = fake_module
    try:
        _build_google("gemini-pro", _config(llm_api_key="goog-key"))
    finally:
        if original is None:
            sys.modules.pop("langchain_google_genai", None)
        else:
            sys.modules["langchain_google_genai"] = original

    fake_google.assert_called_once_with(
        model="gemini-pro", google_api_key="goog-key"
    )


def test_build_llm_routes_claude_models_to_anthropic() -> None:
    with (
        patch("langgraph_kit.llm._build_anthropic") as route_anthropic,
        patch("langgraph_kit.llm.get_config", return_value=_config(llm_model="claude-3-opus")),
    ):
        build_llm()
    route_anthropic.assert_called_once()


def test_build_llm_routes_gemini_models_to_google() -> None:
    with (
        patch("langgraph_kit.llm._build_google") as route_google,
        patch("langgraph_kit.llm.get_config", return_value=_config(llm_model="gemini-1.5")),
    ):
        build_llm()
    route_google.assert_called_once()


def test_build_llm_defaults_to_openai_for_unknown_prefixes() -> None:
    with (
        patch("langgraph_kit.llm._build_openai") as route_openai,
        patch(
            "langgraph_kit.llm.get_config",
            return_value=_config(llm_model="some-custom-model"),
        ),
    ):
        build_llm()
    route_openai.assert_called_once()
