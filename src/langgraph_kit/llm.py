"""LLM factory with multi-provider support.

Detects the provider from the model name and instantiates the appropriate
LangChain chat model. Supports OpenAI-compatible (default), Anthropic
(``claude-*``), and Google (``gemini-*``) models.
"""

from __future__ import annotations

from typing import Any

from langgraph_kit._config import get_config


def build_llm() -> Any:
    """Build a chat model from package configuration.

    Provider detection:
      - ``claude-*`` → ``ChatAnthropic`` (requires ``langchain-anthropic``)
      - ``gemini-*`` → ``ChatGoogleGenerativeAI`` (requires ``langchain-google-genai``)
      - anything else → ``ChatOpenAI`` (default, OpenAI-compatible)

    Returns ``Any`` so callers don't need provider-specific stubs installed.
    """
    config = get_config()
    model = config.llm_model

    if model.startswith("claude"):
        return _build_anthropic(model, config)
    if model.startswith("gemini"):
        return _build_google(model, config)
    return _build_openai(model, config)


def _build_openai(model: str, config: Any) -> Any:
    """Build an OpenAI-compatible chat model."""
    from langchain_openai import (
        ChatOpenAI,  # pyright: ignore[reportMissingModuleSource]
    )

    kwargs: dict[str, Any] = {"model": model, "streaming": True}
    if config.llm_base_url:
        kwargs["base_url"] = config.llm_base_url
    if config.llm_api_key:
        kwargs["api_key"] = config.llm_api_key
    return ChatOpenAI(**kwargs)


def _build_anthropic(model: str, config: Any) -> Any:
    """Build an Anthropic Claude chat model."""
    from langchain_anthropic import (  # pyright: ignore[reportMissingModuleSource]
        ChatAnthropic,
    )

    kwargs: dict[str, Any] = {"model": model, "streaming": True}
    if config.llm_api_key:
        kwargs["api_key"] = config.llm_api_key
    if config.llm_base_url:
        kwargs["base_url"] = config.llm_base_url
    return ChatAnthropic(**kwargs)


def _build_google(model: str, config: Any) -> Any:
    """Build a Google Gemini chat model."""
    from langchain_google_genai import (  # pyright: ignore[reportMissingModuleSource]
        ChatGoogleGenerativeAI,
    )

    kwargs: dict[str, Any] = {"model": model}
    if config.llm_api_key:
        kwargs["google_api_key"] = config.llm_api_key
    return ChatGoogleGenerativeAI(**kwargs)
