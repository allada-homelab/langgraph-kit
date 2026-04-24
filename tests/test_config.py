"""Tests for _config module — AgentConfig, configure_from_settings, _coerce."""

from __future__ import annotations

from langgraph_kit._config import (
    AgentConfig,
    _coerce,
    configure,
    configure_from_settings,
    get_config,
)


class TestAgentConfigRepr:
    def test_repr_masks_api_key(self) -> None:
        config = AgentConfig(llm_api_key="sk-secret1234567890")
        r = repr(config)
        # Long secrets show 4-char prefix + ellipsis, not a plain ``***``.
        assert "sk-s..." in r
        assert "sk-secret1234567890" not in r

    def test_repr_masks_langfuse_secret_key(self) -> None:
        config = AgentConfig(langfuse_secret_key="sk-sec12345678")
        r = repr(config)
        assert "sk-s..." in r
        assert "sk-sec12345678" not in r

    def test_repr_does_not_mask_langfuse_public_key(self) -> None:
        """Public keys are public by design — masking them only confuses log triage."""
        config = AgentConfig(langfuse_public_key="pk-pub12345678")
        r = repr(config)
        assert "pk-pub12345678" in r

    def test_repr_fully_masks_short_secrets(self) -> None:
        """A 1-char secret rendered as ``val[:4] + '...'`` would leak it.
        For values <=8 chars we mask with a fixed ``****`` sentinel."""
        config = AgentConfig(llm_api_key="x")
        r = repr(config)
        assert "llm_api_key='****'" in r
        assert "'x'" not in r

    def test_repr_fully_masks_8_char_boundary(self) -> None:
        config = AgentConfig(llm_api_key="12345678")
        r = repr(config)
        assert "llm_api_key='****'" in r
        assert "12345678" not in r

    def test_repr_shows_empty_keys_unmasked(self) -> None:
        config = AgentConfig()
        r = repr(config)
        assert "llm_api_key=''" in r


class TestConfigureAndGetConfig:
    def test_configure_sets_global(self) -> None:
        original = get_config()
        custom = AgentConfig(llm_model="claude-3-opus")
        configure(custom)
        assert get_config().llm_model == "claude-3-opus"
        # Restore
        configure(original)


class TestConfigureFromSettings:
    def test_exact_match(self) -> None:
        class Settings:
            llm_model = "gpt-4"
            environment = "production"

        config = configure_from_settings(Settings())
        assert config.llm_model == "gpt-4"
        assert config.environment == "production"
        # Restore default
        configure(AgentConfig())

    def test_upper_case_match(self) -> None:
        class Settings:
            LLM_MODEL = "claude-3-haiku"
            DATABASE_URL = "postgres://localhost/db"

        config = configure_from_settings(Settings())
        assert config.llm_model == "claude-3-haiku"
        assert config.database_url == "postgres://localhost/db"
        configure(AgentConfig())

    def test_field_map_override(self) -> None:
        class Settings:
            MY_CUSTOM_DB = "sqlite:///custom.db"
            llm_model = "gpt-4o"

        config = configure_from_settings(
            Settings(),
            field_map={"database_url": "MY_CUSTOM_DB"},
        )
        assert config.database_url == "sqlite:///custom.db"
        assert config.llm_model == "gpt-4o"
        configure(AgentConfig())

    def test_case_insensitive_match(self) -> None:
        class Settings:
            Llm_Model = "gemini-pro"

        config = configure_from_settings(Settings())
        assert config.llm_model == "gemini-pro"
        configure(AgentConfig())

    def test_unmatched_fields_use_defaults(self) -> None:
        class Settings:
            pass

        config = configure_from_settings(Settings())
        assert config.llm_model == "gpt-4o-mini"
        assert config.environment == "local"
        configure(AgentConfig())


class TestCoerce:
    def test_coerce_str_from_non_str(self) -> None:
        import dataclasses

        fields = {f.name: f for f in dataclasses.fields(AgentConfig)}
        # llm_model is a str field
        assert _coerce(fields["llm_model"], 123) == "123"
        assert _coerce(fields["llm_model"], None) == "None"

    def test_coerce_str_passthrough(self) -> None:
        import dataclasses

        fields = {f.name: f for f in dataclasses.fields(AgentConfig)}
        assert _coerce(fields["llm_model"], "gpt-4") == "gpt-4"

    def test_coerce_non_str_passthrough(self) -> None:
        import dataclasses

        fields = {f.name: f for f in dataclasses.fields(AgentConfig)}
        # token_budget_per_thread is an int field — no coercion
        assert _coerce(fields["token_budget_per_thread"], 5000) == 5000
