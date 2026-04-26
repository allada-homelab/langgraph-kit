"""Tests for ``langgraph_kit._config.validate_config`` (issue #41 v1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph_kit._config import (
    AgentConfig,
    ValidationReport,
    validate_config,
)
from langgraph_kit.cli import _cmd_validate_config

if TYPE_CHECKING:
    import pytest


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestValidateConfigHappy:
    def test_default_config_passes(self) -> None:
        """The bare ``AgentConfig()`` should validate clean — defaults are sane."""
        report = validate_config(AgentConfig())
        assert report.errors == ()
        assert report.warnings == ()
        assert report.is_ok is True

    def test_postgres_url_accepted(self) -> None:
        cfg = AgentConfig(database_url="postgresql://user:pw@localhost/db")
        assert validate_config(cfg).errors == ()

    def test_postgres_psycopg_driver_accepted(self) -> None:
        """The ``+psycopg`` SQLAlchemy driver suffix is the kit's preferred shape."""
        cfg = AgentConfig(database_url="postgresql+psycopg://user:pw@localhost/db")
        assert validate_config(cfg).errors == ()


# ---------------------------------------------------------------------------
# Field-level errors
# ---------------------------------------------------------------------------


class TestValidateConfigErrors:
    def test_unknown_db_scheme_is_error(self) -> None:
        cfg = AgentConfig(database_url="mysql://localhost/db")
        report = validate_config(cfg)
        assert any("database_url" in e and "scheme" in e for e in report.errors), report

    def test_empty_db_url_is_error(self) -> None:
        cfg = AgentConfig(database_url="")
        report = validate_config(cfg)
        assert any("database_url is empty" in e for e in report.errors), report

    def test_empty_llm_model_is_error(self) -> None:
        cfg = AgentConfig(llm_model="")
        report = validate_config(cfg)
        assert any("llm_model is empty" in e for e in report.errors), report

    def test_negative_token_budget_is_error(self) -> None:
        cfg = AgentConfig(token_budget_per_thread=-1)
        report = validate_config(cfg)
        assert any("token_budget_per_thread" in e for e in report.errors), report

    def test_negative_shutdown_timeout_is_error(self) -> None:
        cfg = AgentConfig(shutdown_timeout_seconds=-0.5)
        report = validate_config(cfg)
        assert any("shutdown_timeout_seconds" in e for e in report.errors), report

    def test_invalid_prompt_injection_mode_is_error(self) -> None:
        cfg = AgentConfig(prompt_injection_mode="quarantine")
        report = validate_config(cfg)
        assert any("prompt_injection_mode" in e for e in report.errors), report

    def test_invalid_output_safety_mode_is_error(self) -> None:
        cfg = AgentConfig(output_safety_mode="strip")
        report = validate_config(cfg)
        assert any("output_safety_mode" in e for e in report.errors), report

    def test_invalid_mcp_servers_json_is_error(self) -> None:
        cfg = AgentConfig(mcp_servers="not json")
        report = validate_config(cfg)
        assert any("mcp_servers" in e and "JSON" in e for e in report.errors), report

    def test_valid_mcp_servers_json_no_error(self) -> None:
        cfg = AgentConfig(mcp_servers='{"servers": []}')
        assert validate_config(cfg).errors == ()


# ---------------------------------------------------------------------------
# Cross-field warnings
# ---------------------------------------------------------------------------


class TestValidateConfigWarnings:
    def test_langfuse_public_only_warns(self) -> None:
        cfg = AgentConfig(langfuse_public_key="pk_x")
        report = validate_config(cfg)
        assert report.errors == ()
        assert any(
            "langfuse_public_key" in w and "secret_key" in w for w in report.warnings
        ), report

    def test_langfuse_secret_only_warns(self) -> None:
        cfg = AgentConfig(langfuse_secret_key="sk_x")
        report = validate_config(cfg)
        assert report.errors == ()
        assert any(
            "langfuse_secret_key" in w and "public_key" in w for w in report.warnings
        ), report

    def test_langfuse_complete_pair_no_warning(self) -> None:
        cfg = AgentConfig(
            langfuse_public_key="pk_x",
            langfuse_secret_key="sk_x",
            langfuse_host="https://cloud.langfuse.com",
        )
        report = validate_config(cfg)
        assert report.errors == ()
        # Public/secret pair complete; no Langfuse-related warnings.
        assert not any("langfuse" in w.lower() for w in report.warnings), report


# ---------------------------------------------------------------------------
# Cross-field tracing-enabled guard
# ---------------------------------------------------------------------------


class TestValidateConfigTracingEnabled:
    def test_tracing_enabled_without_keys_is_error(self) -> None:
        cfg = AgentConfig(langfuse_tracing_enabled=True)
        report = validate_config(cfg)
        assert any(
            "tracing_enabled" in e and "secret_key" in e for e in report.errors
        ), report

    def test_tracing_enabled_without_host_is_error(self) -> None:
        cfg = AgentConfig(
            langfuse_tracing_enabled=True,
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
        )
        report = validate_config(cfg)
        assert any(
            "tracing_enabled" in e and "langfuse_host" in e for e in report.errors
        ), report

    def test_tracing_enabled_with_full_config_clean(self) -> None:
        cfg = AgentConfig(
            langfuse_tracing_enabled=True,
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
            langfuse_host="https://cloud.langfuse.com",
        )
        assert validate_config(cfg).errors == ()


# ---------------------------------------------------------------------------
# ValidationReport semantics
# ---------------------------------------------------------------------------


class TestValidationReport:
    def test_is_ok_true_when_no_errors_even_with_warnings(self) -> None:
        rep = ValidationReport(warnings=("just a warning",))
        assert rep.is_ok is True

    def test_is_ok_false_when_errors_present(self) -> None:
        rep = ValidationReport(errors=("nope",))
        assert rep.is_ok is False

    def test_is_ok_true_for_empty_report(self) -> None:
        assert ValidationReport().is_ok is True


# ---------------------------------------------------------------------------
# CLI subcommand
# ---------------------------------------------------------------------------


class TestValidateConfigCommand:
    def test_returns_zero_on_clean_config(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from langgraph_kit._config import configure

        configure(AgentConfig())  # default config — clean
        rc = _cmd_validate_config()
        assert rc == 0
        captured = capsys.readouterr()
        assert "Config is valid" in captured.out

    def test_returns_one_on_errors(self, capsys: pytest.CaptureFixture[str]) -> None:
        from langgraph_kit._config import configure

        configure(AgentConfig(database_url="mysql://nope/db"))
        rc = _cmd_validate_config()
        assert rc == 1
        captured = capsys.readouterr()
        assert "FAIL" in captured.err
        assert "database_url" in captured.err

    def test_returns_zero_on_warnings_only(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from langgraph_kit._config import configure

        configure(AgentConfig(langfuse_public_key="pk_x"))
        rc = _cmd_validate_config()
        assert rc == 0
        captured = capsys.readouterr()
        assert "WARN" in captured.err
        # Warnings-only path notes the count instead of "Config is valid".
        assert "warning" in captured.err.lower()
