import logging

import pytest

from app import config
from app.exceptions import ConfigError
from app.logging_config import RedactSecretsFilter


def test_defaults_without_environment():
    s = config.Settings.from_env()
    assert s.host == "127.0.0.1"  # loopback by default, not 0.0.0.0
    assert s.is_loopback_host
    assert s.port == 7860
    assert s.chunk_size == 1000 and s.chunk_overlap == 200
    assert s.auth is None
    assert s.server_openai_api_key == ""


def test_environment_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("PDFCHAT_PORT", "9000")
    monkeypatch.setenv("PDFCHAT_HOST", "0.0.0.0")
    monkeypatch.setenv("PDFCHAT_CHROMA_DIR", str(tmp_path / "db"))
    monkeypatch.setenv("PDFCHAT_USE_SERVER_KEY", "false")
    monkeypatch.setenv("PDFCHAT_OPENAI_TIMEOUT_S", "12.5")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-key-123456")
    s = config.Settings.from_env()
    assert s.port == 9000
    assert not s.is_loopback_host
    assert s.chroma_dir == tmp_path / "db"
    assert s.use_server_key is False
    assert s.openai_timeout_s == 12.5
    assert s.server_openai_api_key == "sk-server-key-123456"


def test_blank_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("PDFCHAT_PORT", "   ")
    assert config.Settings.from_env().port == 7860


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PDFCHAT_PORT", "abc"),
        ("PDFCHAT_PORT", "70000"),
        ("PDFCHAT_RETRIEVER_K", "0"),
        ("PDFCHAT_OPENAI_TIMEOUT_S", "fast"),
        ("PDFCHAT_USE_SERVER_KEY", "maybe"),
        ("PDFCHAT_LOG_LEVEL", "LOUD"),
    ],
)
def test_invalid_values_raise_config_error(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError):
        config.Settings.from_env()


def test_chunk_overlap_must_be_smaller_than_chunk_size(monkeypatch):
    monkeypatch.setenv("PDFCHAT_CHUNK_SIZE", "300")
    monkeypatch.setenv("PDFCHAT_CHUNK_OVERLAP", "300")
    with pytest.raises(ConfigError, match="smaller"):
        config.Settings.from_env()


def test_auth_requires_both_user_and_password(monkeypatch):
    monkeypatch.setenv("PDFCHAT_AUTH_USER", "admin")
    with pytest.raises(ConfigError, match="both"):
        config.Settings.from_env()
    monkeypatch.setenv("PDFCHAT_AUTH_PASSWORD", "hunter2")
    assert config.Settings.from_env().auth == ("admin", "hunter2")


def test_secrets_are_not_in_repr():
    s = config.Settings(server_openai_api_key="sk-secret-value", auth_user="a", auth_password="pw")
    text = repr(s)
    assert "sk-secret-value" not in text
    assert "pw" not in text.replace("password", "")


def test_loading_settings_has_no_filesystem_side_effects(monkeypatch, tmp_path):
    target = tmp_path / "should-not-exist"
    monkeypatch.setenv("PDFCHAT_CHROMA_DIR", str(target))
    s = config.Settings.from_env()
    assert not target.exists()
    config.ensure_directories(s)
    assert target.is_dir()


def test_get_settings_is_cached(monkeypatch):
    first = config.get_settings()
    monkeypatch.setenv("PDFCHAT_PORT", "1234")
    assert config.get_settings() is first
    config.get_settings.cache_clear()
    assert config.get_settings().port == 1234


def test_log_filter_redacts_api_keys():
    record = logging.LogRecord(
        "x", logging.INFO, __file__, 1, "key=%s failed", ("sk-proj-abcdefghijkl123456",), None
    )
    assert RedactSecretsFilter().filter(record)
    assert "abcdefghijkl" not in record.getMessage()
    assert "sk-***" in record.getMessage()


@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "-inf"])
def test_non_finite_numbers_are_rejected(monkeypatch, value):
    """float() accepts these and they pass a ">= minimum" check, but as timeouts they mean
    "wait forever"."""
    monkeypatch.setenv("PDFCHAT_LOCAL_TOKEN_TIMEOUT_S", value)
    with pytest.raises(ConfigError, match="finite"):
        config.Settings.from_env()


@pytest.mark.parametrize("name", ["openai_timeout_s", "local_token_timeout_s"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
def test_timeouts_set_in_code_must_be_positive_and_finite(name, value):
    with pytest.raises(ConfigError, match="positive, finite"):
        config.Settings(**{name: value})


def test_retrieval_mode_defaults_to_hybrid_and_is_case_insensitive(monkeypatch):
    assert config.Settings.from_env().retrieval_mode == config.RETRIEVAL_HYBRID
    monkeypatch.setenv("PDFCHAT_RETRIEVAL_MODE", "Vector")
    assert config.Settings.from_env().retrieval_mode == config.RETRIEVAL_VECTOR


def test_unknown_retrieval_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("PDFCHAT_RETRIEVAL_MODE", "magic")
    with pytest.raises(ConfigError, match="RETRIEVAL_MODE"):
        config.Settings.from_env()
