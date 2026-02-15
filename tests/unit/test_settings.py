# tests/unit/test_settings.py
import os
from unittest.mock import patch

import pytest

from src.infrastructure.config import get_settings


def test_settings_default_values() -> None:
    """Test that default settings are loaded correctly."""
    # Clear settings cache
    get_settings.cache_clear()

    # Unset all related env vars to ensure defaults are used
    with patch.dict(os.environ, {}, clear=True):
        settings = get_settings()

        assert settings.ENVIRONMENT == "development"
        assert settings.llm.MODE == "openai"
        assert settings.llm.OPENAI_API_KEY == "YOUR_OPENAI_API_KEY"
        assert settings.database.URL == "postgresql+asyncpg://user:password@localhost:5432/genai"
        assert settings.redis.URL == "redis://localhost:6379/0"
        assert settings.observability.LOG_LEVEL == "INFO"


def test_settings_override_from_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that settings can be overridden by environment variables."""
    get_settings.cache_clear()

    test_env = {
        "ENVIRONMENT": "production",
        "LLM__MODE": "anthropic",
        "LLM__ANTHROPIC_API_KEY": "test_anthropic_key",
        "DATABASE__URL": "postgresql+asyncpg://prod_user:prod_pass@prod_host:5432/prod_db",
        "REDIS__URL": "redis://prod_host:6379/1",
        "OBSERVABILITY__LOG_LEVEL": "DEBUG",
    }

    for key, value in test_env.items():
        monkeypatch.setenv(key, value)

    settings = get_settings()

    assert settings.ENVIRONMENT == "production"
    assert settings.llm.MODE == "anthropic"
    assert settings.llm.ANTHROPIC_API_KEY == "test_anthropic_key"
    assert (
        settings.database.URL == "postgresql+asyncpg://prod_user:prod_pass@prod_host:5432/prod_db"
    )
    assert settings.redis.URL == "redis://prod_host:6379/1"
    assert settings.observability.LOG_LEVEL == "DEBUG"

    # Clear cache for other tests
    get_settings.cache_clear()


def test_settings_case_insensitivity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that environment variable names are case-insensitive."""
    get_settings.cache_clear()

    monkeypatch.setenv("app_name", "My Test App")
    settings = get_settings()
    assert settings.APP_NAME == "My Test App"

    get_settings.cache_clear()
