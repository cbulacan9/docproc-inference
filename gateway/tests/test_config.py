"""Tests for gateway configuration."""

import pytest
from unittest.mock import patch
import os


class TestSettings:
    """Tests for Settings dataclass."""

    def test_from_env_loads_required_vars(self, mock_env_vars):
        """Should load all required environment variables."""
        from app.config import Settings

        settings = Settings.from_env()

        assert settings.api_key == "test-api-key-12345"
        assert settings.runpod_api_key == "test-runpod-key"
        assert settings.runpod_classify_endpoint == "classify-endpoint-id"
        assert settings.runpod_extract_endpoint == "extract-endpoint-id"

    def test_from_env_loads_optional_vars_with_defaults(self, monkeypatch):
        """Should use defaults for optional environment variables."""
        # Set only required vars
        monkeypatch.setenv("GATEWAY_API_KEY", "key")
        monkeypatch.setenv("RUNPOD_API_KEY", "rpkey")
        monkeypatch.setenv("RUNPOD_CLASSIFY_ENDPOINT", "classify")
        monkeypatch.setenv("RUNPOD_EXTRACT_ENDPOINT", "extract")

        from app.config import Settings

        settings = Settings.from_env()

        assert settings.runpod_timeout == 120  # default
        assert settings.runpod_max_retries == 3  # default
        assert settings.rate_limit_requests == 100  # default
        assert settings.rate_limit_window == 60  # default
        assert settings.log_level == "INFO"  # default

    def test_from_env_loads_custom_optional_vars(self, mock_env_vars, monkeypatch):
        """Should load custom values for optional variables."""
        monkeypatch.setenv("RUNPOD_TIMEOUT", "60")
        monkeypatch.setenv("RUNPOD_MAX_RETRIES", "5")
        monkeypatch.setenv("RATE_LIMIT_REQUESTS", "50")
        monkeypatch.setenv("RATE_LIMIT_WINDOW", "30")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        from app.config import Settings

        settings = Settings.from_env()

        assert settings.runpod_timeout == 60
        assert settings.runpod_max_retries == 5
        assert settings.rate_limit_requests == 50
        assert settings.rate_limit_window == 30
        assert settings.log_level == "DEBUG"

    def test_from_env_raises_on_missing_required_vars(self, monkeypatch):
        """Should raise KeyError when required vars are missing."""
        # Clear all env vars
        for key in ["GATEWAY_API_KEY", "RUNPOD_API_KEY",
                    "RUNPOD_CLASSIFY_ENDPOINT", "RUNPOD_EXTRACT_ENDPOINT"]:
            monkeypatch.delenv(key, raising=False)

        from app.config import Settings

        with pytest.raises(KeyError):
            Settings.from_env()


class TestGetSettings:
    """Tests for get_settings caching function."""

    def test_get_settings_returns_settings_instance(self, mock_env_vars):
        """Should return a Settings instance."""
        from app.config import get_settings, Settings

        get_settings.cache_clear()
        settings = get_settings()

        assert isinstance(settings, Settings)

    def test_get_settings_caches_result(self, mock_env_vars):
        """Should return the same instance on subsequent calls."""
        from app.config import get_settings

        get_settings.cache_clear()
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_get_settings_cache_can_be_cleared(self, mock_env_vars, monkeypatch):
        """Should return new instance after cache clear."""
        from app.config import get_settings

        get_settings.cache_clear()
        settings1 = get_settings()

        # Change env and clear cache
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        get_settings.cache_clear()
        settings2 = get_settings()

        assert settings1 is not settings2
        assert settings2.log_level == "WARNING"
