"""Tests for SparkSessionManager."""

from unittest.mock import MagicMock, patch

import pytest

pyspark = pytest.importorskip("pyspark")

from src.infrastructure.config.settings import SparkSettings  # noqa: E402
from src.infrastructure.data.spark.session_manager import SparkSessionManager  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Reset singleton state between tests."""
    SparkSessionManager._instance = None


class TestSparkSessionManager:
    def test_get_or_create_builds_session(self) -> None:
        settings = SparkSettings()
        mock_session = MagicMock()
        mock_builder = MagicMock()
        mock_builder.appName.return_value = mock_builder
        mock_builder.master.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_session

        with patch("pyspark.sql.SparkSession") as mock_cls:
            mock_cls.builder = mock_builder
            result = SparkSessionManager.get_or_create(settings)

        assert result is mock_session
        mock_builder.appName.assert_called_once_with(settings.APP_NAME)
        mock_builder.master.assert_called_once_with(settings.MASTER)

    def test_singleton_returns_same_instance(self) -> None:
        mock_session = MagicMock()
        SparkSessionManager._instance = mock_session

        settings = SparkSettings()
        result = SparkSessionManager.get_or_create(settings)

        assert result is mock_session

    def test_stop_clears_instance(self) -> None:
        mock_session = MagicMock()
        SparkSessionManager._instance = mock_session

        SparkSessionManager.stop()

        assert SparkSessionManager._instance is None
        mock_session.stop.assert_called_once()

    def test_stop_noop_when_no_instance(self) -> None:
        SparkSessionManager.stop()  # Should not raise

    def test_delta_enabled_adds_extensions(self) -> None:
        settings = SparkSettings(DELTA_ENABLED=True)
        mock_session = MagicMock()
        mock_builder = MagicMock()
        mock_builder.appName.return_value = mock_builder
        mock_builder.master.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_session

        with patch("pyspark.sql.SparkSession") as mock_cls:
            mock_cls.builder = mock_builder
            SparkSessionManager.get_or_create(settings)

        config_calls = mock_builder.config.call_args_list
        config_keys = [call[0][0] for call in config_calls]
        assert "spark.sql.extensions" in config_keys
        assert "spark.sql.catalog.spark_catalog" in config_keys
