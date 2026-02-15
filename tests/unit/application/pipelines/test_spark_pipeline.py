"""Tests for SparkPipeline base class."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

pyspark = pytest.importorskip("pyspark")

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

from src.application.pipelines.spark_pipeline import SparkPipeline  # noqa: E402


class ConcreteSparkPipeline(SparkPipeline):
    """Test implementation of SparkPipeline."""

    def __init__(self, mock_df: MagicMock, record_count: int = 10) -> None:
        self._mock_df = mock_df
        self._record_count = record_count

    def extract(self) -> DataFrame:
        return self._mock_df

    def transform(self, df: DataFrame) -> DataFrame:
        return df

    def load(self, df: DataFrame) -> int:
        return self._record_count


class FailingSparkPipeline(SparkPipeline):
    """Pipeline that fails during transform."""

    def extract(self) -> DataFrame:
        return MagicMock()

    def transform(self, df: DataFrame) -> DataFrame:
        raise ValueError("Transform failed")

    def load(self, df: DataFrame) -> int:
        return 0


class TestSparkPipeline:
    def test_run_success(self) -> None:
        mock_df = MagicMock()
        pipeline = ConcreteSparkPipeline(mock_df=mock_df, record_count=42)

        result = pipeline.run()

        assert result.status == "success"
        assert result.records_processed == 42
        assert result.errors == []
        assert result.duration_seconds >= 0

    def test_run_failure(self) -> None:
        pipeline = FailingSparkPipeline()

        result = pipeline.run()

        assert result.status == "failed"
        assert result.records_processed == 0
        assert len(result.errors) == 1
        assert "Transform failed" in result.errors[0]

    @pytest.mark.asyncio
    async def test_run_async(self) -> None:
        mock_df = MagicMock()
        pipeline = ConcreteSparkPipeline(mock_df=mock_df, record_count=5)

        result = await pipeline.run_async()

        assert result.status == "success"
        assert result.records_processed == 5
