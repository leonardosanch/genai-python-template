"""Tests for SparkJDBCSource."""

from unittest.mock import MagicMock

import pytest

pyspark = pytest.importorskip("pyspark")

from src.infrastructure.data.spark.spark_jdbc_source import SparkJDBCSource  # noqa: E402


class TestSparkJDBCSource:
    def setup_method(self) -> None:
        self.mock_spark = MagicMock()
        self.source = SparkJDBCSource(spark=self.mock_spark, fetch_size=5000)

    def test_parse_jdbc_uri(self) -> None:
        uri = "jdbc:postgresql://localhost:5432/mydb?table=public.users"
        url, table = self.source._parse_jdbc_uri(uri)
        assert table == "public.users"
        assert "postgresql" in url
        assert "localhost:5432" in url

    @pytest.mark.asyncio
    async def test_read_records(self) -> None:
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: {"id": 1, "name": "test"}[key]
        mock_df = MagicMock()
        mock_df.columns = ["id", "name"]
        mock_df.collect.return_value = [mock_row]

        mock_reader = MagicMock()
        mock_reader.format.return_value = mock_reader
        mock_reader.options.return_value = mock_reader
        mock_reader.load.return_value = mock_df
        self.mock_spark.read = mock_reader

        uri = "jdbc:postgresql://localhost:5432/db?table=users"
        records = await self.source.read_records(uri)

        assert len(records) == 1
        mock_reader.format.assert_called_with("jdbc")

    @pytest.mark.asyncio
    async def test_read_records_with_query_override(self) -> None:
        mock_df = MagicMock()
        mock_df.columns = []
        mock_df.collect.return_value = []

        mock_reader = MagicMock()
        mock_reader.format.return_value = mock_reader
        mock_reader.options.return_value = mock_reader
        mock_reader.load.return_value = mock_df
        self.mock_spark.read = mock_reader

        uri = "jdbc:postgresql://localhost:5432/db?table=users"
        await self.source.read_records(uri, query="SELECT id FROM users WHERE active")

        call_kwargs = mock_reader.options.call_args[1]
        assert "subq" in call_kwargs["dbtable"]

    @pytest.mark.asyncio
    async def test_read_records_with_partitioning(self) -> None:
        mock_df = MagicMock()
        mock_df.columns = []
        mock_df.collect.return_value = []

        mock_reader = MagicMock()
        mock_reader.format.return_value = mock_reader
        mock_reader.options.return_value = mock_reader
        mock_reader.load.return_value = mock_df
        self.mock_spark.read = mock_reader

        uri = "jdbc:postgresql://localhost:5432/db?table=users"
        await self.source.read_records(uri, partition_column="id", num_partitions=8)

        call_kwargs = mock_reader.options.call_args[1]
        assert call_kwargs["partitionColumn"] == "id"
        assert call_kwargs["numPartitions"] == "8"
