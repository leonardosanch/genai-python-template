"""Tests for SparkFileSink."""

from unittest.mock import MagicMock

import pytest

pyspark = pytest.importorskip("pyspark")

from src.infrastructure.data.spark.spark_file_sink import SparkFileSink  # noqa: E402


class TestSparkFileSink:
    def setup_method(self) -> None:
        self.mock_spark = MagicMock()
        self.sink = SparkFileSink(spark=self.mock_spark)

    @pytest.mark.asyncio
    async def test_write_records_parquet(self) -> None:
        mock_df = MagicMock()
        mock_writer = MagicMock()
        mock_writer.format.return_value = mock_writer
        mock_writer.mode.return_value = mock_writer
        mock_writer.option.return_value = mock_writer
        mock_df.write = mock_writer
        self.mock_spark.createDataFrame.return_value = mock_df

        records: list[dict[str, object]] = [{"id": 1, "value": "a"}, {"id": 2, "value": "b"}]
        count = await self.sink.write_records("/output/data.parquet", records)

        assert count == 2
        mock_writer.format.assert_called_with("parquet")
        mock_writer.mode.assert_called_with("overwrite")
        mock_writer.save.assert_called_once_with("/output/data.parquet")

    @pytest.mark.asyncio
    async def test_write_records_csv_with_mode(self) -> None:
        mock_df = MagicMock()
        mock_writer = MagicMock()
        mock_writer.format.return_value = mock_writer
        mock_writer.mode.return_value = mock_writer
        mock_writer.option.return_value = mock_writer
        mock_df.write = mock_writer
        self.mock_spark.createDataFrame.return_value = mock_df

        records: list[dict[str, object]] = [{"id": 1}]
        await self.sink.write_records("/output/data.csv", records, format="csv", mode="append")

        mock_writer.format.assert_called_with("csv")
        mock_writer.mode.assert_called_with("append")

    @pytest.mark.asyncio
    async def test_write_records_with_partition_by(self) -> None:
        mock_df = MagicMock()
        mock_writer = MagicMock()
        mock_writer.format.return_value = mock_writer
        mock_writer.mode.return_value = mock_writer
        mock_writer.partitionBy.return_value = mock_writer
        mock_writer.option.return_value = mock_writer
        mock_df.write = mock_writer
        self.mock_spark.createDataFrame.return_value = mock_df

        records: list[dict[str, object]] = [{"id": 1, "region": "us"}]
        await self.sink.write_records("/output/data.parquet", records, partition_by=["region"])

        mock_writer.partitionBy.assert_called_once_with("region")
