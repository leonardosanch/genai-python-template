"""Tests for SPMigrationExamplePipeline."""

from unittest.mock import MagicMock, patch

import pytest

pyspark = pytest.importorskip("pyspark")

from src.application.pipelines.sp_migration_example import (  # noqa: E402
    SPMigrationConfig,
    SPMigrationExamplePipeline,
)


class TestSPMigrationExamplePipeline:
    def setup_method(self) -> None:
        self.mock_spark = MagicMock()
        self.config = SPMigrationConfig(
            jdbc_url="jdbc:postgresql://localhost:5432/testdb",
            source_table="public.orders",
            output_path="/output/orders",
            output_format="parquet",
        )
        self.pipeline = SPMigrationExamplePipeline(spark=self.mock_spark, config=self.config)

    def test_extract_reads_jdbc(self) -> None:
        mock_reader = MagicMock()
        mock_reader.format.return_value = mock_reader
        mock_reader.options.return_value = mock_reader
        mock_reader.load.return_value = MagicMock()
        self.mock_spark.read = mock_reader

        self.pipeline.extract()

        mock_reader.format.assert_called_with("jdbc")
        call_kwargs = mock_reader.options.call_args[1]
        assert call_kwargs["dbtable"] == "public.orders"

    def test_transform_applies_filters(self) -> None:
        mock_df = MagicMock()
        mock_df.filter.return_value = mock_df
        mock_df.withColumn.return_value = mock_df
        mock_df.dropna.return_value = mock_df

        with patch("src.application.pipelines.sp_migration_example.F"):
            result = self.pipeline.transform(mock_df)

        mock_df.filter.assert_called_once()
        mock_df.withColumn.assert_called_once()
        mock_df.dropna.assert_called_once()
        assert result is mock_df

    def test_load_writes_output(self) -> None:
        mock_df = MagicMock()
        mock_writer = MagicMock()
        mock_writer.format.return_value = mock_writer
        mock_writer.mode.return_value = mock_writer
        mock_writer.save.return_value = None
        mock_df.write = mock_writer
        mock_df.count.return_value = 100

        count = self.pipeline.load(mock_df)

        assert count == 100
        mock_writer.format.assert_called_with("parquet")
        mock_writer.mode.assert_called_with("overwrite")
        mock_writer.save.assert_called_with("/output/orders")

    def test_load_with_partitioning(self) -> None:
        config = SPMigrationConfig(
            jdbc_url="jdbc:postgresql://localhost/db",
            source_table="orders",
            output_path="/output",
            partition_by=("region", "year"),
        )
        pipeline = SPMigrationExamplePipeline(spark=self.mock_spark, config=config)

        mock_df = MagicMock()
        mock_writer = MagicMock()
        mock_writer.format.return_value = mock_writer
        mock_writer.mode.return_value = mock_writer
        mock_writer.partitionBy.return_value = mock_writer
        mock_df.write = mock_writer
        mock_df.count.return_value = 50

        count = pipeline.load(mock_df)

        assert count == 50
        mock_writer.partitionBy.assert_called_once_with("region", "year")

    def test_full_run(self) -> None:
        """Integration test: extract -> transform -> load flow."""
        mock_df = MagicMock()
        mock_df.filter.return_value = mock_df
        mock_df.withColumn.return_value = mock_df
        mock_df.dropna.return_value = mock_df
        mock_df.count.return_value = 25

        mock_reader = MagicMock()
        mock_reader.format.return_value = mock_reader
        mock_reader.options.return_value = mock_reader
        mock_reader.load.return_value = mock_df
        self.mock_spark.read = mock_reader

        mock_writer = MagicMock()
        mock_writer.format.return_value = mock_writer
        mock_writer.mode.return_value = mock_writer
        mock_writer.save.return_value = None
        mock_df.write = mock_writer

        with patch("src.application.pipelines.sp_migration_example.F"):
            result = self.pipeline.run()

        assert result.status == "success"
        assert result.records_processed == 25
