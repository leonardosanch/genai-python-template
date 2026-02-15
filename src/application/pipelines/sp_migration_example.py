"""Example: Stored Procedure → PySpark migration pattern.

Demonstrates how to migrate a SQL stored procedure to a PySpark pipeline
using the SparkPipeline base class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog
from pyspark.sql import functions as F

from src.application.pipelines.spark_pipeline import SparkPipeline

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SPMigrationConfig:
    """Configuration for a SP-to-PySpark migration pipeline."""

    jdbc_url: str
    source_table: str
    output_path: str
    output_format: str = "parquet"
    write_mode: str = "overwrite"
    partition_by: tuple[str, ...] = ()
    # JDBC read options
    driver: str = "org.postgresql.Driver"
    fetch_size: int = 10_000
    user: str = ""
    password: str = ""


class SPMigrationExamplePipeline(SparkPipeline):
    """Example pipeline replacing a stored procedure.

    Pattern:
    - extract() replaces FROM / source tables of the SP
    - transform() replaces the SP's business logic (filters, joins, aggregations)
    - load() replaces INSERT INTO / output of the SP
    """

    def __init__(self, spark: SparkSession, config: SPMigrationConfig) -> None:
        self._spark = spark
        self._config = config

    def extract(self) -> DataFrame:
        """Read source table via JDBC (replaces SP's FROM clause)."""
        options: dict[str, str] = {
            "url": self._config.jdbc_url,
            "dbtable": self._config.source_table,
            "driver": self._config.driver,
            "fetchsize": str(self._config.fetch_size),
        }
        if self._config.user:
            options["user"] = self._config.user
        if self._config.password:
            options["password"] = self._config.password

        logger.info(
            "sp_migration.extract",
            table=self._config.source_table,
        )
        return self._spark.read.format("jdbc").options(**options).load()

    def transform(self, df: DataFrame) -> DataFrame:
        """Apply SP business logic as DataFrame operations.

        Override this method with the actual SP logic. This example
        demonstrates common operations found in stored procedures.
        """
        # Example: filter active records, add computed column, drop nulls
        df = df.filter(F.col("status") == "active")
        df = df.withColumn("processed_at", F.current_timestamp())
        df = df.dropna(subset=["id"])
        return df

    def load(self, df: DataFrame) -> int:
        """Write results to output (replaces SP's INSERT INTO)."""
        writer = df.write.format(self._config.output_format).mode(self._config.write_mode)

        if self._config.partition_by:
            writer = writer.partitionBy(*self._config.partition_by)

        logger.info(
            "sp_migration.load",
            path=self._config.output_path,
            format=self._config.output_format,
        )
        writer.save(self._config.output_path)
        return int(df.count())
