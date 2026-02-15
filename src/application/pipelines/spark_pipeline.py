"""Base pipeline for PySpark DataFrame-native ETL workflows.

Parallel hierarchy to Pipeline (dict-based). Justification: the dict-based ports
force collect(), destroying Spark's distributed execution model. This base class
keeps data as DataFrames throughout extract → transform → load.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import structlog

from src.domain.value_objects.pipeline_result import PipelineResult

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

logger = structlog.get_logger(__name__)


class SparkPipeline(ABC):
    """Abstract base for DataFrame-native Spark ETL pipelines.

    Subclasses implement extract/transform/load operating on DataFrames.
    The run() method orchestrates the flow with error handling and timing.
    """

    @abstractmethod
    def extract(self) -> DataFrame:
        """Extract data from source. Returns a Spark DataFrame."""
        ...

    @abstractmethod
    def transform(self, df: DataFrame) -> DataFrame:
        """Transform extracted DataFrame."""
        ...

    @abstractmethod
    def load(self, df: DataFrame) -> int:
        """Load transformed DataFrame to destination. Returns record count."""
        ...

    def run(self) -> PipelineResult:
        """Execute the complete ETL pipeline (synchronous, runs on Spark)."""
        start_time = time.perf_counter()
        pipeline_name = self.__class__.__name__

        logger.info("spark.pipeline.started", pipeline=pipeline_name)

        try:
            logger.info("spark.pipeline.extract.started")
            df = self.extract()
            logger.info("spark.pipeline.extract.completed")

            logger.info("spark.pipeline.transform.started")
            df = self.transform(df)
            logger.info("spark.pipeline.transform.completed")

            logger.info("spark.pipeline.load.started")
            records_processed = self.load(df)
            logger.info("spark.pipeline.load.completed", records=records_processed)

            duration = time.perf_counter() - start_time
            logger.info(
                "spark.pipeline.completed",
                pipeline=pipeline_name,
                records=records_processed,
                duration=duration,
            )
            return PipelineResult(
                status="success",
                records_processed=records_processed,
                records_failed=0,
                duration_seconds=duration,
                errors=[],
            )

        except Exception as e:
            duration = time.perf_counter() - start_time
            error_msg = f"{e.__class__.__name__}: {e}"
            logger.error("spark.pipeline.failed", pipeline=pipeline_name, error=error_msg)
            return PipelineResult(
                status="failed",
                records_processed=0,
                records_failed=0,
                duration_seconds=duration,
                errors=[error_msg],
            )

    async def run_async(self) -> PipelineResult:
        """Run pipeline in a thread to avoid blocking the event loop."""
        return await asyncio.to_thread(self.run)
