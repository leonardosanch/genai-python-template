"""Base pipeline abstraction for ETL workflows.

This module provides the foundation for all data pipelines in the system.
Pipelines follow the ETL pattern: Extract → Transform → Load.
"""

import time
from abc import ABC, abstractmethod

import structlog

from src.domain.value_objects.pipeline_result import PipelineResult

logger = structlog.get_logger(__name__)


class Pipeline(ABC):
    """Abstract base class for ETL pipelines.

    Subclasses must implement extract, transform, and load methods.
    The run() method orchestrates the ETL flow with error handling,
    timing, and structured logging.
    """

    @abstractmethod
    async def extract(self) -> list[dict[str, object]]:
        """Extract data from source.

        Returns:
            List of records as dictionaries

        Raises:
            ExtractionError: If extraction fails
        """
        ...

    @abstractmethod
    async def transform(self, data: list[dict[str, object]]) -> list[dict[str, object]]:
        """Transform extracted data.

        Args:
            data: Raw data from extraction phase

        Returns:
            Transformed data ready for loading

        Raises:
            TransformationError: If transformation fails
        """
        ...

    @abstractmethod
    async def load(self, data: list[dict[str, object]]) -> None:
        """Load transformed data to destination.

        Args:
            data: Transformed data to load

        Raises:
            LoadError: If loading fails
        """
        ...

    async def run(self) -> PipelineResult:
        """Execute the complete ETL pipeline.

        Orchestrates extract → transform → load with:
        - Error handling and partial failure support
        - Duration measurement
        - Record counting
        - Structured logging at each phase

        Returns:
            PipelineResult with execution metrics

        Raises:
            ExtractionError: If extraction phase fails completely
            TransformationError: If transformation phase fails completely
            LoadError: If load phase fails completely
        """
        start_time = time.perf_counter()
        errors: list[str] = []
        records_processed = 0
        records_failed = 0

        logger.info("pipeline.started", pipeline=self.__class__.__name__)

        try:
            # Extract phase
            logger.info("pipeline.extract.started")
            extracted_data = await self.extract()
            total_records = len(extracted_data)
            logger.info(
                "pipeline.extract.completed",
                records_extracted=total_records,
            )

            # Transform phase
            logger.info("pipeline.transform.started")
            transformed_data = await self.transform(extracted_data)
            logger.info(
                "pipeline.transform.completed",
                records_transformed=len(transformed_data),
            )

            # Load phase
            logger.info("pipeline.load.started")
            await self.load(transformed_data)
            records_processed = len(transformed_data)
            logger.info(
                "pipeline.load.completed",
                records_loaded=records_processed,
            )

            duration = time.perf_counter() - start_time

            logger.info(
                "pipeline.completed",
                status="success",
                records_processed=records_processed,
                duration_seconds=duration,
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
            error_msg = f"{e.__class__.__name__}: {str(e)}"
            errors.append(error_msg)

            logger.error(
                "pipeline.failed",
                error=error_msg,
                records_processed=records_processed,
                duration_seconds=duration,
            )

            return PipelineResult(
                status="failed",
                records_processed=records_processed,
                records_failed=records_failed,
                duration_seconds=duration,
                errors=errors,
            )
