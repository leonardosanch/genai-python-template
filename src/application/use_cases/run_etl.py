"""Use case: Run ETL — extract, validate, transform, and load data."""

import time

import structlog

from src.application.dtos.data_engineering import (
    DataQualityReportResponse,
    ETLRunRequest,
    ETLRunResponse,
)
from src.domain.events import DataIngestedEvent, DataValidatedEvent
from src.domain.ports.data_sink_port import DataSinkPort
from src.domain.ports.data_source_port import DataSourcePort
from src.domain.ports.data_validator_port import DataValidatorPort
from src.domain.ports.event_bus_port import EventBusPort

logger = structlog.get_logger(__name__)


class RunETLUseCase:
    """Execute an ETL pipeline: extract → validate → load.

    Dependencies injected via constructor — depends only on ports.
    """

    def __init__(
        self,
        source: DataSourcePort,
        sink: DataSinkPort,
        validator: DataValidatorPort,
        event_bus: EventBusPort,
    ) -> None:
        self._source = source
        self._sink = sink
        self._validator = validator
        self._event_bus = event_bus

    async def execute(self, request: ETLRunRequest) -> ETLRunResponse:
        """Run the ETL pipeline.

        Args:
            request: ETL run configuration.

        Returns:
            ETLRunResponse with execution metrics.
        """
        start = time.perf_counter()
        errors: list[str] = []
        quality_report: DataQualityReportResponse | None = None

        try:
            # Extract
            records = await self._source.read_records(request.source_uri)
            records_extracted = len(records)

            await self._event_bus.publish(
                DataIngestedEvent(
                    dataset_name=request.source_uri,
                    record_count=records_extracted,
                    source_uri=request.source_uri,
                )
            )

            # Validate
            if request.run_validation:
                schema = None
                if request.schema_name:
                    schema = await self._source.read_schema(request.source_uri)
                quality = await self._validator.validate(records, schema)
                quality_report = DataQualityReportResponse(
                    is_valid=quality.is_valid,
                    total_records=quality.total_records,
                    valid_records=quality.valid_records,
                    invalid_records=quality.invalid_records,
                    errors=list(quality.errors),
                )
                await self._event_bus.publish(
                    DataValidatedEvent(
                        dataset_name=request.source_uri,
                        is_valid=quality.is_valid,
                        valid_records=quality.valid_records,
                        invalid_records=quality.invalid_records,
                    )
                )

            # Load
            records_loaded = await self._sink.write_records(request.sink_uri, records)

            duration = time.perf_counter() - start
            return ETLRunResponse(
                status="success",
                records_extracted=records_extracted,
                records_transformed=records_extracted,
                records_loaded=records_loaded,
                duration_seconds=duration,
                quality_result=quality_report,
                errors=errors,
            )

        except Exception as e:
            duration = time.perf_counter() - start
            errors.append(f"{e.__class__.__name__}: {e}")
            return ETLRunResponse(
                status="failed",
                duration_seconds=duration,
                quality_result=quality_report,
                errors=errors,
            )
