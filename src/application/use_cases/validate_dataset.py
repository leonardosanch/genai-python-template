"""Use case: Validate dataset — read and validate data quality."""

import structlog

from src.application.dtos.data_engineering import DataQualityReportResponse
from src.domain.events import DataValidatedEvent
from src.domain.ports.data_source_port import DataSourcePort
from src.domain.ports.data_validator_port import DataValidatorPort
from src.domain.ports.event_bus_port import EventBusPort
from src.domain.value_objects.schema_definition import SchemaDefinition

logger = structlog.get_logger(__name__)


class ValidateDatasetUseCase:
    """Validate a dataset: read records and check quality.

    Dependencies injected via constructor — depends only on ports.
    """

    def __init__(
        self,
        source: DataSourcePort,
        validator: DataValidatorPort,
        event_bus: EventBusPort,
    ) -> None:
        self._source = source
        self._validator = validator
        self._event_bus = event_bus

    async def execute(
        self,
        source_uri: str,
        schema: SchemaDefinition | None = None,
    ) -> DataQualityReportResponse:
        """Validate a dataset from the given source.

        Args:
            source_uri: Location of the data to validate.
            schema: Optional schema to validate against.

        Returns:
            DataQualityReportResponse with validation results.
        """
        records = await self._source.read_records(source_uri)
        quality = await self._validator.validate(records, schema)

        await self._event_bus.publish(
            DataValidatedEvent(
                dataset_name=source_uri,
                is_valid=quality.is_valid,
                valid_records=quality.valid_records,
                invalid_records=quality.invalid_records,
            )
        )

        return DataQualityReportResponse(
            is_valid=quality.is_valid,
            total_records=quality.total_records,
            valid_records=quality.valid_records,
            invalid_records=quality.invalid_records,
            errors=list(quality.errors),
        )
