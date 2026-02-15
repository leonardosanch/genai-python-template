"""Unit tests for ValidateDatasetUseCase."""

from unittest.mock import AsyncMock

import pytest

from src.application.use_cases.validate_dataset import ValidateDatasetUseCase
from src.domain.ports.data_source_port import DataSourcePort
from src.domain.ports.data_validator_port import DataValidatorPort
from src.domain.ports.event_bus_port import EventBusPort
from src.domain.value_objects.data_quality_result import DataQualityResult
from src.domain.value_objects.schema_definition import (
    FieldDefinition,
    SchemaDefinition,
)


@pytest.fixture
def mock_source() -> AsyncMock:
    source = AsyncMock(spec=DataSourcePort)
    source.read_records.return_value = [{"id": 1}, {"id": 2}]
    return source


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    return AsyncMock(spec=EventBusPort)


class TestValidateDatasetUseCase:
    async def test_valid_dataset(
        self,
        mock_source: AsyncMock,
        mock_event_bus: AsyncMock,
    ) -> None:
        validator = AsyncMock(spec=DataValidatorPort)
        validator.validate.return_value = DataQualityResult(
            is_valid=True, total_records=2, valid_records=2, invalid_records=0
        )

        uc = ValidateDatasetUseCase(
            source=mock_source, validator=validator, event_bus=mock_event_bus
        )
        result = await uc.execute("data.csv")

        assert result.is_valid is True
        assert result.total_records == 2
        mock_event_bus.publish.assert_called_once()

    async def test_invalid_dataset(
        self,
        mock_source: AsyncMock,
        mock_event_bus: AsyncMock,
    ) -> None:
        validator = AsyncMock(spec=DataValidatorPort)
        validator.validate.return_value = DataQualityResult(
            is_valid=False,
            total_records=2,
            valid_records=1,
            invalid_records=1,
            errors=["Record 1: field 'id' is null but not nullable"],
        )

        uc = ValidateDatasetUseCase(
            source=mock_source, validator=validator, event_bus=mock_event_bus
        )
        schema = SchemaDefinition(
            name="test",
            version="1.0.0",
            fields=(FieldDefinition(name="id", data_type="int"),),
        )
        result = await uc.execute("data.csv", schema=schema)

        assert result.is_valid is False
        assert result.invalid_records == 1
        assert len(result.errors) == 1
