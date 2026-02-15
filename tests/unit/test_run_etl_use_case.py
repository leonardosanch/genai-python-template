"""Unit tests for RunETLUseCase."""

from unittest.mock import AsyncMock

import pytest

from src.application.dtos.data_engineering import ETLRunRequest
from src.application.use_cases.run_etl import RunETLUseCase
from src.domain.exceptions import DataSourceError
from src.domain.ports.data_sink_port import DataSinkPort
from src.domain.ports.data_source_port import DataSourcePort
from src.domain.ports.data_validator_port import DataValidatorPort
from src.domain.ports.event_bus_port import EventBusPort
from src.domain.value_objects.data_quality_result import DataQualityResult


@pytest.fixture
def mock_source() -> AsyncMock:
    source = AsyncMock(spec=DataSourcePort)
    source.read_records.return_value = [{"id": 1}, {"id": 2}]
    source.read_schema.return_value = None
    return source


@pytest.fixture
def mock_sink() -> AsyncMock:
    sink = AsyncMock(spec=DataSinkPort)
    sink.write_records.return_value = 2
    return sink


@pytest.fixture
def mock_validator() -> AsyncMock:
    validator = AsyncMock(spec=DataValidatorPort)
    validator.validate.return_value = DataQualityResult(
        is_valid=True, total_records=2, valid_records=2, invalid_records=0
    )
    return validator


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    return AsyncMock(spec=EventBusPort)


class TestRunETLUseCase:
    async def test_happy_path(
        self,
        mock_source: AsyncMock,
        mock_sink: AsyncMock,
        mock_validator: AsyncMock,
        mock_event_bus: AsyncMock,
    ) -> None:
        uc = RunETLUseCase(
            source=mock_source,
            sink=mock_sink,
            validator=mock_validator,
            event_bus=mock_event_bus,
        )
        request = ETLRunRequest(source_uri="in.csv", sink_uri="out.csv")
        result = await uc.execute(request)

        assert result.status == "success"
        assert result.records_extracted == 2
        assert result.records_loaded == 2
        assert result.quality_result is not None
        assert result.quality_result.is_valid is True
        assert mock_event_bus.publish.call_count == 2  # ingested + validated

    async def test_without_validation(
        self,
        mock_source: AsyncMock,
        mock_sink: AsyncMock,
        mock_validator: AsyncMock,
        mock_event_bus: AsyncMock,
    ) -> None:
        uc = RunETLUseCase(
            source=mock_source,
            sink=mock_sink,
            validator=mock_validator,
            event_bus=mock_event_bus,
        )
        request = ETLRunRequest(source_uri="in.csv", sink_uri="out.csv", run_validation=False)
        result = await uc.execute(request)

        assert result.status == "success"
        assert result.quality_result is None
        mock_validator.validate.assert_not_called()

    async def test_source_error(
        self,
        mock_sink: AsyncMock,
        mock_validator: AsyncMock,
        mock_event_bus: AsyncMock,
    ) -> None:
        source = AsyncMock(spec=DataSourcePort)
        source.read_records.side_effect = DataSourceError("file not found")

        uc = RunETLUseCase(
            source=source,
            sink=mock_sink,
            validator=mock_validator,
            event_bus=mock_event_bus,
        )
        request = ETLRunRequest(source_uri="missing.csv", sink_uri="out.csv")
        result = await uc.execute(request)

        assert result.status == "failed"
        assert len(result.errors) == 1
        assert "DataSourceError" in result.errors[0]
