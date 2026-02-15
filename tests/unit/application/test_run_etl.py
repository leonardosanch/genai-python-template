"""Tests for RunETLUseCase — extract, validate, load pipeline."""

from unittest.mock import AsyncMock, create_autospec

import pytest

from src.application.dtos.data_engineering import ETLRunRequest
from src.application.use_cases.run_etl import RunETLUseCase
from src.domain.ports.data_sink_port import DataSinkPort
from src.domain.ports.data_source_port import DataSourcePort
from src.domain.ports.data_validator_port import DataValidatorPort
from src.domain.ports.event_bus_port import EventBusPort
from src.domain.value_objects.data_quality_result import DataQualityResult


@pytest.fixture()
def mock_source() -> DataSourcePort:
    return create_autospec(DataSourcePort, instance=True)


@pytest.fixture()
def mock_sink() -> DataSinkPort:
    return create_autospec(DataSinkPort, instance=True)


@pytest.fixture()
def mock_validator() -> DataValidatorPort:
    return create_autospec(DataValidatorPort, instance=True)


@pytest.fixture()
def mock_event_bus() -> EventBusPort:
    return create_autospec(EventBusPort, instance=True)


def _make_records(n: int = 3) -> list[dict[str, object]]:
    return [{"id": i, "name": f"record_{i}"} for i in range(n)]


class TestRunETLUseCase:
    """Tests for the ETL pipeline use case."""

    @pytest.mark.asyncio()
    async def test_success_without_validation(
        self,
        mock_source: DataSourcePort,
        mock_sink: DataSinkPort,
        mock_validator: DataValidatorPort,
        mock_event_bus: EventBusPort,
    ) -> None:
        records = _make_records(5)
        mock_source.read_records = AsyncMock(return_value=records)
        mock_sink.write_records = AsyncMock(return_value=5)
        mock_event_bus.publish = AsyncMock()

        uc = RunETLUseCase(
            source=mock_source,
            sink=mock_sink,
            validator=mock_validator,
            event_bus=mock_event_bus,
        )
        request = ETLRunRequest(
            source_uri="input.csv",
            sink_uri="output.csv",
            run_validation=False,
        )
        result = await uc.execute(request)

        assert result.status == "success"
        assert result.records_extracted == 5
        assert result.records_loaded == 5
        assert result.quality_result is None

    @pytest.mark.asyncio()
    async def test_success_with_validation(
        self,
        mock_source: DataSourcePort,
        mock_sink: DataSinkPort,
        mock_validator: DataValidatorPort,
        mock_event_bus: EventBusPort,
    ) -> None:
        records = _make_records(10)
        mock_source.read_records = AsyncMock(return_value=records)
        mock_source.read_schema = AsyncMock(return_value=None)
        mock_sink.write_records = AsyncMock(return_value=10)
        mock_event_bus.publish = AsyncMock()
        mock_validator.validate = AsyncMock(
            return_value=DataQualityResult(
                is_valid=True,
                total_records=10,
                valid_records=10,
                invalid_records=0,
                errors=(),
            ),
        )

        uc = RunETLUseCase(
            source=mock_source,
            sink=mock_sink,
            validator=mock_validator,
            event_bus=mock_event_bus,
        )
        request = ETLRunRequest(
            source_uri="data.csv",
            sink_uri="out.csv",
            run_validation=True,
            schema_name="my_schema",
        )
        result = await uc.execute(request)

        assert result.status == "success"
        assert result.quality_result is not None
        assert result.quality_result.is_valid is True

    @pytest.mark.asyncio()
    async def test_failure_returns_error_status(
        self,
        mock_source: DataSourcePort,
        mock_sink: DataSinkPort,
        mock_validator: DataValidatorPort,
        mock_event_bus: EventBusPort,
    ) -> None:
        mock_source.read_records = AsyncMock(side_effect=RuntimeError("connection lost"))
        mock_event_bus.publish = AsyncMock()

        uc = RunETLUseCase(
            source=mock_source,
            sink=mock_sink,
            validator=mock_validator,
            event_bus=mock_event_bus,
        )
        request = ETLRunRequest(source_uri="bad.csv", sink_uri="out.csv")
        result = await uc.execute(request)

        assert result.status == "failed"
        assert len(result.errors) == 1
        assert "connection lost" in result.errors[0]

    @pytest.mark.asyncio()
    async def test_publishes_events(
        self,
        mock_source: DataSourcePort,
        mock_sink: DataSinkPort,
        mock_validator: DataValidatorPort,
        mock_event_bus: EventBusPort,
    ) -> None:
        mock_source.read_records = AsyncMock(return_value=_make_records(2))
        mock_sink.write_records = AsyncMock(return_value=2)
        mock_event_bus.publish = AsyncMock()

        uc = RunETLUseCase(
            source=mock_source,
            sink=mock_sink,
            validator=mock_validator,
            event_bus=mock_event_bus,
        )
        request = ETLRunRequest(
            source_uri="s.csv",
            sink_uri="d.csv",
            run_validation=False,
        )
        await uc.execute(request)

        # Should publish DataIngestedEvent
        assert mock_event_bus.publish.call_count == 1
