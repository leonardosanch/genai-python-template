"""Unit tests for DataCleaningPipeline."""

from unittest.mock import AsyncMock

import pytest

from src.application.pipelines.data_cleaning import CleaningConfig, DataCleaningPipeline
from src.domain.exceptions import DataSourceError
from src.domain.ports.data_sink_port import DataSinkPort
from src.domain.ports.data_source_port import DataSourcePort
from src.domain.ports.data_validator_port import DataValidatorPort
from src.domain.value_objects.data_quality_result import DataQualityResult


@pytest.fixture
def mock_source() -> AsyncMock:
    source = AsyncMock(spec=DataSourcePort)
    source.read_records.return_value = [
        {"name": " Alice ", "age": 30},
        {"name": " Bob ", "age": 25},
    ]
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


class TestDataCleaningPipelineETL:
    async def test_full_etl_flow(
        self,
        mock_source: AsyncMock,
        mock_sink: AsyncMock,
        mock_validator: AsyncMock,
    ) -> None:
        pipeline = DataCleaningPipeline(
            source=mock_source,
            sink=mock_sink,
            validator=mock_validator,
            source_uri="input.csv",
            sink_uri="output.csv",
        )

        result = await pipeline.run()

        assert result.status == "success"
        mock_source.read_records.assert_called_once_with("input.csv")
        mock_sink.write_records.assert_called_once()

    async def test_strip_whitespace(
        self,
        mock_source: AsyncMock,
        mock_sink: AsyncMock,
        mock_validator: AsyncMock,
    ) -> None:
        config = CleaningConfig(strip_whitespace=True)
        pipeline = DataCleaningPipeline(
            source=mock_source,
            sink=mock_sink,
            validator=mock_validator,
            source_uri="in.csv",
            sink_uri="out.csv",
            config=config,
        )

        records = await pipeline.extract()
        transformed = await pipeline.transform(records)

        assert transformed[0]["name"] == "Alice"
        assert transformed[1]["name"] == "Bob"

    async def test_drop_duplicates(
        self,
        mock_sink: AsyncMock,
        mock_validator: AsyncMock,
    ) -> None:
        source = AsyncMock(spec=DataSourcePort)
        source.read_records.return_value = [
            {"id": 1, "val": "a"},
            {"id": 1, "val": "a"},
            {"id": 2, "val": "b"},
        ]
        config = CleaningConfig(drop_duplicates=True)
        pipeline = DataCleaningPipeline(
            source=source,
            sink=mock_sink,
            validator=mock_validator,
            source_uri="in.csv",
            sink_uri="out.csv",
            config=config,
        )

        records = await pipeline.extract()
        transformed = await pipeline.transform(records)
        assert len(transformed) == 2

    async def test_drop_nulls(
        self,
        mock_sink: AsyncMock,
        mock_validator: AsyncMock,
    ) -> None:
        source = AsyncMock(spec=DataSourcePort)
        source.read_records.return_value = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": None},
        ]
        config = CleaningConfig(drop_nulls=True, null_columns=("name",))
        pipeline = DataCleaningPipeline(
            source=source,
            sink=mock_sink,
            validator=mock_validator,
            source_uri="in.csv",
            sink_uri="out.csv",
            config=config,
        )

        records = await pipeline.extract()
        transformed = await pipeline.transform(records)
        assert len(transformed) == 1
        assert transformed[0]["name"] == "Alice"


class TestDataCleaningPipelineErrors:
    async def test_source_error(
        self,
        mock_sink: AsyncMock,
        mock_validator: AsyncMock,
    ) -> None:
        source = AsyncMock(spec=DataSourcePort)
        source.read_records.side_effect = DataSourceError("not found")

        pipeline = DataCleaningPipeline(
            source=source,
            sink=mock_sink,
            validator=mock_validator,
            source_uri="missing.csv",
            sink_uri="out.csv",
        )

        result = await pipeline.run()
        assert result.status == "failed"
        assert "DataSourceError" in result.errors[0]
