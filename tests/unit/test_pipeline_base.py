"""Unit tests for Pipeline base class.

Reference test showing:
- Testing abstract base class via concrete mock implementation
- Testing orchestration logic (extract → transform → load)
- Testing error handling and partial failures
- Testing metrics collection (duration, record counts)
"""

import pytest

from src.application.pipelines.base import Pipeline
from src.domain.exceptions import ExtractionError, LoadError, TransformationError
from src.domain.value_objects.pipeline_result import PipelineResult


class MockPipeline(Pipeline):
    """Concrete pipeline implementation for testing."""

    def __init__(
        self,
        extract_data: list[dict[str, object]] | None = None,
        transform_data: list[dict[str, object]] | None = None,
        extract_error: Exception | None = None,
        transform_error: Exception | None = None,
        load_error: Exception | None = None,
    ) -> None:
        self.extract_data = extract_data or []
        self.transform_data = transform_data or []
        self.extract_error = extract_error
        self.transform_error = transform_error
        self.load_error = load_error
        self.loaded_data: list[dict[str, object]] = []

    async def extract(self) -> list[dict[str, object]]:
        if self.extract_error:
            raise self.extract_error
        return self.extract_data

    async def transform(self, data: list[dict[str, object]]) -> list[dict[str, object]]:
        if self.transform_error:
            raise self.transform_error
        return self.transform_data

    async def load(self, data: list[dict[str, object]]) -> None:
        if self.load_error:
            raise self.load_error
        self.loaded_data = data


class TestPipelineSuccessfulExecution:
    """Test successful pipeline execution."""

    async def test_run_executes_etl_in_order(self) -> None:
        """Verify run() calls extract → transform → load in sequence."""
        extract_data = [{"id": 1}, {"id": 2}]
        transform_data = [{"id": 1, "processed": True}, {"id": 2, "processed": True}]

        pipeline = MockPipeline(
            extract_data=extract_data,
            transform_data=transform_data,
        )

        result = await pipeline.run()

        # Verify data was loaded
        assert pipeline.loaded_data == transform_data
        # Verify result
        assert result.status == "success"
        assert result.records_processed == 2
        assert result.records_failed == 0
        assert result.duration_seconds > 0
        assert result.errors == []

    async def test_run_with_empty_data(self) -> None:
        """Test pipeline with no data to process."""
        pipeline = MockPipeline(extract_data=[], transform_data=[])

        result = await pipeline.run()

        assert result.status == "success"
        assert result.records_processed == 0
        assert result.records_failed == 0
        assert result.errors == []

    async def test_duration_measurement(self) -> None:
        """Verify duration is measured correctly."""
        pipeline = MockPipeline(
            extract_data=[{"id": 1}],
            transform_data=[{"id": 1, "processed": True}],
        )

        result = await pipeline.run()

        # Duration should be positive and reasonable (< 1 second for this simple test)
        assert result.duration_seconds > 0
        assert result.duration_seconds < 1.0


class TestPipelineErrorHandling:
    """Test pipeline error handling."""

    async def test_extraction_error_propagates(self) -> None:
        """Verify ExtractionError is handled correctly."""
        pipeline = MockPipeline(extract_error=ExtractionError("Failed to extract"))

        result = await pipeline.run()

        assert result.status == "failed"
        assert result.records_processed == 0
        assert len(result.errors) == 1
        assert "ExtractionError" in result.errors[0]
        assert "Failed to extract" in result.errors[0]

    async def test_transformation_error_propagates(self) -> None:
        """Verify TransformationError is handled correctly."""
        pipeline = MockPipeline(
            extract_data=[{"id": 1}],
            transform_error=TransformationError("Failed to transform"),
        )

        result = await pipeline.run()

        assert result.status == "failed"
        assert len(result.errors) == 1
        assert "TransformationError" in result.errors[0]
        assert "Failed to transform" in result.errors[0]

    async def test_load_error_propagates(self) -> None:
        """Verify LoadError is handled correctly."""
        pipeline = MockPipeline(
            extract_data=[{"id": 1}],
            transform_data=[{"id": 1, "processed": True}],
            load_error=LoadError("Failed to load"),
        )

        result = await pipeline.run()

        assert result.status == "failed"
        assert len(result.errors) == 1
        assert "LoadError" in result.errors[0]
        assert "Failed to load" in result.errors[0]


class TestPipelineResultValidation:
    """Test PipelineResult value object validation."""

    def test_pipeline_result_immutable(self) -> None:
        """Verify PipelineResult is frozen."""
        result = PipelineResult(
            status="success",
            records_processed=10,
            records_failed=0,
            duration_seconds=1.5,
            errors=[],
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            result.status = "failed"  # type: ignore[misc]

    def test_pipeline_result_validates_negative_records_processed(self) -> None:
        """Verify negative records_processed raises ValueError."""
        with pytest.raises(ValueError, match="records_processed must be non-negative"):
            PipelineResult(
                status="success",
                records_processed=-1,
                records_failed=0,
                duration_seconds=1.0,
                errors=[],
            )

    def test_pipeline_result_validates_negative_records_failed(self) -> None:
        """Verify negative records_failed raises ValueError."""
        with pytest.raises(ValueError, match="records_failed must be non-negative"):
            PipelineResult(
                status="success",
                records_processed=10,
                records_failed=-1,
                duration_seconds=1.0,
                errors=[],
            )

    def test_pipeline_result_validates_negative_duration(self) -> None:
        """Verify negative duration raises ValueError."""
        with pytest.raises(ValueError, match="duration_seconds must be non-negative"):
            PipelineResult(
                status="success",
                records_processed=10,
                records_failed=0,
                duration_seconds=-1.0,
                errors=[],
            )
