# tests/unit/test_tasks.py
from unittest.mock import AsyncMock, patch

import pytest

from src.domain.value_objects.pipeline_result import PipelineResult
from src.infrastructure.tasks.worker import run_pipeline_task


@pytest.fixture
def mock_pipeline_result() -> PipelineResult:
    return PipelineResult(
        status="success",
        records_processed=10,
        records_failed=0,
        duration_seconds=1.5,
        errors=[],
    )


def test_run_pipeline_task_success(mock_pipeline_result: PipelineResult) -> None:
    """Test successful pipeline execution via task."""
    with patch("src.infrastructure.tasks.worker.DocumentIngestionPipeline") as MockPipeline:
        # Mock pipeline instance
        mock_instance = MockPipeline.return_value
        mock_instance.run = AsyncMock(return_value=mock_pipeline_result)

        # Test params
        params = {
            "source_path": "data/test",
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

        # Run task synchronously (Celery tasks are callable)
        result = run_pipeline_task(
            pipeline_name="document_ingestion",
            params=params,
        )

        # Verify result structure
        assert result["status"] == "success"
        assert result["records_processed"] == 10
        assert result["duration_seconds"] == 1.5

        # Verify pipeline initialization with correct params
        MockPipeline.assert_called_once()
        call_kwargs = MockPipeline.call_args.kwargs
        assert call_kwargs["source_prefix"] == "data/test"
        assert call_kwargs["chunk_size"] == 500
        assert call_kwargs["chunk_overlap"] == 50


def test_run_pipeline_task_unknown_pipeline() -> None:
    """Test validating pipeline name."""
    with pytest.raises(ValueError, match="Unknown pipeline"):
        run_pipeline_task(
            pipeline_name="unknown_pipeline",
            params={},
        )
