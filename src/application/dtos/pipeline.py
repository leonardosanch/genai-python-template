"""DTOs for pipeline operations."""

from typing import Any

from pydantic import BaseModel, Field


class PipelineRunRequest(BaseModel):
    """Request to run a pipeline."""

    source_path: str = Field(
        ...,
        description="Path prefix or pattern for files to ingest",
        examples=["documents/", "data/*.txt"],
    )
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum characters per chunk",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Overlap between consecutive chunks",
    )


class AsyncTaskResponse(BaseModel):
    """Response for async task submission."""

    task_id: str = Field(..., description="ID of the background task")
    status: str = Field(..., description="Initial status of the task")


class TaskStatusResponse(BaseModel):
    """Response for task status query."""

    task_id: str = Field(..., description="ID of the background task")
    status: str = Field(..., description="Current status of the task")
    result: dict[str, Any] | None = Field(
        None, description="Task result if completed (PipelineRunResponse)"
    )


class PipelineRunResponse(BaseModel):
    """Response from pipeline execution."""

    status: str = Field(
        ...,
        description="Execution status: success, failed, or partial",
    )
    records_processed: int = Field(
        ...,
        ge=0,
        description="Number of records successfully processed",
    )
    records_failed: int = Field(
        ...,
        ge=0,
        description="Number of records that failed processing",
    )
    duration_seconds: float = Field(
        ...,
        ge=0,
        description="Total execution time in seconds",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Error messages if any failures occurred",
    )
