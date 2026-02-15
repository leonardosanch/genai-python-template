"""DTOs for Spark pipeline operations."""

from pydantic import BaseModel, Field

from src.application.dtos.pipeline import PipelineRunResponse


class SparkJobRequest(BaseModel):
    """Request to run a Spark pipeline."""

    pipeline_name: str = Field(..., description="Registered pipeline name to execute")
    jdbc_url: str = Field(default="", description="JDBC connection URL")
    source_table: str = Field(default="", description="Source table name")
    output_path: str = Field(default="", description="Output path for results")
    output_format: str = Field(
        default="parquet", description="Output format (parquet, csv, json, delta)"
    )
    partition_by: list[str] = Field(
        default_factory=list, description="Columns to partition output by"
    )


class SparkJobResponse(PipelineRunResponse):
    """Response from Spark pipeline execution."""

    pipeline_name: str = Field(..., description="Pipeline that was executed")
