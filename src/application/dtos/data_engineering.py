"""DTOs for data engineering operations."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DataQualityReportResponse(BaseModel):
    """Response for a data quality validation report."""

    is_valid: bool
    total_records: int = Field(ge=0)
    valid_records: int = Field(ge=0)
    invalid_records: int = Field(ge=0)
    errors: list[str] = Field(default_factory=list)


class ETLRunRequest(BaseModel):
    """Request to run an ETL pipeline."""

    source_uri: str = Field(..., description="URI of the data source")
    sink_uri: str = Field(..., description="URI of the data destination")
    run_validation: bool = Field(
        default=True, description="Whether to validate data before loading"
    )
    schema_name: str | None = Field(default=None, description="Schema to validate against")


class ETLRunResponse(BaseModel):
    """Response from an ETL pipeline execution."""

    status: str = Field(..., description="Execution status: success, failed, or partial")
    records_extracted: int = Field(default=0, ge=0)
    records_transformed: int = Field(default=0, ge=0)
    records_loaded: int = Field(default=0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0)
    quality_result: DataQualityReportResponse | None = None
    errors: list[str] = Field(default_factory=list)
