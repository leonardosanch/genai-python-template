"""Data engineering API routes — ETL and validation endpoints."""

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from src.application.dtos.data_engineering import (
    DataQualityReportResponse,
    ETLRunRequest,
    ETLRunResponse,
)
from src.application.use_cases.run_etl import RunETLUseCase
from src.application.use_cases.validate_dataset import ValidateDatasetUseCase
from src.infrastructure.container import Container

router = APIRouter(prefix="/api/v1/data", tags=["data-engineering"])


def get_container(request: Request) -> Container:
    """Get dependency injection container from app state."""
    container: Container = request.app.state.container
    return container


@router.post("/etl", response_model=ETLRunResponse)
async def run_etl(
    request: ETLRunRequest,
    container: Container = Depends(get_container),
) -> ETLRunResponse:
    """Run an ETL pipeline: extract → validate → load."""
    use_case = RunETLUseCase(
        source=container.data_source,
        sink=container.data_sink,
        validator=container.data_validator,
        event_bus=container.event_bus,
    )
    return await use_case.execute(request)


class ValidateRequest(BaseModel):
    """Request body for dataset validation."""

    source_uri: str = Field(..., description="URI of the data source to validate")


@router.post("/validate", response_model=DataQualityReportResponse)
async def validate_dataset(
    request: ValidateRequest,
    container: Container = Depends(get_container),
) -> DataQualityReportResponse:
    """Validate a dataset and return a quality report."""
    use_case = ValidateDatasetUseCase(
        source=container.data_source,
        validator=container.data_validator,
        event_bus=container.event_bus,
    )
    return await use_case.execute(source_uri=request.source_uri)
