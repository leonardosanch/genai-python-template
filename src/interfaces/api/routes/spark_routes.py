"""Spark pipeline API routes."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request

from src.application.dtos.spark import SparkJobRequest, SparkJobResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/spark", tags=["spark"])

# Registry of available Spark pipelines — extend as needed
_PIPELINE_REGISTRY: dict[str, type[Any]] = {}


def _get_pipeline(request: SparkJobRequest, app_request: Request) -> Any:
    """Resolve and instantiate a Spark pipeline by name."""
    try:
        from src.application.pipelines.sp_migration_example import (
            SPMigrationConfig,
            SPMigrationExamplePipeline,
        )
        from src.infrastructure.data.spark.session_manager import SparkSessionManager

        settings = app_request.app.state.container._settings

        if request.pipeline_name == "sp_migration_example":
            spark = SparkSessionManager.get_or_create(settings.spark)
            config = SPMigrationConfig(
                jdbc_url=request.jdbc_url,
                source_table=request.source_table,
                output_path=request.output_path,
                output_format=request.output_format,
                partition_by=tuple(request.partition_by),
            )
            return SPMigrationExamplePipeline(spark=spark, config=config)
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="PySpark is not installed. Install with: uv pip install pyspark",
        )

    # Check custom registry
    pipeline_cls = _PIPELINE_REGISTRY.get(request.pipeline_name)
    if pipeline_cls is None:
        raise HTTPException(
            status_code=404,
            detail=f"Pipeline '{request.pipeline_name}' not found",
        )
    return pipeline_cls(request, app_request)


@router.post("/run", response_model=SparkJobResponse)
async def run_spark_pipeline(
    request: SparkJobRequest,
    app_request: Request,
) -> SparkJobResponse:
    """Execute a Spark pipeline synchronously (via thread)."""
    pipeline = _get_pipeline(request, app_request)
    result = await pipeline.run_async()
    return SparkJobResponse(
        pipeline_name=request.pipeline_name,
        status=result.status,
        records_processed=result.records_processed,
        records_failed=result.records_failed,
        duration_seconds=result.duration_seconds,
        errors=result.errors,
    )


@router.post("/run/async", response_model=dict[str, str])
async def submit_spark_pipeline(
    request: SparkJobRequest,
    app_request: Request,
) -> dict[str, str]:
    """Submit a Spark pipeline to Celery for background execution."""
    try:
        import importlib

        mod = importlib.import_module("src.infrastructure.tasks.celery_app")
        run_task = getattr(mod, "run_spark_pipeline_task")
    except (ImportError, AttributeError):
        raise HTTPException(
            status_code=501,
            detail="Celery is not configured for async Spark jobs",
        )

    task = run_task.delay(request.model_dump())
    return {"task_id": task.id, "status": "submitted"}
